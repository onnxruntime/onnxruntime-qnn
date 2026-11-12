# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Unit tests for ``ort_qnn_inference_engine``.

Ported from the legacy ``test_ort_qnn_inference_engine.py`` and extended with the QNN EP registration path (driven
by a fake ``onnxruntime_qnn`` module), real ORT CPU execution and QNN HTP execution.
"""

from __future__ import annotations

import logging
import pickle
import sys
import types
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("onnxruntime")

import numpy as np
import onnx
import ort_qnn_inference_engine as ie
from onnx import TensorProto, helper
from ort_qnn_inference_engine import (
    OrtQnnInferenceEngine,
    OrtQnnInferenceEngineInputConfig,
    OrtQnnInferenceEngineOutputConfig,
    create_session,
    ensure_qnn_ep_registered,
    get_default_engine_logger,
    run_inference_job,
)
from pydantic import ValidationError

import onnxruntime as ort

_MODULE = "ort_qnn_inference_engine"
_FAKE_LIB = "/fake/libonnxruntime_providers_qnn.so"
_FAKE_HTP = "/fake/libQnnHtp.so"


# --------------------------------------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def sample_input() -> dict[str, np.ndarray]:
    return {"input": np.random.default_rng(0).standard_normal((1, 3, 8, 8)).astype(np.float32)}


@pytest.fixture
def touched_model(tmp_path: Path) -> Path:
    path = tmp_path / "model.onnx"
    path.touch()
    return path


@pytest.fixture
def mock_validate() -> Iterator[MagicMock]:
    with patch(f"{_MODULE}.validate_model") as mock:
        mock.return_value = None
        yield mock


@pytest.fixture(autouse=True)
def _clean_registration_state(monkeypatch: pytest.MonkeyPatch, request) -> Iterator[None]:
    """Isolates the module-level registration cache."""
    if request.node.get_closest_marker("qnn"):
        # Keep the real ORT registration in sync with the Python cache. Clearing
        # only the cache makes the capability probe try to register twice.
        yield
        return
    ensure_qnn_ep_registered.cache_clear()
    saved = set(ie._REGISTERED)
    ie._REGISTERED.clear()
    yield
    ensure_qnn_ep_registered.cache_clear()
    ie._REGISTERED.clear()
    ie._REGISTERED.update(saved)


class _FakeDevice:
    def __init__(self, ep_name: str) -> None:
        self.ep_name = ep_name


class _FakeOutput:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeQnnEnv:
    """Fake ``onnxruntime_qnn`` module plus recorders for the patched ORT entry points."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.events: list[tuple] = []
        self.devices: list[_FakeDevice] = [_FakeDevice("QNNExecutionProvider"), _FakeDevice("CPUExecutionProvider")]
        self.register_error: Exception | None = None
        self.output_names = ["output"]
        self.results = [np.ones((1, 4), dtype=np.uint8)]
        self.session_options: list[MagicMock] = []
        self.session_kwargs: list[dict] = []

        module = types.ModuleType("onnxruntime_qnn")
        module.get_ep_name = lambda: "QNNExecutionProvider"
        module.get_library_path = lambda: _FAKE_LIB
        module.get_qnn_htp_path = lambda: _FAKE_HTP
        self.module = module
        monkeypatch.setitem(sys.modules, "onnxruntime_qnn", module)
        monkeypatch.setattr(ort, "register_execution_provider_library", self._register, raising=False)
        monkeypatch.setattr(ort, "unregister_execution_provider_library", self._unregister, raising=False)
        monkeypatch.setattr(ort, "get_ep_devices", self._get_ep_devices, raising=False)
        monkeypatch.setattr(ort, "SessionOptions", self._session_options)
        monkeypatch.setattr(ort, "InferenceSession", self._inference_session)

    def _register(self, name: str, path: str) -> None:
        self.events.append(("register", name, path))
        if self.register_error is not None:
            raise self.register_error

    def _unregister(self, name: str) -> None:
        self.events.append(("unregister", name))

    def _get_ep_devices(self) -> list[_FakeDevice]:
        self.events.append(("get_ep_devices",))
        return self.devices

    def _session_options(self) -> MagicMock:
        options = MagicMock(name="SessionOptions")
        self.session_options.append(options)
        return options

    def _inference_session(self, src: object, **kwargs: object) -> MagicMock:
        self.events.append(("session",))
        self.session_kwargs.append({"src": src, **kwargs})
        session = MagicMock(name="InferenceSession")
        session.get_outputs.return_value = [_FakeOutput(n) for n in self.output_names]

        def _run(output_names: object, feeds: object) -> list[np.ndarray]:
            self.events.append(("run",))
            return self.results

        session.run.side_effect = _run
        return session

    def names(self) -> list[str]:
        return [event[0] for event in self.events]


@pytest.fixture
def fake_qnn(monkeypatch: pytest.MonkeyPatch) -> _FakeQnnEnv:
    return _FakeQnnEnv(monkeypatch)


def _tiny_qdq_model() -> onnx.ModelProto:
    nodes = [
        helper.make_node("QuantizeLinear", ["input", "s", "zp"], ["input_q"], name="input_q"),
        helper.make_node("DequantizeLinear", ["input_q", "s", "zp"], ["output"], name="input_dq"),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor("s", TensorProto.FLOAT, [], [0.1]), helper.make_tensor("zp", TensorProto.UINT8, [], [0])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = 9
    return model


# --------------------------------------------------------------------------------------------------------------------
# Input config (legacy #1-5, plus BUG-IE-1)
# --------------------------------------------------------------------------------------------------------------------
class TestInputConfig:
    def test_valid_config_creation(self, tmp_path, touched_model, sample_input, mock_validate):
        config = OrtQnnInferenceEngineInputConfig(
            qdq_model=touched_model,
            input_sample=sample_input,
            outputs_list=["layer1", "layer2"],
            use_cpu=True,
            dump_output_tensors=True,
            working_directory=tmp_path,
        )
        assert config.qdq_model == touched_model
        assert config.input_sample is sample_input or config.input_sample.keys() == sample_input.keys()
        assert config.outputs_list == ["layer1", "layer2"]
        assert config.use_cpu is True
        assert config.dump_output_tensors is True
        assert config.working_directory == tmp_path
        mock_validate.assert_called_once_with(touched_model, validate_qdq=True)

    def test_default_values(self, tmp_path, touched_model, sample_input, mock_validate):
        config = OrtQnnInferenceEngineInputConfig(
            qdq_model=touched_model, input_sample=sample_input, working_directory=tmp_path
        )
        assert config.outputs_list is None
        assert config.use_cpu is False
        assert config.dump_output_tensors is False
        assert config.disable_cpu_fallback is False
        assert config.working_directory == tmp_path

    def test_invalid_model_path(self, tmp_path, sample_input, mock_validate):
        with pytest.raises(ValidationError):
            OrtQnnInferenceEngineInputConfig(
                qdq_model=tmp_path / "nonexistent.onnx", input_sample=sample_input, working_directory=tmp_path
            )
        mock_validate.assert_not_called()

    def test_model_validation_called(self, tmp_path, touched_model, sample_input, mock_validate):
        OrtQnnInferenceEngineInputConfig(qdq_model=touched_model, input_sample=sample_input, working_directory=tmp_path)
        mock_validate.assert_called_once_with(touched_model, validate_qdq=True)

    def test_model_validation_failure(self, tmp_path, touched_model, sample_input, mock_validate):
        mock_validate.side_effect = ValueError("Invalid QDQ model")
        with pytest.raises(ValueError, match="Invalid QDQ model"):
            OrtQnnInferenceEngineInputConfig(
                qdq_model=touched_model, input_sample=sample_input, working_directory=tmp_path
            )

    def test_working_directory_is_required(self, touched_model, sample_input, mock_validate):
        with pytest.raises(ValidationError, match="working_directory"):
            OrtQnnInferenceEngineInputConfig(qdq_model=touched_model, input_sample=sample_input)

    def test_extra_field_forbidden(self, tmp_path, touched_model, sample_input, mock_validate):
        with pytest.raises(ValidationError):
            OrtQnnInferenceEngineInputConfig(
                qdq_model=touched_model, input_sample=sample_input, working_directory=tmp_path, bogus=1
            )

    def test_real_validation_rejects_fp32_model(self, tmp_path, fp32_chain_model, sample_input):
        with pytest.raises(ValueError, match="Model doesn't contain QDQ nodes"):
            OrtQnnInferenceEngineInputConfig(
                qdq_model=fp32_chain_model, input_sample=sample_input, working_directory=tmp_path
            )

    @pytest.mark.parametrize("use_cpu", [False, True])
    def test_rejects_cached_context(self, tmp_path, cached_context_model, sample_input, use_cpu):
        with pytest.raises(ValueError, match="EPContext models are not supported"):
            OrtQnnInferenceEngineInputConfig(
                qdq_model=cached_context_model,
                input_sample=sample_input,
                working_directory=tmp_path,
                use_cpu=use_cpu,
            )

    def test_config_is_picklable(self, tmp_path, qdq_chain_model, qdq_input_sample):
        config = OrtQnnInferenceEngineInputConfig(
            qdq_model=qdq_chain_model,
            input_sample=qdq_input_sample,
            outputs_list=["conv1_out_dq"],
            working_directory=tmp_path,
        )
        clone = pickle.loads(pickle.dumps(config))
        assert clone.qdq_model == config.qdq_model
        assert clone.outputs_list == ["conv1_out_dq"]
        np.testing.assert_array_equal(clone.input_sample["input"], qdq_input_sample["input"])


# --------------------------------------------------------------------------------------------------------------------
# Output config (legacy #6-8)
# --------------------------------------------------------------------------------------------------------------------
class TestOutputConfig:
    @pytest.fixture
    def sample_output(self) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(1)
        return {
            "output1": rng.standard_normal((1, 1000)).astype(np.float32),
            "output2": rng.standard_normal((1, 512)).astype(np.float32),
        }

    def test_output_config_with_data(self, tmp_path, sample_output):
        config = OrtQnnInferenceEngineOutputConfig(output_data=sample_output, output_dir=tmp_path)
        assert config.output_data == sample_output
        assert config.output_dir == tmp_path

    def test_output_config_with_none_values(self):
        config = OrtQnnInferenceEngineOutputConfig()
        assert config.output_data is None
        assert config.output_dir is None

    def test_output_config_partial_data(self, tmp_path, sample_output):
        config1 = OrtQnnInferenceEngineOutputConfig(output_data=sample_output)
        assert config1.output_data == sample_output
        assert config1.output_dir is None
        config2 = OrtQnnInferenceEngineOutputConfig(output_dir=tmp_path)
        assert config2.output_data is None
        assert config2.output_dir == tmp_path

    def test_output_config_is_picklable(self, tmp_path, sample_output):
        clone = pickle.loads(pickle.dumps(OrtQnnInferenceEngineOutputConfig(output_data=sample_output)))
        np.testing.assert_array_equal(clone.output_data["output1"], sample_output["output1"])


# --------------------------------------------------------------------------------------------------------------------
# Engine construction and dumping (legacy #9-12)
# --------------------------------------------------------------------------------------------------------------------
class TestEngine:
    def test_init_with_default_logger(self):
        engine = OrtQnnInferenceEngine()
        assert engine.logger is logging.getLogger("OrtQnnInference")
        assert engine.logger.level == logging.INFO
        marked = [h for h in engine.logger.handlers if getattr(h, "_adbg_handler", False)]
        assert len(marked) == 1
        # Idempotent: a second engine does not attach another handler.
        OrtQnnInferenceEngine()
        assert len([h for h in engine.logger.handlers if getattr(h, "_adbg_handler", False)]) == 1
        assert get_default_engine_logger() is engine.logger

    def test_init_with_custom_logger(self, mock_logger):
        engine = OrtQnnInferenceEngine(logger=mock_logger)
        assert engine.logger is mock_logger

    def test_dump_inference_outputs(self, tmp_path, mock_logger):
        rng = np.random.default_rng(2)
        outputs = {
            "output1": rng.standard_normal((1, 1000)).astype(np.float32),
            "output2": rng.standard_normal((1, 512)).astype(np.float32),
        }
        engine = OrtQnnInferenceEngine(logger=mock_logger)
        with patch(f"{_MODULE}.sanitize_tensor_name", side_effect=lambda x: x.replace("/", "_")) as mock_sanitize:
            output_dir = engine._dump_inference_outputs(outputs, tmp_path)
        assert output_dir.exists()
        assert output_dir.name == "target_outputs"
        assert len(list(output_dir.glob("*.raw"))) == 2
        assert mock_sanitize.call_count == 2
        np.testing.assert_array_equal(
            np.fromfile(output_dir / "output1.raw", dtype=np.float32).reshape(1, 1000), outputs["output1"]
        )

    def test_dump_inference_outputs_long_filename(self, tmp_path, mock_logger):
        rng = np.random.default_rng(3)
        outputs = {
            "output1": rng.standard_normal((1, 100)).astype(np.float32),
            "a" * 300: rng.standard_normal((1, 100)).astype(np.float32),
        }
        engine = OrtQnnInferenceEngine(logger=mock_logger)
        with patch(f"{_MODULE}.sanitize_tensor_name", side_effect=lambda x: x):
            output_dir = engine._dump_inference_outputs(outputs, tmp_path)
        mock_logger.warning.assert_called_once()
        assert "exceeds 255 bytes" in mock_logger.warning.call_args[0][0]
        assert len(list(output_dir.glob("*.raw"))) == 1

    def test_dump_length_check_uses_sanitized_name(self, tmp_path, mock_logger):
        # 251 bytes raw ("/" is 1 byte), but the sanitised name gains a leading "_" -> 252 + ".raw" = 256 > 255 (B11).
        name = "1" + "a" * 250
        engine = OrtQnnInferenceEngine(logger=mock_logger)
        output_dir = engine._dump_inference_outputs({name: np.zeros(2, dtype=np.float32)}, tmp_path)
        assert list(output_dir.glob("*.raw")) == []
        assert "exceeds 255 bytes" in mock_logger.warning.call_args[0][0]

    def test_dump_sanitizes_names(self, tmp_path, mock_logger):
        engine = OrtQnnInferenceEngine(logger=mock_logger)
        output_dir = engine._dump_inference_outputs({"/conv1/Conv_output_0": np.zeros(2, np.float32)}, tmp_path)
        assert output_dir == tmp_path / "target_outputs"
        assert [p.name for p in output_dir.glob("*.raw")] == ["_conv1_Conv_output_0.raw"]


# --------------------------------------------------------------------------------------------------------------------
# Engine.run with a mocked session (replaces the 7 commented-out legacy tests)
# --------------------------------------------------------------------------------------------------------------------
class TestRun:
    @pytest.fixture
    def config_factory(self, tmp_path, qdq_chain_model, qdq_input_sample):
        def _make(**overrides: object) -> OrtQnnInferenceEngineInputConfig:
            kwargs: dict[str, object] = {
                "qdq_model": qdq_chain_model,
                "input_sample": qdq_input_sample,
                "working_directory": tmp_path,
            }
            kwargs.update(overrides)
            return OrtQnnInferenceEngineInputConfig(**kwargs)

        return _make

    def test_run_without_intermediate_outputs_warns(self, fake_qnn, config_factory, mock_logger):
        result = OrtQnnInferenceEngine(mock_logger).run(config_factory())
        warnings = [c[0][0] for c in mock_logger.warning.call_args_list]
        assert any("No intermediate outputs specified" in w for w in warnings)
        assert list(result.output_data) == ["output"]
        assert result.output_dir is None

    def test_run_output_ordering_and_dedup(self, fake_qnn, config_factory, mock_logger):
        added: list[list[str]] = []
        real_add = ie.add_graph_outputs

        def _spy(model: onnx.ModelProto, names: list[str]) -> list[str]:
            added.append(list(names))
            final = real_add(model, names)
            fake_qnn.output_names = final
            fake_qnn.results = [np.full((1,), i, dtype=np.int32) for i in range(len(final))]
            return final

        with patch(f"{_MODULE}.add_graph_outputs", side_effect=_spy):
            result = OrtQnnInferenceEngine(mock_logger).run(
                config_factory(outputs_list=["conv1_out_dq", "output", "conv1_out_dq", "input_dq"])
            )
        assert added == [["conv1_out_dq", "output", "conv1_out_dq", "input_dq"]]
        assert list(result.output_data) == ["conv1_out_dq", "output", "input_dq"]
        infos = [c[0][0] for c in mock_logger.info.call_args_list]
        assert "Adding 4 intermediate output nodes to the model" in infos

    def test_run_casts_outputs_to_float32(self, fake_qnn, config_factory, mock_logger):
        fake_qnn.results = [np.array([[1, 2, 255]], dtype=np.uint8)]
        result = OrtQnnInferenceEngine(mock_logger).run(config_factory())
        assert result.output_data["output"].dtype == np.float32
        np.testing.assert_array_equal(result.output_data["output"], [[1.0, 2.0, 255.0]])

    def test_run_dumps_outputs(self, fake_qnn, config_factory, mock_logger, tmp_path):
        fake_qnn.output_names = ["/a/b"]
        fake_qnn.results = [np.arange(4, dtype=np.float32)]
        result = OrtQnnInferenceEngine(mock_logger).run(config_factory(dump_output_tensors=True))
        assert result.output_dir == tmp_path / "target_outputs"
        np.testing.assert_array_equal(np.fromfile(result.output_dir / "_a_b.raw", dtype=np.float32), np.arange(4))

    def test_run_htp_registration_order(self, fake_qnn, config_factory, mock_logger):
        OrtQnnInferenceEngine(mock_logger).run(config_factory())
        assert fake_qnn.names() == ["register", "get_ep_devices", "session", "run", "unregister"]
        assert fake_qnn.events[0] == ("register", "QNNExecutionProvider", _FAKE_LIB)
        assert fake_qnn.events[-1] == ("unregister", "QNNExecutionProvider")
        options = fake_qnn.session_options[-1]
        devices, ep_options = options.add_provider_for_devices.call_args[0]
        assert [d.ep_name for d in devices] == ["QNNExecutionProvider"]
        assert ep_options == {"backend_path": _FAKE_HTP}
        options.add_session_config_entry.assert_not_called()
        assert "providers" not in fake_qnn.session_kwargs[-1]
        assert set() == ie._REGISTERED

    def test_run_use_cpu_needs_no_qnn_plugin(self, fake_qnn, config_factory, mock_logger, monkeypatch):
        monkeypatch.setitem(sys.modules, "onnxruntime_qnn", None)
        result = OrtQnnInferenceEngine(mock_logger).run(config_factory(use_cpu=True))
        assert fake_qnn.names() == ["session", "run"]
        assert fake_qnn.session_kwargs[-1]["providers"] == ["CPUExecutionProvider"]
        fake_qnn.session_options[-1].add_provider_for_devices.assert_not_called()
        assert result.cpu_fallback_reason == ""
        assert set() == ie._REGISTERED

    def test_run_disable_cpu_fallback(self, fake_qnn, config_factory, mock_logger):
        OrtQnnInferenceEngine(mock_logger).run(config_factory(disable_cpu_fallback=True))
        fake_qnn.session_options[-1].add_session_config_entry.assert_called_once_with(
            "session.disable_cpu_ep_fallback", "1"
        )

    def test_run_unregisters_when_session_run_fails(self, fake_qnn, config_factory, mock_logger, monkeypatch):
        def _failing_session(src: object, **kwargs: object) -> MagicMock:
            fake_qnn.events.append(("session",))
            session = MagicMock()
            session.run.side_effect = RuntimeError("boom")
            return session

        monkeypatch.setattr(ort, "InferenceSession", _failing_session)
        with pytest.raises(RuntimeError, match="boom"):
            OrtQnnInferenceEngine(mock_logger).run(config_factory())
        assert fake_qnn.events[-1] == ("unregister", "QNNExecutionProvider")
        assert set() == ie._REGISTERED
        # A second run registers again (no stale registration left behind, BUG-IE-3).
        monkeypatch.setattr(ort, "InferenceSession", fake_qnn._inference_session)
        OrtQnnInferenceEngine(mock_logger).run(config_factory())
        assert [e for e in fake_qnn.events if e[0] == "register"] == [
            ("register", "QNNExecutionProvider", _FAKE_LIB)
        ] * 2

    def test_run_device_not_found(self, fake_qnn, config_factory, mock_logger):
        fake_qnn.devices = [_FakeDevice("CPUExecutionProvider")]
        with pytest.raises(RuntimeError, match=r"^QNN EP device not found\.$"):
            OrtQnnInferenceEngine(mock_logger).run(config_factory())
        assert fake_qnn.events[-1] == ("unregister", "QNNExecutionProvider")

    def test_run_registration_failure(self, fake_qnn, config_factory, mock_logger):
        fake_qnn.register_error = RuntimeError("QNN EP requires ORT >= 1.24 (API 24)")
        with pytest.raises(RuntimeError, match=r"^Failed to register QNN EP library: QNN EP requires ORT"):
            OrtQnnInferenceEngine(mock_logger).run(config_factory())
        assert "unregister" not in fake_qnn.names()

    def test_run_keeps_persistent_registration(self, fake_qnn, config_factory, mock_logger):
        assert ensure_qnn_ep_registered() == "QNNExecutionProvider"
        OrtQnnInferenceEngine(mock_logger).run(config_factory())
        assert fake_qnn.names().count("register") == 1
        assert "unregister" not in fake_qnn.names()

    def test_run_without_onnxruntime_qnn(self, monkeypatch, config_factory, mock_logger):
        monkeypatch.setitem(sys.modules, "onnxruntime_qnn", None)
        with pytest.raises(RuntimeError, match="onnxruntime_qnn is not installed"):
            OrtQnnInferenceEngine(mock_logger).run(config_factory())

    def test_run_inference_job(self, fake_qnn, config_factory):
        result = run_inference_job(config_factory(), "adbg_test")
        assert isinstance(result, OrtQnnInferenceEngineOutputConfig)
        assert list(result.output_data) == ["output"]

    @pytest.mark.parametrize("traced", [True, False])
    def test_compile_only_traces_without_inference(self, fake_qnn, config_factory, mock_logger, traced):
        compiled: list[list[str]] = []
        real_create_session = ie.create_session

        def _create_session(model: onnx.ModelProto, *args: object, **kwargs: object) -> object:
            compiled.append([output.name for output in model.graph.output])
            if traced:
                (kwargs["op_trace_dir"] / "qnn_op_trace.json").write_text("{}")
            return real_create_session(model, *args, **kwargs)

        with patch(f"{_MODULE}.create_session", side_effect=_create_session):
            result = OrtQnnInferenceEngine(mock_logger).run(
                config_factory(input_sample={}, enable_framework_op_trace=True, compile_only=True)
            )
        assert fake_qnn.names() == ["register", "get_ep_devices", "session", "unregister"]
        assert compiled == [["output"]]
        assert result.output_data is None
        assert (result.op_trace_file is not None) == traced
        if traced:
            assert result.op_trace_file.name == "qnn_op_trace.json"
        # The caller reports a missing trace; the engine does not warn about it or about the missing outputs.
        mock_logger.warning.assert_not_called()


# --------------------------------------------------------------------------------------------------------------------
# Registration and create_session
# --------------------------------------------------------------------------------------------------------------------
class TestRegistration:
    def test_ensure_registered_once(self, fake_qnn):
        assert ensure_qnn_ep_registered() == "QNNExecutionProvider"
        assert ensure_qnn_ep_registered() == "QNNExecutionProvider"
        assert fake_qnn.events == [("register", "QNNExecutionProvider", _FAKE_LIB)]

    def test_ensure_registered_failure_is_retried(self, fake_qnn):
        fake_qnn.register_error = ValueError("bad api")
        with pytest.raises(RuntimeError, match=r"^Failed to register QNN EP library: bad api"):
            ensure_qnn_ep_registered()
        fake_qnn.register_error = None
        assert ensure_qnn_ep_registered() == "QNNExecutionProvider"
        assert fake_qnn.names().count("register") == 2

    def test_ensure_registered_import_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "onnxruntime_qnn", None)
        with pytest.raises(RuntimeError, match=r"^onnxruntime_qnn is not installed; it is required to run on QNN EP\."):
            ensure_qnn_ep_registered()

    def test_create_session_htp(self, fake_qnn):
        session = create_session(_tiny_qdq_model())
        assert session is not None
        devices, ep_options = fake_qnn.session_options[-1].add_provider_for_devices.call_args[0]
        assert [d.ep_name for d in devices] == ["QNNExecutionProvider"]
        assert ep_options == {"backend_path": _FAKE_HTP}
        assert isinstance(fake_qnn.session_kwargs[-1]["src"], bytes)

    def test_create_session_passes_all_matching_devices(self, fake_qnn):
        fake_qnn.devices = [_FakeDevice("QNNExecutionProvider"), _FakeDevice("QNNExecutionProvider")]
        create_session(_tiny_qdq_model())
        devices, _ = fake_qnn.session_options[-1].add_provider_for_devices.call_args[0]
        assert len(devices) == 2

    def test_create_session_device_not_found(self, fake_qnn):
        fake_qnn.devices = []
        with pytest.raises(RuntimeError, match=r"^QNN EP device not found\.$"):
            create_session(_tiny_qdq_model())

    def test_create_session_cpu_ignores_qnn_fallback_option(self, fake_qnn, monkeypatch):
        monkeypatch.setitem(sys.modules, "onnxruntime_qnn", None)
        create_session(_tiny_qdq_model(), use_cpu=True, disable_cpu_fallback=True)
        options = fake_qnn.session_options[-1]
        assert fake_qnn.names() == ["session"]
        assert fake_qnn.session_kwargs[-1]["providers"] == ["CPUExecutionProvider"]
        options.add_provider_for_devices.assert_not_called()
        options.add_session_config_entry.assert_not_called()

    @pytest.mark.parametrize("use_cpu", [False, True])
    def test_create_session_rejects_cached_context(self, fake_qnn, cached_context_model, use_cpu):
        with pytest.raises(ValueError, match="EPContext models are not supported"):
            create_session(onnx.load(cached_context_model), use_cpu=use_cpu)
        assert fake_qnn.names() == []

    @pytest.mark.parametrize("use_cpu", [False, True])
    def test_create_session_rejects_fp32_model(self, fake_qnn, fp32_chain_model, use_cpu):
        with pytest.raises(ValueError, match="Model doesn't contain QDQ nodes"):
            create_session(onnx.load(fp32_chain_model), use_cpu=use_cpu)
        assert fake_qnn.names() == []


# --------------------------------------------------------------------------------------------------------------------
# Real ORT runs
# --------------------------------------------------------------------------------------------------------------------
class TestRealCpu:
    def test_run_on_ort_cpu(self, monkeypatch, tmp_path, qdq_chain_model, qdq_input_sample, logger):
        monkeypatch.setitem(sys.modules, "onnxruntime_qnn", None)
        config = OrtQnnInferenceEngineInputConfig(
            qdq_model=qdq_chain_model,
            input_sample=qdq_input_sample,
            outputs_list=["input_dq", "conv1_out_dq"],
            dump_output_tensors=True,
            working_directory=tmp_path,
            use_cpu=True,
        )
        result = OrtQnnInferenceEngine(logger).run(config)
        assert list(result.output_data) == ["input_dq", "conv1_out_dq", "output"]
        for array in result.output_data.values():
            assert array.dtype == np.float32
        assert result.output_data["conv1_out_dq"].shape == (1, 4, 8, 8)
        # input_dq is the fake-quantized input: within half a quantization step of the original.
        np.testing.assert_allclose(result.output_data["input_dq"], qdq_input_sample["input"], atol=0.5 / 255 + 1e-6)
        assert sorted(p.name for p in result.output_dir.glob("*.raw")) == [
            "conv1_out_dq.raw",
            "input_dq.raw",
            "output.raw",
        ]

    def test_run_inference_job_in_spawn_pool(self, tmp_path, qdq_chain_model, qdq_input_sample):
        import multiprocessing  # noqa: PLC0415
        from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

        config = OrtQnnInferenceEngineInputConfig(
            qdq_model=qdq_chain_model,
            input_sample=qdq_input_sample,
            outputs_list=["conv1_out_dq"],
            working_directory=tmp_path,
            use_cpu=True,
        )
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
            result = pool.submit(run_inference_job, config, "adbg_test").result(timeout=300)
        assert list(result.output_data) == ["conv1_out_dq", "output"]

    def test_run_on_cpu_execution_provider(self, tmp_path, qdq_chain_model, qdq_input_sample, logger, monkeypatch):
        monkeypatch.setitem(sys.modules, "onnxruntime_qnn", None)
        config = OrtQnnInferenceEngineInputConfig(
            qdq_model=qdq_chain_model,
            input_sample=qdq_input_sample,
            use_cpu=True,
            disable_cpu_fallback=True,
            working_directory=tmp_path,
        )
        result = OrtQnnInferenceEngine(logger).run(config)
        session = ort.InferenceSession(str(qdq_chain_model), providers=["CPUExecutionProvider"])
        expected = session.run(None, qdq_input_sample)[0]
        assert result.output_data["output"].shape == (1, 4, 8, 8)
        np.testing.assert_array_equal(result.output_data["output"], expected)
        assert result.cpu_fallback_reason == ""


@pytest.mark.qnn
class TestRealQnn:
    def test_run_on_qnn_htp(self, tmp_path, qdq_chain_model, qdq_input_sample, logger):
        from conftest import requires_qnn_ep  # noqa: PLC0415

        if not requires_qnn_ep():
            pytest.skip("QNN EP (onnxruntime_qnn) is not available")
        config = OrtQnnInferenceEngineInputConfig(
            qdq_model=qdq_chain_model,
            input_sample=qdq_input_sample,
            outputs_list=["conv1_out_dq"],
            dump_output_tensors=True,
            working_directory=tmp_path,
        )
        result = OrtQnnInferenceEngine(logger).run(config)
        assert list(result.output_data) == ["conv1_out_dq", "output"]
        assert result.output_data["output"].dtype == np.float32
        assert result.output_data["output"].shape == (1, 4, 8, 8)

        assert (result.output_dir / "output.raw").exists()

    def test_run_returns_q_output_with_its_dq_output(self, tmp_path, qdq_chain_model, qdq_input_sample, logger):
        # QNN EP alone fails with "Inconsistent output sizes" here, and ORT reruns the model on CPU.
        from conftest import _S_RELU, _ZP_RELU, requires_qnn_ep  # noqa: PLC0415

        if not requires_qnn_ep():
            pytest.skip("QNN EP (onnxruntime_qnn) is not available")
        config = OrtQnnInferenceEngineInputConfig(
            qdq_model=qdq_chain_model,
            input_sample=qdq_input_sample,
            outputs_list=["relu1_out_q"],
            working_directory=tmp_path,
        )
        result = OrtQnnInferenceEngine(logger).run(config)
        assert result.cpu_fallback_reason == ""
        assert list(result.output_data) == ["relu1_out_q", "output"]
        expected = (result.output_data["relu1_out_q"] - np.float32(_ZP_RELU)) * _S_RELU
        np.testing.assert_allclose(result.output_data["output"], expected, rtol=1e-6)


@pytest.mark.parametrize("use_cpu", [False, True])
def test_framework_trace_provider_options(fake_qnn, tmp_path, use_cpu):
    trace_dir = tmp_path / "new_trace"
    create_session(_tiny_qdq_model(), use_cpu=use_cpu, op_trace_dir=trace_dir, dump_qnn_ep_input_graph=True)
    if use_cpu:
        assert fake_qnn.names() == ["session"]
        assert not trace_dir.exists()
        fake_qnn.session_options[-1].add_provider_for_devices.assert_not_called()
        return
    assert trace_dir.is_dir()
    assert fake_qnn.session_options[-1].add_provider_for_devices.call_args.args[1] == {
        "backend_path": _FAKE_HTP,
        "enable_framework_op_trace": "1",
        "framework_op_trace_dir": str(trace_dir.resolve()),
        "dump_qnn_ep_input_graph": "1",
        "dump_qnn_ep_input_graph_dir": str(trace_dir.resolve()),
    }


def test_engine_returns_only_fresh_trace(fake_qnn, tmp_path, qdq_chain_model, qdq_input_sample, monkeypatch, logger):
    config = OrtQnnInferenceEngineInputConfig(
        qdq_model=qdq_chain_model,
        input_sample=qdq_input_sample,
        working_directory=tmp_path,
        enable_framework_op_trace=True,
    )
    original = ie.create_session
    directories = []

    def compile_with_trace(model, *args, **kwargs):
        session = original(model, *args, **kwargs)
        directories.append(kwargs["op_trace_dir"])
        if len(directories) == 1:
            (directories[-1] / "qnn_op_trace.json").write_text('{"schema_version":2,"subgraph_traces":[]}')
        return session

    monkeypatch.setattr(ie, "create_session", compile_with_trace)
    first = OrtQnnInferenceEngine(logger).run(config)
    second = OrtQnnInferenceEngine(logger).run(config)
    assert first.op_trace_file.is_file()
    assert second.op_trace_file is None
    assert directories[0] != directories[1]
    assert {path.parent for path in directories} == {(tmp_path / "op_trace").resolve()}
    assert pickle.loads(pickle.dumps(first)).op_trace_file == first.op_trace_file


def test_runtime_cpu_retry_is_reported(fake_qnn, tmp_path, qdq_chain_model, qdq_input_sample, logger, monkeypatch):
    config = OrtQnnInferenceEngineInputConfig(
        qdq_model=qdq_chain_model,
        input_sample=qdq_input_sample,
        working_directory=tmp_path,
    )
    session = MagicMock()
    session.get_providers.side_effect = [["QNNExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]]
    session.run.return_value = [np.ones((1, 4), np.float32)]
    session.get_outputs.return_value = [_FakeOutput("output")]
    monkeypatch.setattr(ie, "create_session", lambda *args, **kwargs: session)
    output = OrtQnnInferenceEngine(logger).run(config)
    assert "QNN HTP execution failure" in output.cpu_fallback_reason


@pytest.mark.parametrize(
    ("requested", "detached", "kept"),
    [
        (["conv1_out_q", "conv1_out_dq"], ["conv1_out_dq"], ["conv1_out_q", "output"]),
        (["relu1_out_q"], ["output"], ["relu1_out_q"]),
        (["conv1_out_dq", "input_q"], [], ["conv1_out_dq", "input_q", "output"]),
    ],
)
def test_detach_paired_dq_outputs(qdq_chain_model, requested, detached, kept):
    model = onnx.load(qdq_chain_model)
    ie.add_graph_outputs(model, requested)
    assert [node.output[0] for node in ie._detach_paired_dq_outputs(model)] == detached
    assert [value_info.name for value_info in model.graph.output] == kept


def test_detach_keeps_dq_with_runtime_scale():
    model = _tiny_qdq_model()
    # The scale depends on the input, so only the session can compute the DQ output.
    model.graph.node.insert(0, helper.make_node("ReduceMax", ["input"], ["s_runtime"], keepdims=0))
    model.graph.node[-1].input[1] = "s_runtime"
    ie.add_graph_outputs(model, ["input_q"])
    assert ie._detach_paired_dq_outputs(model) == []
    assert [value_info.name for value_info in model.graph.output] == ["input_q", "output"]


def _constant(output: str, data_type: int, value: float) -> onnx.NodeProto:
    return helper.make_node("Constant", [], [output], value=helper.make_tensor("value", data_type, [], [value]))


@pytest.mark.parametrize(
    "param_nodes",
    [
        pytest.param(
            [_constant("s_dq", TensorProto.FLOAT, 0.2), _constant("zp_dq", TensorProto.UINT8, 3)], id="constant"
        ),
        # PyTorch exports share quantization parameters through Identity nodes of their initializers.
        pytest.param(
            [helper.make_node("Identity", ["s"], ["s_dq"]), helper.make_node("Identity", ["zp"], ["zp_dq"])],
            id="identity",
        ),
        pytest.param(
            [
                _constant("s_64", TensorProto.DOUBLE, 0.2),
                helper.make_node("Cast", ["s_64"], ["s_32"], to=TensorProto.FLOAT),
                helper.make_node("Identity", ["s_32"], ["s_dq"]),
                helper.make_node("Identity", ["zp"], ["zp_dq"]),
            ],
            id="chain",
        ),
    ],
)
def test_dequantize_on_cpu_matches_ort(param_nodes):
    model = _tiny_qdq_model()
    # The nodes that compute the DQ's scale and zero point from initializers must be copied along with it.
    for index, node in enumerate(param_nodes):
        model.graph.node.insert(index, node)
    model.graph.node[-1].input[1] = "s_dq"
    model.graph.node[-1].input[2] = "zp_dq"
    ie.add_graph_outputs(model, ["input_q"])
    sample = {"input": np.array([[0.0, 0.1, 0.25, 9.0]], np.float32)}
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    input_q, expected = session.run(None, sample)
    detached = ie._detach_paired_dq_outputs(model)
    assert [node.output[0] for node in detached] == ["output"]
    values = ie._dequantize_on_cpu(model, detached, {"input_q": input_q})
    assert values.keys() == {"output"}
    np.testing.assert_array_equal(values["output"], expected)


@pytest.mark.parametrize("option", ["", "use_cpu", "disable_cpu_fallback"])
def test_run_computes_paired_dq_outputs_on_cpu(
    tmp_path, qdq_chain_model, qdq_input_sample, logger, monkeypatch, option
):
    from conftest import _cpu_session  # noqa: PLC0415

    session_outputs = []

    def target_session(model, *args, **kwargs):
        session_outputs.append([value_info.name for value_info in model.graph.output])
        return _cpu_session(model)

    monkeypatch.setattr(ie, "create_session", target_session)
    monkeypatch.setattr(ie, "_ep_library_registered", lambda *args: nullcontext())
    requested = ["conv1_out_q", "conv1_out_dq", "relu1_out_q"]
    config = OrtQnnInferenceEngineInputConfig(
        qdq_model=qdq_chain_model,
        input_sample=qdq_input_sample,
        outputs_list=requested,
        working_directory=tmp_path,
        **({option: True} if option else {}),
    )
    result = OrtQnnInferenceEngine(logger).run(config)
    assert list(result.output_data) == [*requested, "output"]
    assert session_outputs == [["conv1_out_q", "relu1_out_q"] if not option else [*requested, "output"]]
    assert result.host_dequantized_outputs == (() if option else ("conv1_out_dq", "output"))
    model = onnx.load(qdq_chain_model)
    ie.add_graph_outputs(model, requested)
    expected = _cpu_session(model).run(None, qdq_input_sample)
    for name, value in zip([*requested, "output"], expected, strict=True):
        np.testing.assert_array_equal(result.output_data[name], value.astype(np.float32))
