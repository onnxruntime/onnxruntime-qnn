# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Shared pytest bootstrap and fixtures for the standalone accuracy debugger."""

from __future__ import annotations

import importlib.util
import logging
import sys
from collections.abc import Callable, Iterable, Sequence
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

# Match the import path used by ``python accuracy_debugger.py``, including in spawned workers.
DEBUGGER_DIR = Path(__file__).resolve().parents[1]
if str(DEBUGGER_DIR) not in sys.path:
    sys.path.insert(0, str(DEBUGGER_DIR))

import log as adbg_log  # noqa: E402

_TESTS_DIR = Path(__file__).resolve().parent

try:
    import onnxruntime as _ort

    # False when the repo-root onnxruntime/ namespace directory shadows a missing installation.
    ORT_OK = hasattr(_ort, "InferenceSession")
except ImportError:
    ORT_OK = False

QNN_EP_NAME = "QNNExecutionProvider"
INPUT_SHAPE = (1, 3, 8, 8)


def pytest_configure(config: pytest.Config) -> None:
    """Registers the custom markers."""
    config.addinivalue_line("markers", "qnn: test needs the onnxruntime-qnn QNN Execution Provider")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skips this directory's tests when onnxruntime is not importable (never skips at conftest import time)."""
    if ORT_OK:
        return
    skip = pytest.mark.skip(reason="onnxruntime is not importable")
    for item in items:
        if _TESTS_DIR in Path(item.path).resolve().parents:
            item.add_marker(skip)


def requires_qnn_ep() -> bool:
    """Returns True iff the QNN EP plugin is installed, registrable and exposes a QNN device.

    Only call this from ``qnn``-marked tests (never at collection time).
    """
    if not ORT_OK or importlib.util.find_spec("onnxruntime_qnn") is None:
        return False
    try:
        from ort_qnn_inference_engine import ensure_qnn_ep_registered  # noqa: PLC0415

        ensure_qnn_ep_registered()
    except (ImportError, RuntimeError):
        return False
    import onnxruntime as ort  # noqa: PLC0415

    return any(getattr(d, "ep_name", None) == QNN_EP_NAME for d in ort.get_ep_devices())


def _all_loggers() -> list[logging.Logger]:
    loggers = [logging.getLogger()]
    loggers += [lg for lg in logging.Logger.manager.loggerDict.values() if isinstance(lg, logging.Logger)]
    return loggers


def _marked_handler_ids() -> set[int]:
    return {id(h) for lg in _all_loggers() for h in lg.handlers if getattr(h, "_adbg_handler", False)}


@pytest.fixture(autouse=True)
def _reset_adbg_state() -> Iterable[None]:
    """Resets the VERBOSE warning state and drops the ``_adbg_handler`` handlers attached during the test."""
    adbg_log._VERBOSE_WARNED.clear()
    before = _marked_handler_ids()
    yield
    adbg_log._VERBOSE_WARNED.clear()
    for lg in _all_loggers():
        for handler in list(lg.handlers):
            if getattr(handler, "_adbg_handler", False) and id(handler) not in before:
                lg.removeHandler(handler)
                handler.close()


@pytest.fixture
def logger() -> logging.Logger:
    """A real, picklable logger."""
    return logging.getLogger("adbg_test")


@pytest.fixture
def mock_logger() -> MagicMock:
    """A ``logging.Logger`` mock that reports every level as enabled."""
    mock = MagicMock(spec=logging.Logger)
    mock.isEnabledFor.return_value = True
    return mock


def _make_model(
    nodes: Sequence[onnx.NodeProto],
    inputs: Sequence[onnx.ValueInfoProto],
    outputs: Sequence[onnx.ValueInfoProto],
    initializers: Sequence[onnx.TensorProto] = (),
    opset: int = 19,
    ir_version: int = 9,
    value_info: Sequence[onnx.ValueInfoProto] = (),
) -> onnx.ModelProto:
    graph = helper.make_graph(
        list(nodes), "adbg_test_graph", list(inputs), list(outputs), list(initializers), value_info=list(value_info)
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = ir_version
    return model


@pytest.fixture
def make_model() -> Callable[..., onnx.ModelProto]:
    """Callable ``(nodes, inputs, outputs, initializers=(), opset=19, ir_version=9, value_info=()) -> ModelProto``."""
    return _make_model


def _save_model(model: onnx.ModelProto, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path


@pytest.fixture
def save_model() -> Callable[[onnx.ModelProto, str | Path], Path]:
    """Callable ``(model, path) -> Path`` that saves a model (creating parent directories)."""
    return _save_model


# Quantisation parameters shared by the QDQ chain and its FP32 twin.
_S_IN = np.float32(1.0 / 255.0)
_ZP_IN = np.uint8(0)
_S_W = np.float32(0.01)
_ZP_W = np.int8(0)
_S_CONV = np.float32(0.05)
_ZP_CONV = np.uint8(128)
_S_RELU = np.float32(0.025)
_ZP_RELU = np.uint8(0)


def _weight_q() -> np.ndarray:
    return np.random.default_rng(42).integers(-100, 100, size=(4, 3, 3, 3), dtype=np.int8)


def _weight_fp() -> np.ndarray:
    return ((_weight_q().astype(np.float32) - np.float32(_ZP_W)) * _S_W).astype(np.float32)


def _scalar(name: str, value: np.generic) -> onnx.TensorProto:
    return numpy_helper.from_array(np.array(value), name)


@pytest.fixture
def qdq_chain_model(tmp_path: Path) -> Path:
    """Path to a runnable QDQ ``Q/DQ -> Conv -> Q/DQ -> Relu -> Q/DQ`` model (opset 19, IR 9)."""
    q, dq = "QuantizeLinear", "DequantizeLinear"
    nodes = [
        helper.make_node(q, ["input", "s_in", "zp_in"], ["input_q"], name="input_q"),
        helper.make_node(dq, ["input_q", "s_in", "zp_in"], ["input_dq"], name="input_dq"),
        helper.make_node(dq, ["w_q", "s_w", "zp_w"], ["w_dq"], name="w_dq"),
        helper.make_node("Conv", ["input_dq", "w_dq"], ["conv1_out"], name="conv1", pads=[1, 1, 1, 1]),
        helper.make_node(q, ["conv1_out", "s_conv", "zp_conv"], ["conv1_out_q"], name="conv1_q"),
        helper.make_node(dq, ["conv1_out_q", "s_conv", "zp_conv"], ["conv1_out_dq"], name="conv1_dq"),
        helper.make_node("Relu", ["conv1_out_dq"], ["relu1_out"], name="relu1"),
        helper.make_node(q, ["relu1_out", "s_relu", "zp_relu"], ["relu1_out_q"], name="relu1_q"),
        helper.make_node(dq, ["relu1_out_q", "s_relu", "zp_relu"], ["output"], name="relu1_dq"),
    ]
    initializers = [
        _scalar("s_in", _S_IN),
        _scalar("zp_in", _ZP_IN),
        numpy_helper.from_array(_weight_q(), "w_q"),
        _scalar("s_w", _S_W),
        _scalar("zp_w", _ZP_W),
        _scalar("s_conv", _S_CONV),
        _scalar("zp_conv", _ZP_CONV),
        _scalar("s_relu", _S_RELU),
        _scalar("zp_relu", _ZP_RELU),
    ]
    model = _make_model(
        nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, list(INPUT_SHAPE))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 8, 8])],
        initializers,
    )
    onnx.checker.check_model(model)
    return _save_model(model, tmp_path / "qdq_chain" / "qdq_chain.onnx")


@pytest.fixture
def fp32_chain_model(tmp_path: Path) -> Path:
    """Path to the FP32 twin of ``qdq_chain_model``: ``Conv "conv1" -> Relu "relu1"`` with dequantized weights."""
    nodes = [
        helper.make_node("Conv", ["input", "w_fp"], ["conv1_out"], name="conv1", pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["conv1_out"], ["output"], name="relu1"),
    ]
    model = _make_model(
        nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, list(INPUT_SHAPE))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 8, 8])],
        [numpy_helper.from_array(_weight_fp(), "w_fp")],
    )
    onnx.checker.check_model(model)
    return _save_model(model, tmp_path / "fp32_chain" / "fp32_chain.onnx")


@pytest.fixture(params=[False, True], ids=["compiled-only", "qdq-boundary"])
def cached_context_model(request, qdq_chain_model: Path, tmp_path: Path) -> Path:
    """Unsupported compiled targets, with and without visible Q/DQ boundary nodes."""
    model = onnx.load(qdq_chain_model)
    boundary = list(model.graph.node[:2]) if request.param else []
    del model.graph.node[:]
    model.graph.node.extend(boundary)
    model.graph.node.append(
        helper.make_node(
            "EPContext",
            ["input_dq" if request.param else "input"],
            ["output"],
            domain="com.microsoft",
        )
    )
    path = tmp_path / "cached.onnx"
    onnx.save(model, path)
    return path


@pytest.fixture
def qdq_input_sample() -> dict[str, np.ndarray]:
    """One deterministic float32 input sample for the chain models."""
    return {"input": np.random.default_rng(0).random(INPUT_SHAPE, dtype=np.float32)}


def _write_input_list(sample: dict[str, np.ndarray], directory: Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tokens = []
    for name, array in sample.items():
        raw = (directory / f"{name}.raw").resolve()
        np.ascontiguousarray(array).tofile(raw)
        tokens.append(f'{name}:="{raw}"')
    input_list = directory / "input_list.txt"
    input_list.write_text(" ".join(tokens) + "\n", encoding="utf8")
    return input_list


@pytest.fixture
def write_input_list() -> Callable[[dict[str, np.ndarray], Path], Path]:
    """Callable ``(sample, dir) -> Path`` writing ``<name>.raw`` files and a one-line ``input_list.txt``."""
    return _write_input_list


def _cpu_session(model, *args, **kwargs):
    """Real ORT CPU target for tests, including spawned workers."""
    import onnxruntime as ort  # noqa: PLC0415

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])


def cpu_inference_job(config, logger_name):
    import ort_qnn_inference_engine as engine  # noqa: PLC0415

    with (
        patch.object(engine, "create_session", _cpu_session),
        patch.object(engine, "_ep_library_registered", lambda *args: nullcontext()),
    ):
        return engine.run_inference_job(config, logger_name)


@pytest.fixture
def ort_cpu_target(monkeypatch):
    import ort_qnn_inference_engine as engine  # noqa: PLC0415
    import ort_qnn_snooper as snooper  # noqa: PLC0415

    monkeypatch.setattr(engine, "create_session", _cpu_session)
    monkeypatch.setattr(engine, "_ep_library_registered", lambda *args: nullcontext())
    monkeypatch.setattr(snooper, "run_inference_job", cpu_inference_job)


@pytest.fixture
def quantized_model(tmp_path: Path) -> Path:
    """Path to an FP32 ``Conv -> Relu -> Add(residual)`` model quantized to QDQ (QUInt8 activations, QInt8 weights)."""
    quantization = pytest.importorskip("onnxruntime.quantization")

    weight = np.random.default_rng(7).standard_normal((3, 3, 3, 3)).astype(np.float32) * np.float32(0.2)
    bias = np.random.default_rng(8).standard_normal(3).astype(np.float32) * np.float32(0.1)
    nodes = [
        helper.make_node("Conv", ["input", "w", "b"], ["conv1_out"], name="conv1", pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["conv1_out"], ["relu1_out"], name="relu1"),
        helper.make_node("Add", ["relu1_out", "input"], ["output"], name="add1"),
    ]
    model = _make_model(
        nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, list(INPUT_SHAPE))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, list(INPUT_SHAPE))],
        [numpy_helper.from_array(weight, "w"), numpy_helper.from_array(bias, "b")],
        opset=17,
        ir_version=8,
    )
    fp32_path = _save_model(model, tmp_path / "quantized" / "fp32_residual.onnx")
    qdq_path = tmp_path / "quantized" / "residual.qdq.onnx"

    class _Reader(quantization.CalibrationDataReader):
        def __init__(self) -> None:
            rng = np.random.default_rng(1)
            self._samples = iter([{"input": rng.random(INPUT_SHAPE, dtype=np.float32)} for _ in range(4)])

        def get_next(self) -> dict[str, np.ndarray] | None:
            return next(self._samples, None)

    quantization.quantize_static(
        str(fp32_path),
        str(qdq_path),
        _Reader(),
        quant_format=quantization.QuantFormat.QDQ,
        activation_type=quantization.QuantType.QUInt8,
        weight_type=quantization.QuantType.QInt8,
    )
    return qdq_path
