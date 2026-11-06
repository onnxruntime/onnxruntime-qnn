# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Unit tests for ort_qnn_snooper (ported from the legacy snooper tests plus new ones).

Collaborators from other modules are patched at the snooper's own namespace
(``ort_qnn_snooper.<name>``).
"""

from __future__ import annotations

import pytest

pytest.importorskip("onnxruntime")

import json
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import onnx
import ort_qnn_snooper as snooper_module
import pandas as pd
from common import QDQ_OPS_INDEX_DEFAULT, ActivationStatus, Algorithm, VerificationError
from comparators import get_comparator
from inputs import InputProcessor
from onnx import TensorProto, helper
from ort_qnn_inference_engine import (
    OrtQnnInferenceEngine,
    OrtQnnInferenceEngineOutputConfig,
)
from ort_qnn_snooper import (
    OrtQnnSnooper,
    OrtQnnSnooperInputConfig,
    OrtQnnSnooperOutputConfig,
)
from reporting import ActivationInfo, get_activations_info

MODULE = "ort_qnn_snooper"
LAYERWISE_COLUMNS = [
    "Layer Name",
    "Layer Output",
    "Layer Status",
    "Layer Type",
    "Layer Shape",
    "Source(Min, Max, Median)",
    "Target(Min, Max, Median)",
    "mse",
    "Exception",
]
# Subgraph inputs of the two layers of ``qdq_chain_model`` (end tensor -> extracted inputs).
CHAIN_SUBGRAPH_INPUTS = {"conv1_out_dq": ["input_dq"], "output": ["conv1_out_dq"]}
CHAIN_TENSORS = ["input_q", "input_dq", "w_dq", "conv1_out_q", "conv1_out_dq", "relu1_out_q", "output"]


# =============================================================================
# Fixtures and helpers
# =============================================================================


@pytest.fixture
def snooper(mock_logger: MagicMock) -> OrtQnnSnooper:
    return OrtQnnSnooper(mock_logger, name=Algorithm.ONESHOT)


@pytest.fixture
def layerwise_snooper(mock_logger: MagicMock) -> OrtQnnSnooper:
    return OrtQnnSnooper(mock_logger, name=Algorithm.LAYERWISE, isolate_inference=False)


@pytest.fixture
def working_directory(tmp_path: Path) -> Path:
    wd = tmp_path / "wd"
    wd.mkdir()
    return wd


@pytest.fixture
def input_processor(qdq_input_sample: dict[str, np.ndarray]) -> MagicMock:
    processor = MagicMock(spec=InputProcessor)
    processor.get_framework_inputs.return_value = [qdq_input_sample]
    return processor


@pytest.fixture
def sample_input_config(
    qdq_chain_model: Path, input_processor: MagicMock, working_directory: Path
) -> OrtQnnSnooperInputConfig:
    return OrtQnnSnooperInputConfig(
        qdq_model=qdq_chain_model,
        input_processor=input_processor,
        algorithm=Algorithm.ONESHOT,
        working_directory=working_directory,
    )


@pytest.fixture
def chain_reference() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    return {name: rng.random((1, 4, 8, 8), dtype=np.float32) for name in CHAIN_TENSORS}


def _report_paths(tmp_path: Path) -> tuple[Path, Path]:
    csv_path, json_path = tmp_path / "report.csv", tmp_path / "report.json"
    csv_path.touch()
    json_path.touch()
    return csv_path, json_path


def _fake_extractor_class(model_to_copy: Path, inputs: dict[str, list[str]] | None = None, fail: bool = False):
    """Builds an ``OnnxSubgraphExtractor`` stand-in that saves a real QDQ model under the subgraph dir."""
    inputs = CHAIN_SUBGRAPH_INPUTS if inputs is None else inputs

    class FakeExtractor:
        instances: list[FakeExtractor] = []  # noqa: RUF012

        def __init__(self, model_path: str | Path, *, logger: logging.Logger | None = None) -> None:
            self.model_path = Path(model_path)
            self.calls: list[tuple[str, str, Path, bool, tuple[str, ...]]] = []
            FakeExtractor.instances.append(self)

        def extract(
            self,
            start: str,
            end: str,
            out_dir: str | Path,
            *,
            additional_outputs: tuple[str, ...] | list[str] = (),
            requantize_inputs: bool = False,
        ) -> SimpleNamespace:
            self.calls.append((start, end, Path(out_dir), requantize_inputs, tuple(additional_outputs)))
            if fail:
                raise RuntimeError("boom")
            out_path = Path(out_dir) / "extracted_model.onnx"
            shutil.copyfile(model_to_copy, out_path)
            return SimpleNamespace(model_path=out_path, input_names=list(inputs[end]), output_names=[end])

    return FakeExtractor


def _echo_run(reference: dict[str, np.ndarray], offset: float = 0.0):
    """Fake ``OrtQnnInferenceEngine.run`` returning the reference values of the requested outputs (+ offset)."""

    def run(config: Any) -> OrtQnnInferenceEngineOutputConfig:
        return OrtQnnInferenceEngineOutputConfig(
            output_data={name: reference[name] + np.float32(offset) for name in config.outputs_list}
        )

    return run


def _run_layerwise(
    snooper: OrtQnnSnooper,
    qdq_model: Path,
    reference: dict[str, np.ndarray],
    working_directory: Path,
    sample: dict[str, np.ndarray],
    **overrides: Any,
) -> tuple[Path, Path]:
    kwargs: dict[str, Any] = {
        "reference_model": qdq_model,
        "qdq_model": qdq_model,
        "input_sample": sample,
        "reference_outputs": reference,
        "dump_output_tensors": False,
        "comparators": [get_comparator("mse")],
        "working_directory": working_directory,
        "intermediates_layers_list": [],
        "cpu_layers_list": [],
        "reference_model_is_qdq": True,
        "dump_elementwise_stats": None,
    }
    kwargs.update(overrides)
    return snooper._execute_layerwise(**kwargs)


def _qdq_model_basic(make_model, save_model, path: Path) -> Path:
    nodes = [
        helper.make_node("QuantizeLinear", ["input", "s1", "z1"], ["intermediate"]),
        helper.make_node("DequantizeLinear", ["intermediate", "s2", "z2"], ["output"]),
    ]
    model = make_model(
        nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 4, 4])],
    )
    return save_model(model, path)


def _conv_qdq_model_named(make_model, save_model, path: Path) -> Path:
    nodes = [
        helper.make_node("Conv", ["input", "w"], ["conv_out"], name="conv1"),
        helper.make_node("QuantizeLinear", ["conv_out", "s", "z"], ["conv_out_q"], name="quant1"),
        helper.make_node("DequantizeLinear", ["conv_out_q", "s", "z"], ["output"], name="dequant1"),
    ]
    model = make_model(
        nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 4, 4])],
    )
    return save_model(model, path)


def _conv_weight_qdq_model(make_model, weight: str) -> onnx.ModelProto:
    """Builds ``input -> Q/DQ -> Conv(w_dq, b_dq) -> Q/DQ``, where ``weight`` selects how ``w_q`` is produced.

    ``"initializer"``: ``w_q`` is an initializer. ``"sparse"``: a sparse initializer. ``"constant"``: a Constant node.
    The other kinds quantize a float ``w``. ``"quantized"``: ``w`` is an initializer. ``"transposed"``, ``"scaled"``:
    a Transpose or a Mul of float initializers. ``"reshaped"``, ``"concatenated"``, ``"exponentiated"``: a Reshape, a
    Concat or an Exp of dequantized int8 initializers; ORT folds the ``DQ -> Exp -> Q`` unit. ``"matmul"``,
    ``"relu_chain"``: a MatMul or two Relus of dequantized int8 initializers, which QNN EP computes. The bias ``b_q``
    is an int32 initializer.
    """
    q, dq = "QuantizeLinear", "DequantizeLinear"
    w_q = helper.make_tensor("w_q", TensorProto.INT8, [2, 1, 1, 1], [1, 2])
    quantize_w = helper.make_node(q, ["w", "s_w", "z_w"], ["w_q"])

    def int8(name: str, dims: list[int]) -> onnx.TensorProto:
        return helper.make_tensor(name, TensorProto.INT8, dims, [1] * int(np.prod(dims)))

    def dequantize(name: str) -> onnx.NodeProto:
        return helper.make_node(dq, [name, "s_w", "z_w"], [f"{name}_dq"])

    weight_nodes, weight_initializers = {
        "initializer": ([], [w_q]),
        "sparse": ([], []),
        "constant": ([helper.make_node("Constant", [], ["w_q"], value=w_q)], []),
        "quantized": ([quantize_w], [helper.make_tensor("w", TensorProto.FLOAT, [2, 1, 1, 1], [0.01, 0.02])]),
        "transposed": (
            [helper.make_node("Transpose", ["w_t"], ["w"], perm=[1, 0, 2, 3]), quantize_w],
            [helper.make_tensor("w_t", TensorProto.FLOAT, [1, 2, 1, 1], [0.01, 0.02])],
        ),
        "scaled": (
            [helper.make_node("Mul", ["w_f", "w_scale"], ["w"]), quantize_w],
            [
                helper.make_tensor("w_f", TensorProto.FLOAT, [2, 1, 1, 1], [0.01, 0.02]),
                helper.make_tensor("w_scale", TensorProto.FLOAT, [], [2.0]),
            ],
        ),
        "reshaped": (
            [dequantize("w_i"), helper.make_node("Reshape", ["w_i_dq", "w_shape"], ["w"]), quantize_w],
            [int8("w_i", [2]), helper.make_tensor("w_shape", TensorProto.INT64, [4], [2, 1, 1, 1])],
        ),
        "concatenated": (
            [
                dequantize("w_i"),
                dequantize("w_j"),
                helper.make_node("Concat", ["w_i_dq", "w_j_dq"], ["w"], axis=0),
                quantize_w,
            ],
            [int8("w_i", [1, 1, 1, 1]), int8("w_j", [1, 1, 1, 1])],
        ),
        "exponentiated": (
            [dequantize("w_i"), helper.make_node("Exp", ["w_i_dq"], ["w"]), quantize_w],
            [int8("w_i", [2, 1, 1, 1])],
        ),
        "matmul": (
            [dequantize("w_i"), dequantize("w_j"), helper.make_node("MatMul", ["w_i_dq", "w_j_dq"], ["w"]), quantize_w],
            [int8("w_i", [2, 1, 1, 1]), int8("w_j", [1, 1])],
        ),
        "relu_chain": (
            [
                dequantize("w_i"),
                helper.make_node("Relu", ["w_i_dq"], ["w_relu"]),
                helper.make_node("Relu", ["w_relu"], ["w"]),
                quantize_w,
            ],
            [int8("w_i", [2, 1, 1, 1])],
        ),
    }[weight]
    nodes = [
        *weight_nodes,
        helper.make_node(dq, ["w_q", "s_w", "z_w"], ["w_dq"]),
        helper.make_node(dq, ["b_q", "s_b"], ["b_dq"]),
        helper.make_node(q, ["input", "s", "z"], ["input_q"]),
        helper.make_node(dq, ["input_q", "s", "z"], ["input_dq"]),
        helper.make_node("Conv", ["input_dq", "w_dq", "b_dq"], ["conv_out"], name="conv1"),
        helper.make_node(q, ["conv_out", "s", "z"], ["conv_out_q"]),
        helper.make_node(dq, ["conv_out_q", "s", "z"], ["output"]),
    ]
    initializers = [
        *weight_initializers,
        helper.make_tensor("s", TensorProto.FLOAT, [], [0.1]),
        helper.make_tensor("z", TensorProto.UINT8, [], [0]),
        helper.make_tensor("s_w", TensorProto.FLOAT, [], [0.01]),
        helper.make_tensor("z_w", TensorProto.INT8, [], [0]),
        helper.make_tensor("b_q", TensorProto.INT32, [2], [3, 4]),
        helper.make_tensor("s_b", TensorProto.FLOAT, [], [0.001]),
    ]
    model = make_model(
        nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 2, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 2, 2])],
        initializers,
    )
    if weight == "sparse":
        values = helper.make_tensor("w_q", TensorProto.INT8, [2], [1, 2])
        indices = helper.make_tensor("w_q_indices", TensorProto.INT64, [2], [0, 1])
        model.graph.sparse_initializer.append(helper.make_sparse_tensor(values, indices, [2, 1, 1, 1]))
    return model


def _interleaved_branches_model(make_model) -> onnx.ModelProto:
    """Builds a residual block start whose Convs both precede their Q/DQ nodes, as PyTorch exports a downsample."""
    q, dq = "QuantizeLinear", "DequantizeLinear"
    nodes = [
        helper.make_node(q, ["input", "s", "z"], ["input_q"]),
        helper.make_node(dq, ["input_q", "s", "z"], ["block_in"]),
        helper.make_node("Conv", ["block_in", "w"], ["conv1_out"], name="conv1"),
        helper.make_node("Conv", ["block_in", "w"], ["ds_out"], name="downsample"),
        helper.make_node(q, ["conv1_out", "s", "z"], ["conv1_out_q"]),
        helper.make_node(q, ["ds_out", "s", "z"], ["ds_out_q"]),
        helper.make_node(dq, ["conv1_out_q", "s", "z"], ["conv1_out_dq"]),
        helper.make_node(dq, ["ds_out_q", "s", "z"], ["ds_out_dq"]),
        helper.make_node("Add", ["conv1_out_dq", "ds_out_dq"], ["output"], name="add"),
    ]
    return make_model(
        nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 2, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 2, 2])],
        [
            helper.make_tensor("s", TensorProto.FLOAT, [], [0.1]),
            helper.make_tensor("z", TensorProto.UINT8, [], [0]),
            helper.make_tensor("w", TensorProto.FLOAT, [1, 1, 1, 1], [0.5]),
        ],
    )


def _split_branches_model(make_model) -> onnx.ModelProto:
    """Builds ``Split`` with a Q/DQ pair on each output, as a two-output layer."""
    q, dq = "QuantizeLinear", "DequantizeLinear"
    nodes = [
        helper.make_node(q, ["input", "s", "z"], ["input_q"]),
        helper.make_node(dq, ["input_q", "s", "z"], ["input_dq"]),
        helper.make_node("Split", ["input_dq"], ["a", "b"], name="split", axis=1, num_outputs=2),
        helper.make_node(q, ["a", "s", "z"], ["a_q"]),
        helper.make_node(dq, ["a_q", "s", "z"], ["a_dq"]),
        helper.make_node(q, ["b", "s", "z"], ["b_q"]),
        helper.make_node(dq, ["b_q", "s", "z"], ["b_dq"]),
        helper.make_node("Add", ["a_dq", "b_dq"], ["output"], name="add"),
    ]
    return make_model(
        nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 2, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 2, 2])],
        [helper.make_tensor("s", TensorProto.FLOAT, [], [0.1]), helper.make_tensor("z", TensorProto.UINT8, [], [0])],
    )


def _node(name: str, op_type: str, outputs: list[str], inputs: list[str] | None = None) -> MagicMock:
    node = MagicMock()
    node.name = name
    node.op_type = op_type
    node.input = inputs or []
    node.output = outputs
    return node


def _renamed_chain(qdq_chain_model: Path, tmp_path: Path, renames: dict[str, str], name: str) -> Path:
    model = onnx.load(qdq_chain_model)
    for node in model.graph.node:
        if node.name in renames:
            node.name = renames[node.name]
    path = tmp_path / name
    onnx.save(model, path)
    return path


# =============================================================================
# Initialisation
# =============================================================================


class TestInitialization:
    def test_init_with_default_algorithm(self, mock_logger: MagicMock) -> None:
        snooper = OrtQnnSnooper(mock_logger)
        assert snooper._name == Algorithm.ONESHOT
        assert snooper._logger is mock_logger
        assert snooper._isolate_inference is True

    def test_init_with_layerwise_algorithm(self, mock_logger: MagicMock) -> None:
        snooper = OrtQnnSnooper(mock_logger, name=Algorithm.LAYERWISE, isolate_inference=False)
        assert snooper._name == Algorithm.LAYERWISE
        assert snooper._isolate_inference is False

    def test_init_with_custom_logger(self) -> None:
        custom_logger = logging.getLogger("custom_test_logger")
        assert OrtQnnSnooper(custom_logger)._logger is custom_logger


# =============================================================================
# Config validators
# =============================================================================


class TestInputConfigValidators:
    def test_reference_model_only_with_oneshot(
        self, qdq_chain_model: Path, fp32_chain_model: Path, input_processor: MagicMock, working_directory: Path
    ) -> None:
        with pytest.raises(ValueError, match="reference_model is supported only with oneshot algorithm."):
            OrtQnnSnooperInputConfig(
                qdq_model=qdq_chain_model,
                reference_model=fp32_chain_model,
                input_processor=input_processor,
                algorithm=Algorithm.LAYERWISE,
                working_directory=working_directory,
            )

    def test_set_cpu_layers_only_with_layerwise(
        self, qdq_chain_model: Path, input_processor: MagicMock, working_directory: Path
    ) -> None:
        with pytest.raises(ValueError, match="set_cpu_layers option is supported only with layerwise algorithm."):
            OrtQnnSnooperInputConfig(
                qdq_model=qdq_chain_model,
                input_processor=input_processor,
                algorithm=Algorithm.ONESHOT,
                set_cpu_layers=["conv1"],
                working_directory=working_directory,
            )

    def test_elementwise_requires_qdq_reference(
        self, qdq_chain_model: Path, fp32_chain_model: Path, input_processor: MagicMock, working_directory: Path
    ) -> None:
        with pytest.raises(
            ValueError,
            match="dump_elementwise_stats option is supported only when reference_model is a quantized model.",
        ):
            OrtQnnSnooperInputConfig(
                qdq_model=qdq_chain_model,
                reference_model=fp32_chain_model,
                input_processor=input_processor,
                dump_elementwise_stats={"conv1": 1},
                working_directory=working_directory,
            )

    def test_elementwise_with_qdq_reference_ok(
        self, qdq_chain_model: Path, input_processor: MagicMock, working_directory: Path
    ) -> None:
        config = OrtQnnSnooperInputConfig(
            qdq_model=qdq_chain_model,
            reference_model=qdq_chain_model,
            input_processor=input_processor,
            dump_elementwise_stats={"conv1": 1},
            working_directory=working_directory,
        )
        assert config.dump_elementwise_stats == {"conv1": 1}
        assert [c.name for c in config.comparators] == ["mse"]

    def test_non_qdq_qdq_model(
        self, fp32_chain_model: Path, input_processor: MagicMock, working_directory: Path
    ) -> None:
        with pytest.raises(ValueError, match="Model doesn't contain QDQ nodes. Please provide a valid QDQ ONNX file."):
            OrtQnnSnooperInputConfig(
                qdq_model=fp32_chain_model, input_processor=input_processor, working_directory=working_directory
            )

    def test_wrong_suffix(self, tmp_path: Path, input_processor: MagicMock, working_directory: Path) -> None:
        bad = tmp_path / "model.txt"
        bad.write_text("not a model")
        with pytest.raises(ValueError, match="Invalid model format specified. Please provide a valid ONNX file."):
            OrtQnnSnooperInputConfig(
                qdq_model=bad, input_processor=input_processor, working_directory=working_directory
            )

    def test_missing_path(self, tmp_path: Path, input_processor: MagicMock, working_directory: Path) -> None:
        with pytest.raises(ValueError, match="qdq_model"):
            OrtQnnSnooperInputConfig(
                qdq_model=tmp_path / "missing.onnx",
                input_processor=input_processor,
                working_directory=working_directory,
            )

    def test_working_directory_required(self, qdq_chain_model: Path, input_processor: MagicMock) -> None:
        with pytest.raises(ValueError, match="working_directory"):
            OrtQnnSnooperInputConfig(qdq_model=qdq_chain_model, input_processor=input_processor)

    def test_extra_fields_forbidden(
        self, qdq_chain_model: Path, input_processor: MagicMock, working_directory: Path
    ) -> None:
        with pytest.raises(ValueError, match="bogus"):
            OrtQnnSnooperInputConfig(
                qdq_model=qdq_chain_model,
                input_processor=input_processor,
                working_directory=working_directory,
                bogus=1,
            )


# =============================================================================
# run()
# =============================================================================


class TestRun:
    @patch(f"{MODULE}.is_qdq_model")
    @patch.object(OrtQnnSnooper, "_execute_oneshot")
    @patch(f"{MODULE}.generate_framework_outputs")
    @patch.object(OrtQnnSnooper, "_get_valid_intermediates")
    def test_run_oneshot_success(
        self,
        mock_get_valid_intermediates: MagicMock,
        mock_generate: MagicMock,
        mock_execute_oneshot: MagicMock,
        mock_is_qdq: MagicMock,
        snooper: OrtQnnSnooper,
        sample_input_config: OrtQnnSnooperInputConfig,
        tmp_path: Path,
    ) -> None:
        csv_path, json_path = _report_paths(tmp_path)
        mock_is_qdq.return_value = False
        mock_get_valid_intermediates.return_value = ["layer1", "layer2"]
        mock_generate.return_value = {"output": np.random.rand(1, 1000)}
        mock_execute_oneshot.return_value = (csv_path, json_path)

        result = snooper.run(sample_input_config)

        assert isinstance(result, OrtQnnSnooperOutputConfig)
        assert result.csv_snooping_report == csv_path
        assert result.json_snooping_report == json_path
        mock_is_qdq.assert_called_once_with(Path(sample_input_config.qdq_model))
        mock_get_valid_intermediates.assert_called()
        mock_generate.assert_called_once()
        assert mock_generate.call_args.kwargs["inputs"] is sample_input_config.input_processor.get_framework_inputs()[0]
        assert mock_generate.call_args.kwargs["intermediate_output_tensors"] == ["layer1", "layer2"]
        mock_execute_oneshot.assert_called_once()

    @patch(f"{MODULE}.is_qdq_model", return_value=False)
    @patch.object(OrtQnnSnooper, "_execute_oneshot")
    @patch(f"{MODULE}.generate_framework_outputs")
    @patch.object(OrtQnnSnooper, "_get_valid_intermediates", return_value=[])
    def test_run_with_working_directory_provided(
        self,
        _mock_get_valid_intermediates: MagicMock,
        mock_generate: MagicMock,
        mock_execute_oneshot: MagicMock,
        _mock_is_qdq: MagicMock,
        snooper: OrtQnnSnooper,
        sample_input_config: OrtQnnSnooperInputConfig,
        tmp_path: Path,
    ) -> None:
        mock_generate.return_value = {"output": np.random.rand(1, 1000)}
        mock_execute_oneshot.return_value = _report_paths(tmp_path)
        before = sorted(Path(sample_input_config.working_directory).iterdir())

        snooper.run(sample_input_config)

        assert sorted(Path(sample_input_config.working_directory).iterdir()) == before
        assert mock_execute_oneshot.call_args.kwargs["working_directory"] == Path(sample_input_config.working_directory)

    @patch(f"{MODULE}.is_qdq_model", return_value=True)
    @patch.object(OrtQnnSnooper, "_execute_oneshot")
    @patch(f"{MODULE}.generate_framework_outputs")
    @patch.object(OrtQnnSnooper, "_get_valid_intermediates", return_value=[])
    def test_run_with_qdq_reference_model(
        self,
        _mock_get_valid_intermediates: MagicMock,
        mock_generate: MagicMock,
        mock_execute_oneshot: MagicMock,
        _mock_is_qdq: MagicMock,
        snooper: OrtQnnSnooper,
        sample_input_config: OrtQnnSnooperInputConfig,
        tmp_path: Path,
    ) -> None:
        mock_generate.return_value = {"output": np.random.rand(1, 1000)}
        mock_execute_oneshot.return_value = _report_paths(tmp_path)
        snooper.run(sample_input_config)
        assert mock_execute_oneshot.call_args.kwargs["reference_model_is_qdq"] is True

    @patch.object(OrtQnnSnooper, "_execute_oneshot")
    @patch(f"{MODULE}.generate_framework_outputs")
    def test_run_warns_on_multiple_samples(
        self,
        mock_generate: MagicMock,
        mock_execute_oneshot: MagicMock,
        snooper: OrtQnnSnooper,
        sample_input_config: OrtQnnSnooperInputConfig,
        qdq_input_sample: dict[str, np.ndarray],
        tmp_path: Path,
    ) -> None:
        second = {"input": qdq_input_sample["input"] + 1}
        sample_input_config.input_processor.get_framework_inputs.return_value = [qdq_input_sample, second]
        mock_generate.return_value = {"output": np.zeros(3, np.float32)}
        mock_execute_oneshot.return_value = _report_paths(tmp_path)

        snooper.run(sample_input_config)

        snooper._logger.warning.assert_any_call("Input list contains 2 samples; only the first sample is used.")
        assert mock_execute_oneshot.call_args.kwargs["input_sample"] is qdq_input_sample

    @patch.object(OrtQnnSnooper, "_execute_layerwise")
    @patch(f"{MODULE}.generate_framework_outputs")
    def test_run_layerwise_dispatch(
        self,
        mock_generate: MagicMock,
        mock_execute_layerwise: MagicMock,
        layerwise_snooper: OrtQnnSnooper,
        sample_input_config: OrtQnnSnooperInputConfig,
        tmp_path: Path,
    ) -> None:
        sample_input_config.algorithm = Algorithm.LAYERWISE
        sample_input_config.set_cpu_layers = ["conv1"]
        mock_generate.return_value = {"output": np.zeros(3, np.float32)}
        mock_execute_layerwise.return_value = _report_paths(tmp_path)

        layerwise_snooper.run(sample_input_config)

        assert mock_generate.call_args.kwargs["intermediate_output_tensors"] is None
        kwargs = mock_execute_layerwise.call_args.kwargs
        assert kwargs["cpu_layers_list"] == ["conv1"]
        assert kwargs["reference_model"] == Path(sample_input_config.qdq_model)
        layerwise_snooper._logger.info.assert_any_call("Started layerwise snooping")

    @patch.object(OrtQnnSnooper, "_execute_oneshot")
    @patch(f"{MODULE}.generate_framework_outputs")
    def test_run_with_golden_reference(
        self,
        mock_generate: MagicMock,
        mock_execute_oneshot: MagicMock,
        snooper: OrtQnnSnooper,
        sample_input_config: OrtQnnSnooperInputConfig,
        tmp_path: Path,
    ) -> None:
        golden = tmp_path / "golden"
        golden.mkdir()
        conv_out = np.arange(256, dtype=np.float32).reshape(1, 4, 8, 8)
        conv_out.tofile(golden / "conv1_out_dq.raw")
        sample_input_config.golden_reference_path = golden
        mock_execute_oneshot.return_value = _report_paths(tmp_path)

        snooper.run(sample_input_config)

        mock_generate.assert_not_called()
        reference_outputs = mock_execute_oneshot.call_args.kwargs["reference_outputs"]
        np.testing.assert_array_equal(reference_outputs["conv1_out_dq"], conv_out)
        snooper._logger.info.assert_any_call(f"Loading reference data from given {golden}")

    @patch.object(OrtQnnSnooper, "_execute_oneshot")
    @patch(f"{MODULE}.generate_framework_outputs")
    def test_f10_reference_names_differ_dumps_everything(
        self,
        mock_generate: MagicMock,
        mock_execute_oneshot: MagicMock,
        snooper: OrtQnnSnooper,
        sample_input_config: OrtQnnSnooperInputConfig,
        fp32_chain_model: Path,
        tmp_path: Path,
    ) -> None:
        renamed = _renamed_chain(fp32_chain_model, tmp_path, {"conv1": "c1", "relu1": "r1"}, "fp32_renamed.onnx")
        sample_input_config.reference_model = renamed
        sample_input_config.set_intermediate_layers = ["conv1"]
        mock_generate.return_value = {"output": np.zeros(3, np.float32)}
        mock_execute_oneshot.return_value = _report_paths(tmp_path)

        snooper.run(sample_input_config)

        assert mock_generate.call_args.kwargs["intermediate_output_tensors"] is None
        assert mock_generate.call_args.kwargs["model"] == renamed

    @patch.object(OrtQnnSnooper, "_execute_oneshot")
    @patch(f"{MODULE}.generate_framework_outputs")
    def test_f10_nothing_matches_compares_graph_outputs(
        self,
        mock_generate: MagicMock,
        mock_execute_oneshot: MagicMock,
        snooper: OrtQnnSnooper,
        sample_input_config: OrtQnnSnooperInputConfig,
        tmp_path: Path,
    ) -> None:
        sample_input_config.set_intermediate_layers = ["does_not_exist"]
        mock_generate.return_value = {"output": np.zeros(3, np.float32)}
        mock_execute_oneshot.return_value = _report_paths(tmp_path)

        snooper.run(sample_input_config)

        assert mock_generate.call_args.kwargs["intermediate_output_tensors"] == []
        snooper._logger.warning.assert_any_call(
            "No layers matched set_intermediate_layers; only graph outputs are compared."
        )

    @patch.object(OrtQnnSnooper, "_execute_oneshot")
    @patch(f"{MODULE}.generate_framework_outputs")
    def test_reference_captures_weight_dq(
        self,
        mock_generate: MagicMock,
        mock_execute_oneshot: MagicMock,
        snooper: OrtQnnSnooper,
        sample_input_config: OrtQnnSnooperInputConfig,
        tmp_path: Path,
    ) -> None:
        # The reference dump can be a layerwise golden reference, whose extracted Conv reads the weight DQ "w_dq".
        mock_generate.return_value = {"output": np.zeros(3, np.float32)}
        mock_execute_oneshot.return_value = _report_paths(tmp_path)

        snooper.run(sample_input_config)

        assert mock_generate.call_args.kwargs["intermediate_output_tensors"] == CHAIN_TENSORS

    @patch(f"{MODULE}.generate_framework_outputs", return_value={"output": np.zeros(3, np.float32)})
    def test_run_invalid_algorithm(
        self, _mock_generate: MagicMock, snooper: OrtQnnSnooper, sample_input_config: OrtQnnSnooperInputConfig
    ) -> None:
        object.__setattr__(sample_input_config, "algorithm", "cumulative")
        with pytest.raises(ValueError, match="Invalid algorithm 'cumulative' supplied"):
            snooper.run(sample_input_config)


# =============================================================================
# _execute_oneshot
# =============================================================================


class TestExecuteOneshot:
    def _call(self, snooper: OrtQnnSnooper, qdq_model: Path, reference: dict, wd: Path, **overrides: Any):
        kwargs: dict[str, Any] = {
            "qdq_model": qdq_model,
            "input_sample": {"input": np.zeros((1, 3, 8, 8), np.float32)},
            "reference_outputs": reference,
            "dump_output_tensors": False,
            "comparators": [get_comparator("mse"), get_comparator("cosine")],
            "working_directory": wd,
            "intermediates_layers_list": [],
            "reference_model_is_qdq": True,
            "dump_elementwise_stats": None,
        }
        kwargs.update(overrides)
        return snooper._execute_oneshot(**kwargs)

    @pytest.mark.parametrize(
        ("reference_is_qdq", "expected"),
        [
            # The weight DQ "w_dq" is never exposed.
            (True, ["input_q", "input_dq", "conv1_out_q", "conv1_out_dq", "relu1_out_q", "output"]),
            (False, ["input_dq", "conv1_out_dq", "output"]),
        ],
    )
    @patch(f"{MODULE}.plot_graphs")
    def test_qdq_ops_index(
        self,
        _mock_plot: MagicMock,
        snooper: OrtQnnSnooper,
        qdq_chain_model: Path,
        chain_reference: dict[str, np.ndarray],
        working_directory: Path,
        reference_is_qdq: bool,
        expected: list[str],
    ) -> None:
        with patch.object(OrtQnnInferenceEngine, "run", side_effect=_echo_run(dict.fromkeys(expected, np.ones(4)))):
            mock_run = OrtQnnInferenceEngine.run
            reference = {name: np.ones(4) for name in expected}
            self._call(snooper, qdq_chain_model, reference, working_directory, reference_model_is_qdq=reference_is_qdq)
            assert mock_run.call_args[0][0].outputs_list == expected

    @pytest.mark.parametrize(("layers", "warned"), [(["Transpose"], True), (["does_not_exist"], False)])
    @patch(f"{MODULE}.plot_graphs")
    def test_warns_when_only_weights_match(
        self,
        _mock_plot: MagicMock,
        snooper: OrtQnnSnooper,
        make_model,
        save_model,
        working_directory: Path,
        tmp_path: Path,
        layers: list[str],
        warned: bool,
    ) -> None:
        # The Transpose of the "transposed" weight feeds only the weight Q -> DQ.
        qdq_model = save_model(_conv_weight_qdq_model(make_model, "transposed"), tmp_path / "m.onnx")
        reference = {"output": np.ones(4, np.float32)}
        output = OrtQnnInferenceEngineOutputConfig(output_data=dict(reference))
        with patch.object(OrtQnnInferenceEngine, "run", return_value=output) as mock_run:
            self._call(snooper, qdq_model, reference, working_directory, intermediates_layers_list=layers)
        assert mock_run.call_args[0][0].outputs_list == []
        warnings = [entry.args[0] for entry in snooper._logger.warning.call_args_list]
        message = (
            "set_intermediate_layers matched only weight and bias Q/DQ outputs, which are not compared; "
            "only graph outputs are compared."
        )
        assert (message in warnings) is warned

    @patch(f"{MODULE}.plot_graphs")
    def test_warns_when_model_has_only_weight_qdq(
        self,
        _mock_plot: MagicMock,
        snooper: OrtQnnSnooper,
        make_model,
        save_model,
        working_directory: Path,
        tmp_path: Path,
    ) -> None:
        # Weight-only quantization: the activations are float, so every Q/DQ output is a weight or bias.
        dq = "DequantizeLinear"
        nodes = [
            helper.make_node(dq, ["w_q", "s_w", "z_w"], ["w_dq"]),
            helper.make_node(dq, ["b_q", "s_b"], ["b_dq"]),
            helper.make_node("Conv", ["input", "w_dq", "b_dq"], ["output"], name="conv1"),
        ]
        model = make_model(
            nodes,
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 2, 2])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 2, 2])],
            [
                helper.make_tensor("w_q", TensorProto.INT8, [2, 1, 1, 1], [1, 2]),
                helper.make_tensor("s_w", TensorProto.FLOAT, [], [0.01]),
                helper.make_tensor("z_w", TensorProto.INT8, [], [0]),
                helper.make_tensor("b_q", TensorProto.INT32, [2], [3, 4]),
                helper.make_tensor("s_b", TensorProto.FLOAT, [], [0.001]),
            ],
        )
        qdq_model = save_model(model, tmp_path / "m.onnx")
        reference = {"output": np.ones(4, np.float32)}
        output = OrtQnnInferenceEngineOutputConfig(output_data=dict(reference))
        with patch.object(OrtQnnInferenceEngine, "run", return_value=output) as mock_run:
            self._call(snooper, qdq_model, reference, working_directory)
        assert mock_run.call_args[0][0].outputs_list == []
        snooper._logger.warning.assert_any_call(
            "All Q/DQ outputs of the QDQ model are weights and biases, which are not compared; "
            "only graph outputs are compared."
        )

    @patch(f"{MODULE}.plot_graphs")
    def test_reports_and_plot(
        self,
        mock_plot: MagicMock,
        snooper: OrtQnnSnooper,
        qdq_chain_model: Path,
        chain_reference: dict[str, np.ndarray],
        working_directory: Path,
    ) -> None:
        with patch.object(OrtQnnInferenceEngine, "run", side_effect=_echo_run(chain_reference)):
            csv_path, json_path = self._call(
                snooper, qdq_chain_model, chain_reference, working_directory, intermediates_layers_list=["Conv"]
            )
        assert csv_path == working_directory / "oneshot.csv"
        assert json_path == working_directory / "oneshot.json"
        df = pd.read_csv(csv_path)
        assert df.columns.tolist() == [
            "Layer Name",
            "Layer Output",
            "Layer Type",
            "Layer Shape",
            "Source(Min, Max, Median)",
            "Target(Min, Max, Median)",
            "mse",
            "cosine",
        ]
        assert df["Layer Output"].tolist() == ["conv1_out_q", "conv1_out_dq"]
        assert df["Layer Name"].tolist() == ["conv1", "conv1"]
        assert df["Layer Type"].tolist() == ["Conv", "Conv"]
        assert df["Layer Shape"].tolist() == ["(1, 4, 8, 8)", "(1, 4, 8, 8)"]
        assert (df["mse"] == 0).all()
        report = json.loads(json_path.read_text(encoding="utf-8"))
        assert report["header"]["algorithm"] == "oneshot"
        assert report["header"]["artifact_type"] == "ONESHOT_SNOOPING_REPORT"
        mock_plot.assert_called_once_with(
            csv_path=csv_path,
            layer_names_column="Layer Output",
            comparator_columns=["mse", "cosine"],
            output_dir=working_directory,
            algorithm=Algorithm.ONESHOT,
            logger=snooper._logger,
        )

    @patch(f"{MODULE}.plot_graphs")
    def test_verification_error_on_empty_result(
        self, _mock_plot: MagicMock, snooper: OrtQnnSnooper, qdq_chain_model: Path, working_directory: Path
    ) -> None:
        reference = {"unrelated": np.ones(4, np.float32)}
        output = OrtQnnInferenceEngineOutputConfig(output_data={"output": np.ones(4, np.float32)})
        with (
            patch.object(OrtQnnInferenceEngine, "run", return_value=output),
            pytest.raises(VerificationError, match="Verification of tensors failed"),
        ):
            self._call(snooper, qdq_chain_model, reference, working_directory)

    @patch(f"{MODULE}.plot_graphs")
    @patch.object(OrtQnnSnooper, "_dump_elementwise_stats")
    def test_elementwise_subset(
        self,
        mock_dump: MagicMock,
        _mock_plot: MagicMock,
        snooper: OrtQnnSnooper,
        qdq_chain_model: Path,
        chain_reference: dict[str, np.ndarray],
        working_directory: Path,
    ) -> None:
        with patch.object(OrtQnnInferenceEngine, "run", side_effect=_echo_run(chain_reference)):
            self._call(
                snooper, qdq_chain_model, chain_reference, working_directory, dump_elementwise_stats={"conv1": 2}
            )
        mock_dump.assert_called_once()
        reference_subset, target_subset, steps, wd = mock_dump.call_args[0]
        assert list(reference_subset) == ["conv1_out_q", "conv1_out_dq"]
        assert list(target_subset) == ["conv1_out_q", "conv1_out_dq"]
        assert steps == 2
        assert wd == working_directory


# =============================================================================
# _execute_layerwise
# =============================================================================


class TestExecuteLayerwise:
    def test_success_rows_csv_and_json(
        self,
        layerwise_snooper: OrtQnnSnooper,
        qdq_chain_model: Path,
        chain_reference: dict[str, np.ndarray],
        working_directory: Path,
        qdq_input_sample: dict[str, np.ndarray],
    ) -> None:
        fake = _fake_extractor_class(qdq_chain_model)
        with (
            patch(f"{MODULE}.OnnxSubgraphExtractor", fake),
            patch.object(OrtQnnInferenceEngine, "run", side_effect=_echo_run(chain_reference)) as mock_run,
            patch(f"{MODULE}.plot_graphs") as mock_plot,
        ):
            csv_path, json_path = _run_layerwise(
                layerwise_snooper, qdq_chain_model, chain_reference, working_directory, qdq_input_sample
            )

        assert len(fake.instances) == 1
        assert [call[:2] for call in fake.instances[0].calls] == [
            ("input_dq", "conv1_out_dq"),
            ("conv1_out_dq", "output"),
        ]
        # The selected Q/DQ outputs are kept as subgraph outputs, even where the end tensor does not depend on them.
        assert [call[4] for call in fake.instances[0].calls] == [
            ("conv1_out_q", "conv1_out_dq"),
            ("relu1_out_q", "output"),
        ]
        assert mock_run.call_count == 2
        first_config = mock_run.call_args_list[0][0][0]
        assert first_config.outputs_list == ["conv1_out_q", "conv1_out_dq"]
        assert first_config.use_cpu is False
        np.testing.assert_array_equal(first_config.input_sample["input_dq"], chain_reference["input_dq"])

        df = pd.read_csv(csv_path, keep_default_na=False)
        assert csv_path == working_directory / "layerwise.csv"
        assert df.columns.tolist() == LAYERWISE_COLUMNS
        assert df["Layer Status"].tolist() == [ActivationStatus.SUCCESS] * 4
        assert df["Layer Name"].tolist() == ["conv1", "conv1", "relu1", "relu1"]
        assert df["Layer Output"].tolist() == ["conv1_out_q", "conv1_out_dq", "relu1_out_q", "output"]
        assert df["Layer Shape"].tolist() == ["(1, 4, 8, 8)"] * 4
        assert [float(v) for v in df["mse"]] == [0.0] * 4
        assert (working_directory / "subgraph_data" / "conv1" / "extracted_model.onnx").exists()

        report = json.loads(json_path.read_text(encoding="utf-8"))
        assert json_path == working_directory / "layerwise.json"
        assert report["header"]["algorithm"] == "layerwise"
        assert report["header"]["artifact_type"] == "LAYERWISE_SNOOPING_REPORT"
        assert report["layers"][0]["layer_status"] == "SUCCESS"
        assert report["layers"][0]["exception"] == ""
        assert mock_plot.call_args.kwargs["algorithm"] == Algorithm.LAYERWISE

    @pytest.mark.parametrize(
        ("build", "rows"),
        [
            pytest.param(
                _interleaved_branches_model,
                {"conv1": ["conv1_out_q", "conv1_out_dq"], "downsample": ["ds_out_q", "ds_out_dq"]},
                id="interleaved",
            ),
            pytest.param(_split_branches_model, {"split": ["a_q", "b_q", "a_dq", "b_dq"]}, id="split"),
        ],
    )
    def test_layers_follow_qdq_edges(
        self,
        layerwise_snooper: OrtQnnSnooper,
        make_model,
        save_model,
        tmp_path: Path,
        working_directory: Path,
        build,
        rows: dict[str, list[str]],
    ) -> None:
        # Each layer is extracted with and compared on its own Q/DQ outputs, whatever the node order.
        proto = build(make_model)
        model = save_model(proto, tmp_path / "model.onnx")
        rng = np.random.default_rng(5)
        tensors = ["input", *(name for node in proto.graph.node for name in node.output)]
        reference = {name: rng.random((1, 1, 2, 2), dtype=np.float32) for name in tensors}
        with (
            patch.object(OrtQnnInferenceEngine, "run", side_effect=_echo_run(reference)) as mock_run,
            patch(f"{MODULE}.plot_graphs"),
        ):
            csv_path, _ = _run_layerwise(
                layerwise_snooper, model, reference, working_directory, {"input": reference["input"]}
            )

        df = pd.read_csv(csv_path, keep_default_na=False)
        assert df["Layer Status"].tolist() == [ActivationStatus.SUCCESS] * len(df)
        assert df.groupby("Layer Name", sort=False)["Layer Output"].apply(list).to_dict() == rows
        for call, (layer, outputs) in zip(mock_run.call_args_list, rows.items(), strict=True):
            extracted = onnx.load(call[0][0].qdq_model)
            layers = [node.name for node in extracted.graph.node if node.op_type not in QDQ_OPS_INDEX_DEFAULT]
            assert layers == [layer]
            assert sorted(output.name for output in extracted.graph.output) == sorted(outputs)

    def test_skip_ops_and_filter(
        self,
        layerwise_snooper: OrtQnnSnooper,
        qdq_chain_model: Path,
        chain_reference: dict[str, np.ndarray],
        working_directory: Path,
        qdq_input_sample: dict[str, np.ndarray],
    ) -> None:
        fake = _fake_extractor_class(qdq_chain_model)
        with (
            patch(f"{MODULE}.OnnxSubgraphExtractor", fake),
            patch.object(OrtQnnInferenceEngine, "run", side_effect=_echo_run(chain_reference)),
            patch(f"{MODULE}.plot_graphs"),
        ):
            csv_path, _ = _run_layerwise(
                layerwise_snooper,
                qdq_chain_model,
                chain_reference,
                working_directory,
                qdq_input_sample,
                intermediates_layers_list=["Relu"],
            )
        df = pd.read_csv(csv_path)
        assert df["Layer Name"].unique().tolist() == ["relu1"]
        layerwise_snooper._logger.debug.assert_any_call("Skipping layer input_q as it is MATH INVARIANT")

    def test_skip_row_on_extraction_failure(
        self,
        layerwise_snooper: OrtQnnSnooper,
        qdq_chain_model: Path,
        chain_reference: dict[str, np.ndarray],
        working_directory: Path,
        qdq_input_sample: dict[str, np.ndarray],
    ) -> None:
        fake = _fake_extractor_class(qdq_chain_model, fail=True)
        with (
            patch(f"{MODULE}.OnnxSubgraphExtractor", fake),
            patch.object(OrtQnnInferenceEngine, "run") as mock_run,
            patch(f"{MODULE}.plot_graphs"),
        ):
            csv_path, _ = _run_layerwise(
                layerwise_snooper, qdq_chain_model, chain_reference, working_directory, qdq_input_sample
            )
        mock_run.assert_not_called()
        df = pd.read_csv(csv_path, keep_default_na=False)
        assert df["Layer Status"].tolist() == [ActivationStatus.SKIP] * 2
        assert df["Layer Output"].tolist() == ["conv1_out_dq", "output"]
        assert df["Exception"].tolist() == ["Extraction failed with error: boom"] * 2
        assert df["mse"].tolist() == ["NaN", "NaN"]
        assert df["Layer Shape"].tolist() == ["()", "()"]
        layerwise_snooper._logger.error.assert_any_call("Couldn't extract current layer, proceeding to next layer.")

    def test_inference_failure_row(
        self,
        layerwise_snooper: OrtQnnSnooper,
        qdq_chain_model: Path,
        chain_reference: dict[str, np.ndarray],
        working_directory: Path,
        qdq_input_sample: dict[str, np.ndarray],
    ) -> None:
        fake = _fake_extractor_class(qdq_chain_model)
        with (
            patch(f"{MODULE}.OnnxSubgraphExtractor", fake),
            patch.object(OrtQnnInferenceEngine, "run", side_effect=RuntimeError("session died")),
            patch(f"{MODULE}.plot_graphs"),
        ):
            csv_path, _ = _run_layerwise(
                layerwise_snooper, qdq_chain_model, chain_reference, working_directory, qdq_input_sample
            )
        df = pd.read_csv(csv_path, keep_default_na=False)
        assert df["Layer Status"].tolist() == [ActivationStatus.INFERENCE_FAILURE] * 2
        assert df["Exception"].iloc[0] == (
            "Couldn't execute extracted model, proceeding to next layer. Reason: ORT session error - session died"
        )

    def test_input_fetch_failure_is_inference_failure(
        self,
        layerwise_snooper: OrtQnnSnooper,
        qdq_chain_model: Path,
        chain_reference: dict[str, np.ndarray],
        working_directory: Path,
        qdq_input_sample: dict[str, np.ndarray],
    ) -> None:
        fake = _fake_extractor_class(qdq_chain_model, inputs={"conv1_out_dq": ["nowhere"], "output": ["nowhere"]})
        with (
            patch(f"{MODULE}.OnnxSubgraphExtractor", fake),
            patch.object(OrtQnnInferenceEngine, "run") as mock_run,
            patch(f"{MODULE}.plot_graphs"),
        ):
            csv_path, _ = _run_layerwise(
                layerwise_snooper, qdq_chain_model, chain_reference, working_directory, qdq_input_sample
            )
        mock_run.assert_not_called()
        df = pd.read_csv(csv_path, keep_default_na=False)
        assert df["Exception"].iloc[0] == (
            "Couldn't execute extracted model, proceeding to next layer. Reason: Couldn't fetch input: nowhere"
        )

    def test_verification_failure_rows(
        self,
        layerwise_snooper: OrtQnnSnooper,
        qdq_chain_model: Path,
        chain_reference: dict[str, np.ndarray],
        working_directory: Path,
        qdq_input_sample: dict[str, np.ndarray],
    ) -> None:
        fake = _fake_extractor_class(qdq_chain_model)
        missing = OrtQnnInferenceEngineOutputConfig(output_data={"ghost": np.ones(4, np.float32)})
        with (
            patch(f"{MODULE}.OnnxSubgraphExtractor", fake),
            patch.object(OrtQnnInferenceEngine, "run", return_value=missing),
            patch(f"{MODULE}.plot_graphs"),
        ):
            csv_path, _ = _run_layerwise(
                layerwise_snooper, qdq_chain_model, chain_reference, working_directory, qdq_input_sample
            )
        df = pd.read_csv(csv_path, keep_default_na=False)
        assert df["Layer Status"].tolist() == [ActivationStatus.VERIFICATION_FAILURE] * 2
        assert df["Exception"].iloc[0] == (
            "Couldn't compare outputs, proceeding to next layer. Reason: Couldn't fetch golden reference for: ghost"
        )

    def test_empty_verification_row(
        self,
        layerwise_snooper: OrtQnnSnooper,
        qdq_chain_model: Path,
        chain_reference: dict[str, np.ndarray],
        working_directory: Path,
        qdq_input_sample: dict[str, np.ndarray],
    ) -> None:
        fake = _fake_extractor_class(qdq_chain_model)
        with (
            patch(f"{MODULE}.OnnxSubgraphExtractor", fake),
            patch.object(OrtQnnInferenceEngine, "run", side_effect=_echo_run(chain_reference)),
            patch(f"{MODULE}.verify", return_value={}),
            patch(f"{MODULE}.plot_graphs"),
        ):
            csv_path, _ = _run_layerwise(
                layerwise_snooper, qdq_chain_model, chain_reference, working_directory, qdq_input_sample
            )
        df = pd.read_csv(csv_path, keep_default_na=False)
        assert df["Layer Status"].tolist() == [ActivationStatus.VERIFICATION_FAILURE] * 2
        assert df["Exception"].tolist() == ["No comparable tensors between target and reference outputs."] * 2

    def test_csv_written_after_every_row(
        self,
        layerwise_snooper: OrtQnnSnooper,
        qdq_chain_model: Path,
        chain_reference: dict[str, np.ndarray],
        working_directory: Path,
        qdq_input_sample: dict[str, np.ndarray],
    ) -> None:
        fake = _fake_extractor_class(qdq_chain_model)
        echo = _echo_run(chain_reference)
        csv_path = working_directory / "layerwise.csv"
        csv_path.write_text("stale content\n")
        rows_seen: list[int] = []

        def run(config: Any) -> OrtQnnInferenceEngineOutputConfig:
            rows_seen.append(len(pd.read_csv(csv_path)) if csv_path.exists() else 0)
            return echo(config)

        with (
            patch(f"{MODULE}.OnnxSubgraphExtractor", fake),
            patch.object(OrtQnnInferenceEngine, "run", side_effect=run),
            patch(f"{MODULE}.plot_graphs"),
        ):
            _run_layerwise(layerwise_snooper, qdq_chain_model, chain_reference, working_directory, qdq_input_sample)
        assert rows_seen == [0, 2]
        assert len(pd.read_csv(csv_path)) == 4

    @pytest.mark.parametrize(("cpu_layers", "expected"), [(["conv1"], [True, False]), (["Relu"], [False, True])])
    def test_use_cpu_by_name_and_op_type(
        self,
        layerwise_snooper: OrtQnnSnooper,
        qdq_chain_model: Path,
        chain_reference: dict[str, np.ndarray],
        working_directory: Path,
        qdq_input_sample: dict[str, np.ndarray],
        cpu_layers: list[str],
        expected: list[bool],
    ) -> None:
        fake = _fake_extractor_class(qdq_chain_model)
        with (
            patch(f"{MODULE}.OnnxSubgraphExtractor", fake),
            patch.object(OrtQnnInferenceEngine, "run", side_effect=_echo_run(chain_reference)) as mock_run,
            patch(f"{MODULE}.plot_graphs"),
        ):
            _run_layerwise(
                layerwise_snooper,
                qdq_chain_model,
                chain_reference,
                working_directory,
                qdq_input_sample,
                cpu_layers_list=cpu_layers,
            )
        assert [call[0][0].use_cpu for call in mock_run.call_args_list] == expected
        # CPU subgraphs keep their float inputs; QNN subgraphs get them quantized again.
        assert [call[3] for call in fake.instances[0].calls] == [not use_cpu for use_cpu in expected]

    @pytest.mark.parametrize(("cpu_layers", "uses_target"), [(["conv1"], True), ([], False)])
    def test_mixed_backend_accumulation(
        self,
        layerwise_snooper: OrtQnnSnooper,
        qdq_chain_model: Path,
        chain_reference: dict[str, np.ndarray],
        working_directory: Path,
        qdq_input_sample: dict[str, np.ndarray],
        cpu_layers: list[str],
        uses_target: bool,
    ) -> None:
        fake = _fake_extractor_class(qdq_chain_model)
        with (
            patch(f"{MODULE}.OnnxSubgraphExtractor", fake),
            patch.object(OrtQnnInferenceEngine, "run", side_effect=_echo_run(chain_reference, offset=1.0)) as mock_run,
            patch(f"{MODULE}.plot_graphs"),
        ):
            _run_layerwise(
                layerwise_snooper,
                qdq_chain_model,
                chain_reference,
                working_directory,
                qdq_input_sample,
                cpu_layers_list=cpu_layers,
            )
        relu_input = mock_run.call_args_list[1][0][0].input_sample["conv1_out_dq"]
        expected = chain_reference["conv1_out_dq"] + (np.float32(1.0) if uses_target else np.float32(0.0))
        np.testing.assert_array_equal(relu_input, expected)

    def test_sanitised_subgraph_dir(
        self,
        layerwise_snooper: OrtQnnSnooper,
        qdq_chain_model: Path,
        chain_reference: dict[str, np.ndarray],
        working_directory: Path,
        qdq_input_sample: dict[str, np.ndarray],
        tmp_path: Path,
    ) -> None:
        model = _renamed_chain(qdq_chain_model, tmp_path, {"conv1": "/conv1/Conv", "relu1": ""}, "renamed.onnx")
        fake = _fake_extractor_class(qdq_chain_model)
        with (
            patch(f"{MODULE}.OnnxSubgraphExtractor", fake),
            patch.object(OrtQnnInferenceEngine, "run", side_effect=_echo_run(chain_reference)),
            patch(f"{MODULE}.plot_graphs"),
        ):
            _run_layerwise(layerwise_snooper, model, chain_reference, working_directory, qdq_input_sample)
        out_dirs = [call[2] for call in fake.instances[0].calls]
        assert out_dirs == [
            working_directory / "subgraph_data" / "_conv1_Conv",
            working_directory / "subgraph_data" / "node_6",
        ]
        assert not Path("/conv1").exists()

    def test_duplicate_dir_names_get_index_suffix(
        self,
        layerwise_snooper: OrtQnnSnooper,
        qdq_chain_model: Path,
        chain_reference: dict[str, np.ndarray],
        working_directory: Path,
        qdq_input_sample: dict[str, np.ndarray],
        tmp_path: Path,
    ) -> None:
        model = _renamed_chain(qdq_chain_model, tmp_path, {"conv1": "a/b", "relu1": "a_b"}, "dup.onnx")
        fake = _fake_extractor_class(qdq_chain_model)
        with (
            patch(f"{MODULE}.OnnxSubgraphExtractor", fake),
            patch.object(OrtQnnInferenceEngine, "run", side_effect=_echo_run(chain_reference)),
            patch(f"{MODULE}.plot_graphs"),
        ):
            _run_layerwise(layerwise_snooper, model, chain_reference, working_directory, qdq_input_sample)
        assert [call[2].name for call in fake.instances[0].calls] == ["a_b", "a_b_6"]

    def test_layer_without_primary_input(
        self,
        layerwise_snooper: OrtQnnSnooper,
        qdq_chain_model: Path,
        chain_reference: dict[str, np.ndarray],
        working_directory: Path,
        qdq_input_sample: dict[str, np.ndarray],
        tmp_path: Path,
    ) -> None:
        model = onnx.load(qdq_chain_model)
        model.graph.node[6].input[0] = ""
        model_path = tmp_path / "no_input.onnx"
        onnx.save(model, model_path)
        fake = _fake_extractor_class(qdq_chain_model)
        with (
            patch(f"{MODULE}.OnnxSubgraphExtractor", fake),
            patch.object(OrtQnnInferenceEngine, "run", side_effect=_echo_run(chain_reference)),
            patch(f"{MODULE}.plot_graphs"),
        ):
            csv_path, _ = _run_layerwise(
                layerwise_snooper, model_path, chain_reference, working_directory, qdq_input_sample
            )
        df = pd.read_csv(csv_path, keep_default_na=False)
        relu_rows = df[df["Layer Name"] == "relu1"]
        assert relu_rows["Layer Status"].tolist() == [ActivationStatus.SKIP]
        assert relu_rows["Exception"].tolist() == ["Layer has no primary input."]
        assert relu_rows["Layer Output"].tolist() == ["output"]

    def test_zero_rows_writes_header_only_csv(
        self,
        layerwise_snooper: OrtQnnSnooper,
        qdq_chain_model: Path,
        chain_reference: dict[str, np.ndarray],
        working_directory: Path,
        qdq_input_sample: dict[str, np.ndarray],
    ) -> None:
        fake = _fake_extractor_class(qdq_chain_model)
        with (
            patch(f"{MODULE}.OnnxSubgraphExtractor", fake),
            patch.object(OrtQnnInferenceEngine, "run") as mock_run,
            patch(f"{MODULE}.plot_graphs"),
        ):
            csv_path, json_path = _run_layerwise(
                layerwise_snooper,
                qdq_chain_model,
                chain_reference,
                working_directory,
                qdq_input_sample,
                intermediates_layers_list=["nothing_matches"],
            )
        mock_run.assert_not_called()
        header_only = pd.read_csv(csv_path)
        assert header_only.columns.tolist() == LAYERWISE_COLUMNS
        assert header_only.empty
        assert json.loads(json_path.read_text())["layers"] == []
        OrtQnnSnooperOutputConfig(csv_snooping_report=csv_path, json_snooping_report=json_path)

    def test_fresh_pool_per_layer(
        self,
        mock_logger: MagicMock,
        qdq_chain_model: Path,
        chain_reference: dict[str, np.ndarray],
        working_directory: Path,
        qdq_input_sample: dict[str, np.ndarray],
    ) -> None:
        snooper = OrtQnnSnooper(mock_logger, name=Algorithm.LAYERWISE)  # isolate_inference=True
        mock_logger.name = "isolated_logger"
        mock_logger.handlers = []
        mock_logger.getEffectiveLevel.return_value = logging.DEBUG
        echo = _echo_run(chain_reference)
        fake = _fake_extractor_class(qdq_chain_model)

        def submit(fn: Any, config: Any, logger_name: str) -> MagicMock:
            assert logger_name == "isolated_logger"
            future = MagicMock()
            future.result.return_value = echo(config)
            return future

        with (
            patch(f"{MODULE}.OnnxSubgraphExtractor", fake),
            patch(f"{MODULE}.ProcessPoolExecutor") as mock_pool_cls,
            patch(f"{MODULE}.plot_graphs"),
        ):
            mock_pool_cls.return_value.__enter__.return_value.submit.side_effect = submit
            csv_path, _ = _run_layerwise(snooper, qdq_chain_model, chain_reference, working_directory, qdq_input_sample)
        assert mock_pool_cls.call_count == 2
        kwargs = mock_pool_cls.call_args.kwargs
        assert kwargs["max_workers"] == 1
        assert kwargs["mp_context"].get_start_method() == "spawn"
        assert kwargs["initializer"] is snooper_module.configure_worker_logging
        assert kwargs["initargs"] == ("isolated_logger", logging.DEBUG, None)
        submitted = mock_pool_cls.return_value.__enter__.return_value.submit.call_args_list
        assert all(call[0][0] is snooper_module.run_inference_job for call in submitted)
        assert pd.read_csv(csv_path)["Layer Status"].tolist() == [ActivationStatus.SUCCESS] * 4

    def test_pool_failure_becomes_ort_session_error(
        self, mock_logger: MagicMock, qdq_chain_model: Path, working_directory: Path
    ) -> None:
        snooper = OrtQnnSnooper(mock_logger, name=Algorithm.LAYERWISE)
        mock_logger.name = "isolated_logger"
        mock_logger.handlers = []
        mock_logger.getEffectiveLevel.return_value = logging.INFO
        with patch(f"{MODULE}.ProcessPoolExecutor") as mock_pool_cls:
            future = mock_pool_cls.return_value.__enter__.return_value.submit.return_value
            future.result.side_effect = OSError("worker crashed")
            with pytest.raises(RuntimeError, match="ORT session error - worker crashed"):
                snooper._run_inference(MagicMock())

    @patch(f"{MODULE}.plot_graphs")
    def test_layerwise_elementwise_lookup(
        self,
        _mock_plot: MagicMock,
        layerwise_snooper: OrtQnnSnooper,
        qdq_chain_model: Path,
        chain_reference: dict[str, np.ndarray],
        working_directory: Path,
        qdq_input_sample: dict[str, np.ndarray],
    ) -> None:
        fake = _fake_extractor_class(qdq_chain_model)
        with (
            patch(f"{MODULE}.OnnxSubgraphExtractor", fake),
            patch.object(OrtQnnInferenceEngine, "run", side_effect=_echo_run(chain_reference)),
            patch.object(OrtQnnSnooper, "_dump_elementwise_stats") as mock_dump,
        ):
            _run_layerwise(
                layerwise_snooper,
                qdq_chain_model,
                chain_reference,
                working_directory,
                qdq_input_sample,
                dump_elementwise_stats={"conv1": 3, "Relu": 1},
            )
        assert [call[0][2] for call in mock_dump.call_args_list] == [3, 1]


# =============================================================================
# Tensor mapping
# =============================================================================


class TestGetOnnxTensorMapping:
    def test_exact_match(self, snooper: OrtQnnSnooper) -> None:
        names = ["layer1", "layer2", "layer3"]
        assert snooper._get_onnx_tensor_mapping(names, names, reference_model_is_qdq=False) == {n: n for n in names}

    def test_prefix_match_fp32_reference(self, snooper: OrtQnnSnooper) -> None:
        result = snooper._get_onnx_tensor_mapping(
            ["conv1_output", "fc_output", "layer2"],
            ["conv1_output_quantized", "fc_output_qdq", "layer2_dequantized"],
            reference_model_is_qdq=False,
        )
        assert result == {
            "conv1_output_quantized": "conv1_output",
            "fc_output_qdq": "fc_output",
            "layer2_dequantized": "layer2",
        }

    def test_no_prefix_match_qdq_reference(self, snooper: OrtQnnSnooper) -> None:
        result = snooper._get_onnx_tensor_mapping(
            ["conv1_output", "fc_output"], ["conv1_output_quantized", "fc_output_qdq"], reference_model_is_qdq=True
        )
        assert result == {}

    def test_transformed_names(self, snooper: OrtQnnSnooper) -> None:
        result = snooper._get_onnx_tensor_mapping(
            ["_layer1_output", "_layer2_output"], ["/layer1/output", "/layer2/output"], reference_model_is_qdq=False
        )
        assert result == {"/layer1/output": "_layer1_output", "/layer2/output": "_layer2_output"}

    def test_transformed_prefix(self, snooper: OrtQnnSnooper) -> None:
        result = snooper._get_onnx_tensor_mapping(
            ["_layer1_output"], ["/layer1/output_quantized"], reference_model_is_qdq=False
        )
        assert result == {"/layer1/output_quantized": "_layer1_output"}

    def test_partial_match(self, snooper: OrtQnnSnooper) -> None:
        result = snooper._get_onnx_tensor_mapping(
            ["layer1", "layer2"], ["layer1_qdq", "layer3_qdq"], reference_model_is_qdq=False
        )
        assert result == {"layer1_qdq": "layer1"}
        snooper._logger.debug.assert_any_call("Unmapped target tensors: ['layer3_qdq']")

    def test_empty_inputs(self, snooper: OrtQnnSnooper) -> None:
        assert snooper._get_onnx_tensor_mapping([], [], reference_model_is_qdq=False) == {}

    def test_prevents_false_prefix_match(self, snooper: OrtQnnSnooper) -> None:
        assert snooper._get_onnx_tensor_mapping(["conv1"], ["conv10_output"], reference_model_is_qdq=False) == {}

    def test_valid_separators(self, snooper: OrtQnnSnooper) -> None:
        result = snooper._get_onnx_tensor_mapping(
            ["layer"], ["layer_output", "layer/output", "layer.output", "layer:output"], reference_model_is_qdq=False
        )
        assert len(result) == 4
        assert all(v == "layer" for v in result.values())

    def test_mapping_log(self, snooper: OrtQnnSnooper) -> None:
        snooper._get_onnx_tensor_mapping(["a"], ["a", "b"], reference_model_is_qdq=True)
        snooper._logger.info.assert_any_call("Tensor mapping created: 1 out of 2 target tensors mapped to reference.")

    @pytest.mark.parametrize(("reference_is_qdq", "expected"), [(False, True), (True, False)])
    def test_prefix_behavior(self, snooper: OrtQnnSnooper, reference_is_qdq: bool, expected: bool) -> None:
        result = snooper._get_onnx_tensor_mapping(
            ["layer1"], ["layer1_quantized"], reference_model_is_qdq=reference_is_qdq
        )
        assert ("layer1_quantized" in result) is expected

    @pytest.mark.parametrize("separator", ["_", "/", ".", ":"])
    def test_separators(self, snooper: OrtQnnSnooper, separator: str) -> None:
        result = snooper._get_onnx_tensor_mapping(["layer"], [f"layer{separator}output"], reference_model_is_qdq=False)
        assert result == {f"layer{separator}output": "layer"}

    def test_mismatched_tensor_shapes(self, snooper: OrtQnnSnooper) -> None:
        assert snooper._get_onnx_tensor_mapping(["layer1"], ["layer1"], reference_model_is_qdq=False) == {
            "layer1": "layer1"
        }


# =============================================================================
# Activation info
# =============================================================================


class TestGetActivationsInfo:
    def test_basic(self, snooper: OrtQnnSnooper) -> None:
        result = get_activations_info({"layer1": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], np.float32)})
        assert result["layer1"].dtype == "float32"
        assert result["layer1"].shape == (2, 3)
        assert result["layer1"].distribution == (1.0, 6.0, 3.5)

    def test_multiple_tensors(self, snooper: OrtQnnSnooper) -> None:
        result = get_activations_info(
            {"layer1": np.array([1.0, 2.0, 3.0], np.float32), "layer2": np.array([10.0, 20.0, 30.0], np.float64)}
        )
        assert len(result) == 2
        assert result["layer1"].dtype == "float32"
        assert result["layer2"].dtype == "float64"
        assert result["layer1"].distribution[0] == 1.0
        assert result["layer2"].distribution[0] == 10.0

    def test_empty_array(self, snooper: OrtQnnSnooper) -> None:
        result = get_activations_info({"layer1": np.array([], np.float32)})
        assert result["layer1"].shape == (0,)
        assert all(np.isnan(v) for v in result["layer1"].distribution)

    def test_empty_dict(self, snooper: OrtQnnSnooper) -> None:
        with pytest.raises(ValueError, match="outputs dictionary cannot be empty"):
            get_activations_info({})

    def test_invalid_type(self, snooper: OrtQnnSnooper) -> None:
        with pytest.raises(TypeError, match="Expected np.ndarray"):
            get_activations_info({"layer1": [1, 2, 3]})

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64, np.uint8])
    def test_dtypes(self, snooper: OrtQnnSnooper, dtype: type) -> None:
        result = get_activations_info({"layer1": np.array([1, 2, 3], dtype=dtype)})
        assert result["layer1"].dtype == np.dtype(dtype).name


# =============================================================================
# Report generation
# =============================================================================


class TestGenerateSnoopingReport:
    def _activation_info(self, *names: str) -> dict[str, ActivationInfo]:
        return {
            name: ActivationInfo(dtype="float32", shape=(1, 64, 112, 112), distribution=(0.0, 1.0, 0.5))
            for name in names
        }

    @patch.object(OrtQnnSnooper, "_get_qdq_parent_map", return_value=None)
    @patch(f"{MODULE}.dump_json_report")
    def test_basic(
        self,
        mock_dump_json: MagicMock,
        _mock_parent: MagicMock,
        snooper: OrtQnnSnooper,
        tmp_path: Path,
        qdq_chain_model: Path,
    ) -> None:
        mock_dump_json.return_value = tmp_path / "report.json"
        csv_path, json_path = snooper._generate_snooping_report(
            {("layer1_qdq", "layer1"): {"mse": 0.01, "cosine": 45.2}},
            [get_comparator("mse"), get_comparator("cosine")],
            tmp_path,
            self._activation_info("layer1"),
            self._activation_info("layer1_qdq"),
            qdq_chain_model,
        )
        assert csv_path == tmp_path / "oneshot.csv"
        assert json_path == tmp_path / "report.json"
        df = pd.read_csv(csv_path)
        assert {"Layer Name", "Layer Output", "mse", "cosine"} <= set(df.columns)
        assert len(df) == 1
        assert df.iloc[0]["Source(Min, Max, Median)"] == "(0.0, 1.0, 0.5)"

    @patch.object(OrtQnnSnooper, "_get_qdq_parent_map")
    @patch(f"{MODULE}.dump_json_report")
    def test_with_parent_op(
        self,
        mock_dump_json: MagicMock,
        mock_parent: MagicMock,
        snooper: OrtQnnSnooper,
        tmp_path: Path,
        qdq_chain_model: Path,
    ) -> None:
        mock_parent.return_value = _node("conv1", "Conv", [])
        mock_dump_json.return_value = tmp_path / "report.json"
        info = self._activation_info("layer1", "layer1_qdq")
        csv_path, _ = snooper._generate_snooping_report(
            {("layer1_qdq", "layer1"): {"mse": 0.01}}, [get_comparator("mse")], tmp_path, info, info, qdq_chain_model
        )
        df = pd.read_csv(csv_path)
        assert df.iloc[0]["Layer Name"] == "conv1"
        assert df.iloc[0]["Layer Type"] == "Conv"

    def test_real_parent_lookup_and_missing_reference(
        self, snooper: OrtQnnSnooper, tmp_path: Path, qdq_chain_model: Path
    ) -> None:
        info = self._activation_info("conv1_out_dq", "output")
        csv_path, json_path = snooper._generate_snooping_report(
            {("conv1_out_dq", "conv1_out_dq"): {"mse": 0.5}, ("output", "nope"): {"mse": 0.25}},
            [get_comparator("mse")],
            tmp_path,
            {"conv1_out_dq": info["conv1_out_dq"]},
            info,
            onnx.load(qdq_chain_model),
        )
        df = pd.read_csv(csv_path)
        assert df["Layer Name"].tolist() == ["conv1", "relu1"]
        assert df["Layer Type"].tolist() == ["Conv", "Relu"]
        assert df["Source(Min, Max, Median)"].tolist() == ["(0.0, 1.0, 0.5)", "()"]
        assert json_path == tmp_path / "oneshot.json"

    @patch(f"{MODULE}.dump_json_report")
    def test_generate_json_report(self, mock_dump_json: MagicMock, snooper: OrtQnnSnooper, tmp_path: Path) -> None:
        mock_dump_json.return_value = tmp_path / "oneshot.json"
        df = pd.DataFrame({"Layer Name": ["conv1"], "Layer Output": ["conv1_output"], "mse": [0.01]})
        assert snooper._generate_json_report(df, tmp_path) == tmp_path / "oneshot.json"
        mock_dump_json.assert_called_once_with(df, tmp_path / "oneshot.json", Algorithm.ONESHOT)

    @patch(f"{MODULE}.dump_json_report")
    def test_generate_json_report_layerwise(
        self, mock_dump_json: MagicMock, snooper: OrtQnnSnooper, tmp_path: Path
    ) -> None:
        df = pd.DataFrame()
        snooper._generate_json_report(df, tmp_path, Algorithm.LAYERWISE)
        mock_dump_json.assert_called_once_with(df, tmp_path / "layerwise.json", Algorithm.LAYERWISE)


# =============================================================================
# Parent node mapping
# =============================================================================


class TestGetQdqParentMap:
    def test_dequant_returns_op(self, snooper: OrtQnnSnooper) -> None:
        conv = _node("conv1", "Conv", ["conv_out"], ["input"])
        nodes = [
            conv,
            _node("quant1", "QuantizeLinear", ["conv_out_q"], ["conv_out"]),
            _node("dq1", "DequantizeLinear", ["dq"], ["conv_out_q"]),
        ]
        assert snooper._get_qdq_parent_map("dq", nodes) is conv

    def test_quant_returns_op(self, snooper: OrtQnnSnooper) -> None:
        conv = _node("conv1", "Conv", ["conv_out"], ["input"])
        nodes = [conv, _node("quant1", "QuantizeLinear", ["conv_out_q"], ["conv_out"])]
        assert snooper._get_qdq_parent_map("conv_out_q", nodes) is conv

    def test_interleaved_branches_follow_edges(self, snooper: OrtQnnSnooper) -> None:
        # A residual block start: both Convs precede their Q/DQ nodes, so node positions would name the wrong Conv.
        nodes = [
            _node("conv1", "Conv", ["conv1_out"], ["block_in"]),
            _node("downsample", "Conv", ["ds_out"], ["block_in"]),
            _node("conv1_q", "QuantizeLinear", ["conv1_out_q"], ["conv1_out"]),
            _node("ds_q", "QuantizeLinear", ["ds_out_q"], ["ds_out"]),
            _node("conv1_dq", "DequantizeLinear", ["conv1_out_dq"], ["conv1_out_q"]),
            _node("ds_dq", "DequantizeLinear", ["ds_out_dq"], ["ds_out_q"]),
        ]
        tensors = ["conv1_out_q", "ds_out_q", "conv1_out_dq", "ds_out_dq"]
        parents = [snooper._get_qdq_parent_map(tensor, nodes).name for tensor in tensors]
        assert parents == ["conv1", "downsample", "conv1", "downsample"]

    def test_requantization_chain_returns_op(self, snooper: OrtQnnSnooper) -> None:
        relu = _node("relu1", "Relu", ["relu_out"], ["input"])
        nodes = [
            relu,
            _node("q1", "QuantizeLinear", ["q1_out"], ["relu_out"]),
            _node("dq1", "DequantizeLinear", ["dq1_out"], ["q1_out"]),
            _node("q2", "QuantizeLinear", ["q2_out"], ["dq1_out"]),
            _node("dq2", "DequantizeLinear", ["dq2_out"], ["q2_out"]),
        ]
        assert snooper._get_qdq_parent_map("dq2_out", nodes) is relu

    def test_non_qdq_returns_self(self, snooper: OrtQnnSnooper) -> None:
        relu = _node("relu1", "Relu", ["relu_out"], ["input"])
        assert snooper._get_qdq_parent_map("relu_out", [relu]) is relu

    def test_graph_input_chain_returns_producer(self, snooper: OrtQnnSnooper) -> None:
        # The chain quantizes a graph input; the weight DQ before it in node order is unrelated.
        nodes = [
            _node("w_dq", "DequantizeLinear", ["w"], ["w_q"]),
            _node("input_quant", "QuantizeLinear", ["input_q"], ["input"]),
            _node("input_dequant", "DequantizeLinear", ["input_dq"], ["input_q"]),
        ]
        assert snooper._get_qdq_parent_map("input_q", nodes) is nodes[1]
        assert snooper._get_qdq_parent_map("input_dq", nodes) is nodes[2]

    def test_early_dequant_returns_self(self, snooper: OrtQnnSnooper) -> None:
        dq = _node("dequant1", "DequantizeLinear", ["output"], ["input_q"])
        assert snooper._get_qdq_parent_map("output", [dq]) is dq

    def test_early_quant_returns_self(self, snooper: OrtQnnSnooper) -> None:
        q = _node("quant1", "QuantizeLinear", ["output"], ["input"])
        assert snooper._get_qdq_parent_map("output", [q]) is q

    def test_cycle_returns_producer(self, snooper: OrtQnnSnooper) -> None:
        nodes = [
            _node("q", "QuantizeLinear", ["q_out"], ["dq_out"]),
            _node("dq", "DequantizeLinear", ["dq_out"], ["q_out"]),
        ]
        assert snooper._get_qdq_parent_map("dq_out", nodes) is nodes[1]

    def test_tensor_not_found(self, snooper: OrtQnnSnooper) -> None:
        assert snooper._get_qdq_parent_map("missing", [_node("conv1", "Conv", ["conv_out"], ["input"])]) is None

    def test_with_output_index(self, snooper: OrtQnnSnooper) -> None:
        conv = _node("conv1", "Conv", ["conv_out"], ["input"])
        nodes = [
            conv,
            _node("q", "QuantizeLinear", ["q_out"], ["conv_out"]),
            _node("dq", "DequantizeLinear", ["dq_out"], ["q_out"]),
        ]
        index = {"conv_out": 0, "q_out": 1, "dq_out": 2}
        assert snooper._get_qdq_parent_map("dq_out", nodes, index) is conv
        assert snooper._get_qdq_parent_map("missing", nodes, index) is None


# =============================================================================
# Subgraph inputs and layerwise rows
# =============================================================================


class TestGetSubgraphInputs:
    def test_precedence_and_transform(self, snooper: OrtQnnSnooper) -> None:
        result = snooper._get_subgraph_inputs(
            ["a", "b", "c", "/e"],
            {"a": np.array([1], np.float32)},
            {"b": np.array([2], np.float32)},
            {"c": np.array([3], np.float32), "_e": np.array([5], np.float32), "b": np.array([9], np.float32)},
            mixed_backend=True,
        )
        assert {k: v.tolist() for k, v in result.items()} == {"a": [1], "b": [2], "c": [3], "/e": [5]}

    def test_target_ignored_without_mixed_backend(self, snooper: OrtQnnSnooper) -> None:
        result = snooper._get_subgraph_inputs(
            ["b"], {}, {"b": np.array([2], np.float32)}, {"b": np.array([9], np.float32)}, mixed_backend=False
        )
        assert result["b"].tolist() == [9]

    def test_missing_raises(self, snooper: OrtQnnSnooper) -> None:
        with pytest.raises(Exception, match="Couldn't fetch input") as excinfo:
            snooper._get_subgraph_inputs(["missing"], {}, {}, {}, mixed_backend=False)
        assert str(excinfo.value) == "Couldn't fetch input: missing"


class TestBuildRows:
    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(columns=[*LAYERWISE_COLUMNS[:-2], "mse", "cosine", "Exception"])

    def test_build_row_success(self, snooper: OrtQnnSnooper) -> None:
        df = snooper._build_data_frame_for_subgraph(
            status=ActivationStatus.SUCCESS,
            layer_name="conv1",
            output_name="conv1_dq",
            verifier_scores={"mse": 0.01, "cosine": 0.9},
            layer_type="Conv",
            layer_shape=(1, 64, 56, 56),
            source_distribution=(0.0, 1.0, 0.5),
            target_distribution=(0.0, 1.0, 0.5),
            comparator_names=["mse", "cosine"],
            data_frame=self._frame(),
        )
        assert len(df) == 1
        assert df.iloc[0]["Layer Name"] == "conv1"
        assert df.iloc[0]["mse"] == 0.01
        assert df.iloc[0]["cosine"] == 0.9
        assert df.iloc[0]["Exception"] == ""

    def test_build_row_non_success_nan_metrics(self, snooper: OrtQnnSnooper) -> None:
        df = snooper._build_data_frame_for_subgraph(
            status=ActivationStatus.SKIP,
            layer_name="conv1",
            output_name="conv1_dq",
            verifier_scores=None,
            layer_type="Conv",
            layer_shape=(),
            source_distribution=(),
            target_distribution=(),
            comparator_names=["mse", "cosine"],
            data_frame=self._frame(),
            exception="Extraction failed",
        )
        assert df.iloc[0]["mse"] == "NaN"
        assert df.iloc[0]["cosine"] == "NaN"
        assert df.iloc[0]["Exception"] == "Extraction failed"

    def test_layerwise_row_scalar_conversion(self, snooper: OrtQnnSnooper) -> None:
        row = snooper._build_layerwise_row(
            status=ActivationStatus.SUCCESS,
            layer_name="l",
            output_name="o",
            verifier_scores={"mae": np.float32(0.00474)},
            layer_type="Conv",
            layer_shape=(1,),
            source_distribution=(),
            target_distribution=(),
            comparator_names=["mae"],
        )
        assert type(row["mae"]) is float
        assert str(row["mae"]) == "0.00474"
        assert list(row) == [*LAYERWISE_COLUMNS[:-2], "mae", "Exception"]


# =============================================================================
# Elementwise stats
# =============================================================================


class TestDumpElementwiseStats:
    def test_creates_log_when_exceeds_threshold(self, snooper: OrtQnnSnooper, tmp_path: Path) -> None:
        reference = {"out": np.array([0.0, 1.0, 3.0], np.float32)}
        target = {"out": np.array([0.0, 4.0, 3.0], np.float32)}
        snooper._dump_elementwise_stats(reference, target, steps_allowed="2", working_directory=tmp_path)
        content = (tmp_path / "elementwise_stats.log").read_text()
        assert content == (
            "Starting Elementwise comparison for out\n"
            "Difference=3.0 at index (1,), Steps allowed=2.0, Target value: 4.0, Reference value: 1.0\n"
            "Total number of elements that are further than 2.0 steps: 1\n"
            "Maximum step difference: 3.0\n\n\n"
        )

    def test_appends(self, snooper: OrtQnnSnooper, tmp_path: Path) -> None:
        reference = {"out": np.array([0.0], np.float32)}
        target = {"out": np.array([5.0], np.float32)}
        snooper._dump_elementwise_stats(reference, target, 1, tmp_path)
        snooper._dump_elementwise_stats(reference, target, 1, tmp_path)
        assert (tmp_path / "elementwise_stats.log").read_text().count("Starting Elementwise comparison") == 2

    def test_uint8_no_wraparound_and_2d_index(self, snooper: OrtQnnSnooper, tmp_path: Path) -> None:
        reference = {"q": np.array([[10, 200], [3, 3]], np.uint8)}
        target = {"q": np.array([[5, 200], [3, 0]], np.uint8)}
        snooper._dump_elementwise_stats(reference, target, steps_allowed=2, working_directory=tmp_path)
        assert (tmp_path / "elementwise_stats.log").read_text() == (
            "Starting Elementwise comparison for q\n"
            "Difference=5 at index (0, 0), Steps allowed=2.0, Target value: 5, Reference value: 10\n"
            "Difference=3 at index (1, 1), Steps allowed=2.0, Target value: 0, Reference value: 3\n"
            "Total number of elements that are further than 2.0 steps: 2\n"
            "Maximum step difference: 5\n\n\n"
        )

    def test_no_file_when_within_threshold(self, snooper: OrtQnnSnooper, tmp_path: Path) -> None:
        values = {"out": np.array([1.0, 2.0], np.float32)}
        snooper._dump_elementwise_stats(values, values, 0, tmp_path)
        assert not (tmp_path / "elementwise_stats.log").exists()

    def test_invalid_steps_and_mismatched_shapes(self, snooper: OrtQnnSnooper, tmp_path: Path) -> None:
        reference = {"out": np.array([0.0, 1.0], np.float32)}
        target = {"out": np.array([0.0], np.float32)}
        snooper._dump_elementwise_stats(reference, target, steps_allowed="abc", working_directory=tmp_path)
        assert not (tmp_path / "elementwise_stats.log").exists()
        snooper._logger.warning.assert_any_call("Invalid steps_allowed='abc'. Falling back to 0.")

    def test_missing_reference_is_skipped(self, snooper: OrtQnnSnooper, tmp_path: Path) -> None:
        snooper._dump_elementwise_stats({}, {"out": np.ones(2, np.float32)}, 0, tmp_path)
        snooper._logger.warning.assert_any_call(
            "Skipping elementwise stats for 'out' as it is missing in reference/target outputs."
        )


# =============================================================================
# _get_valid_intermediates
# =============================================================================


class TestGetValidIntermediates:
    def test_fp32_all_outputs_when_no_filter(
        self, snooper: OrtQnnSnooper, make_model, save_model, tmp_path: Path
    ) -> None:
        path = _conv_qdq_model_named(make_model, save_model, tmp_path / "m.onnx")
        result = snooper._get_valid_intermediates([], path, is_qdq_model=False)
        assert result == ["conv_out", "conv_out_q", "output"]

    def test_none_equals_empty(self, snooper: OrtQnnSnooper, make_model, save_model, tmp_path: Path) -> None:
        path = _conv_qdq_model_named(make_model, save_model, tmp_path / "m.onnx")
        assert snooper._get_valid_intermediates(None, path, True) == snooper._get_valid_intermediates([], path, True)

    def test_qdq_dequant_only_with_no_filter(
        self, snooper: OrtQnnSnooper, make_model, save_model, tmp_path: Path
    ) -> None:
        path = _qdq_model_basic(make_model, save_model, tmp_path / "m.onnx")
        assert snooper._get_valid_intermediates([], path, True, {"DequantizeLinear": 2}) == ["output"]

    def test_qdq_filter_by_quant_offset(self, snooper: OrtQnnSnooper, make_model, save_model, tmp_path: Path) -> None:
        path = _qdq_model_basic(make_model, save_model, tmp_path / "m.onnx")
        assert snooper._get_valid_intermediates(["QuantizeLinear"], path, True, {"DequantizeLinear": 1}) == ["output"]

    def test_qdq_from_op_to_dq(self, snooper: OrtQnnSnooper, make_model, save_model, tmp_path: Path) -> None:
        path = _conv_qdq_model_named(make_model, save_model, tmp_path / "m.onnx")
        assert snooper._get_valid_intermediates(["Conv"], path, True, {"DequantizeLinear": 2}) == ["output"]

    def test_default_index_and_proto(self, snooper: OrtQnnSnooper, qdq_chain_model: Path) -> None:
        proto = onnx.load(qdq_chain_model)
        assert snooper._get_valid_intermediates(["conv1"], proto, True) == ["conv1_out_q", "conv1_out_dq"]

    def test_quant_without_dequant_is_captured(
        self, snooper: OrtQnnSnooper, make_model, save_model, tmp_path: Path
    ) -> None:
        nodes = [
            helper.make_node("Conv", ["input", "w"], ["conv_out"], name="conv1"),
            helper.make_node("QuantizeLinear", ["conv_out", "s", "z"], ["conv_out_q"], name="quant1"),
        ]
        model = make_model(
            nodes,
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("conv_out_q", TensorProto.UINT8, [1])],
        )
        path = save_model(model, tmp_path / "m.onnx")
        assert snooper._get_valid_intermediates(["conv1"], path, True) == ["conv_out_q"]
        snooper._logger.warning.assert_not_called()

    def test_interleaved_branches_select_own_qdq(self, snooper: OrtQnnSnooper, make_model) -> None:
        # A residual block start: both Convs precede their Q/DQ nodes, which node positions would mix up.
        proto = _interleaved_branches_model(make_model)
        assert snooper._get_valid_intermediates(["conv1"], proto, True) == ["conv1_out_q", "conv1_out_dq"]
        assert snooper._get_valid_intermediates(["downsample"], proto, True) == ["ds_out_q", "ds_out_dq"]
        assert snooper._get_valid_intermediates(["conv1"], proto, True, {"DequantizeLinear": 2}) == ["conv1_out_dq"]

    def test_requantization_chain_is_followed(self, snooper: OrtQnnSnooper, make_model) -> None:
        q, dq = "QuantizeLinear", "DequantizeLinear"
        nodes = [
            helper.make_node("Relu", ["input"], ["relu_out"], name="relu1"),
            helper.make_node(q, ["relu_out", "s", "z"], ["q1_out"]),
            helper.make_node(dq, ["q1_out", "s", "z"], ["dq1_out"]),
            helper.make_node(q, ["dq1_out", "s2", "z"], ["q2_out"]),
            helper.make_node(dq, ["q2_out", "s2", "z"], ["output"]),
        ]
        model = make_model(
            nodes,
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
            [
                helper.make_tensor("s", TensorProto.FLOAT, [], [0.1]),
                helper.make_tensor("s2", TensorProto.FLOAT, [], [0.2]),
                helper.make_tensor("z", TensorProto.UINT8, [], [0]),
            ],
        )
        expected = ["q1_out", "dq1_out", "q2_out", "output"]
        assert snooper._get_valid_intermediates(["relu1"], model, True) == expected

    def test_every_output_branch_is_selected(self, snooper: OrtQnnSnooper, make_model) -> None:
        proto = _split_branches_model(make_model)
        assert snooper._get_valid_intermediates(["split"], proto, True) == ["a_q", "b_q", "a_dq", "b_dq"]

    def test_qdq_skips_weight_dq(self, snooper: OrtQnnSnooper, qdq_chain_model: Path) -> None:
        # "w_dq" dequantizes the initializer "w_q": a weight, not an activation.
        expected = [name for name in CHAIN_TENSORS if name != "w_dq"]
        assert snooper._get_valid_intermediates([], qdq_chain_model, True) == expected

    @pytest.mark.parametrize(
        "weight",
        [
            "initializer",
            "sparse",
            "constant",
            "quantized",
            "transposed",
            "scaled",
            "reshaped",
            "concatenated",
            "exponentiated",
        ],
    )
    def test_qdq_skips_weight_and_bias_qdq(self, snooper: OrtQnnSnooper, make_model, weight: str) -> None:
        # QNN EP does not take a weight or bias DQ that generates a graph output; only activations are captured.
        proto = _conv_weight_qdq_model(make_model, weight)
        assert snooper._get_valid_intermediates([], proto, True) == ["input_q", "input_dq", "conv_out_q", "output"]
        assert snooper._get_valid_intermediates([], proto, True, {"DequantizeLinear": 2}) == ["input_dq", "output"]

    @pytest.mark.parametrize("weight", ["matmul", "relu_chain"])
    def test_qdq_keeps_qdq_computed_from_weights(self, snooper: OrtQnnSnooper, make_model, weight: str) -> None:
        # QNN EP computes "w" from the dequantized weights, so the Q/DQ of "w" are activations.
        proto = _conv_weight_qdq_model(make_model, weight)
        expected = ["w_q", "w_dq", "input_q", "input_dq", "conv_out_q", "output"]
        assert snooper._get_valid_intermediates([], proto, True) == expected

    def test_qdq_keeps_weight_qdq_when_asked(self, snooper: OrtQnnSnooper, qdq_chain_model: Path) -> None:
        assert snooper._get_valid_intermediates([], qdq_chain_model, True, skip_constant_tensors=False) == CHAIN_TENSORS

    def test_constant_operand_keeps_activation(self, snooper: OrtQnnSnooper, make_model) -> None:
        # "one" is constant, but the Sub also reads the activation "input_dq".
        q, dq = "QuantizeLinear", "DequantizeLinear"
        nodes = [
            helper.make_node(q, ["input", "s", "z"], ["input_q"]),
            helper.make_node(dq, ["input_q", "s", "z"], ["input_dq"]),
            helper.make_node("Sub", ["one", "input_dq"], ["sub_out"], name="sub"),
            helper.make_node(q, ["sub_out", "s", "z"], ["sub_out_q"]),
            helper.make_node(dq, ["sub_out_q", "s", "z"], ["output"]),
        ]
        model = make_model(
            nodes,
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
            [
                helper.make_tensor("s", TensorProto.FLOAT, [], [0.1]),
                helper.make_tensor("z", TensorProto.UINT8, [], [0]),
                helper.make_tensor("one", TensorProto.FLOAT, [], [1.0]),
            ],
        )
        assert snooper._get_valid_intermediates([], model, True) == ["input_q", "input_dq", "sub_out_q", "output"]

    def test_fp32_keeps_constant_tensors(self, snooper: OrtQnnSnooper, make_model) -> None:
        # Only QDQ selections drop constant tensors.
        nodes = [
            helper.make_node("Constant", [], ["c"], value=helper.make_tensor("c", TensorProto.FLOAT, [], [1.0])),
            helper.make_node("Transpose", ["w_t"], ["w"], perm=[1, 0, 2, 3]),
            helper.make_node("Conv", ["input", "w"], ["conv_out"], name="conv1"),
            helper.make_node("Add", ["conv_out", "c"], ["output"]),
        ]
        model = make_model(
            nodes,
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 2, 2])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 2, 2])],
            [helper.make_tensor("w_t", TensorProto.FLOAT, [1, 2, 1, 1], [0.01, 0.02])],
        )
        assert snooper._get_valid_intermediates(None, model, False) == ["c", "w", "conv_out", "output"]

    def test_layer_list_skips_weight_qdq(self, snooper: OrtQnnSnooper, make_model) -> None:
        proto = _conv_weight_qdq_model(make_model, "transposed")
        # The weight Transpose feeds the weight Q -> DQ, which are not captured.
        assert snooper._get_valid_intermediates(["Transpose", "conv1"], proto, True) == ["conv_out_q", "output"]

    def test_interleaved_weight_dq_is_skipped(self, snooper: OrtQnnSnooper, make_model) -> None:
        q, dq = "QuantizeLinear", "DequantizeLinear"
        nodes = [
            helper.make_node("Relu", ["input"], ["relu_out"], name="relu1"),
            helper.make_node(q, ["relu_out", "s", "z"], ["relu_out_q"]),
            # The Conv's weight DQ sits between the Relu's Q and DQ in node order; it is not captured.
            helper.make_node(dq, ["w_q", "s", "z"], ["w_dq"]),
            helper.make_node(dq, ["relu_out_q", "s", "z"], ["relu_out_dq"]),
            helper.make_node("Conv", ["relu_out_dq", "w_dq"], ["output"], name="conv1"),
        ]
        model = make_model(
            nodes,
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 2, 2])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 2, 2])],
            [
                helper.make_tensor("s", TensorProto.FLOAT, [], [0.1]),
                helper.make_tensor("z", TensorProto.UINT8, [], [0]),
                helper.make_tensor("w_q", TensorProto.UINT8, [1, 1, 1, 1], [1]),
            ],
        )
        assert snooper._get_valid_intermediates(["relu1"], model, True) == ["relu_out_q", "relu_out_dq"]

    def test_if_output_is_not_constant(self, snooper: OrtQnnSnooper, make_model) -> None:
        # The If's only input is constant, but its branches read the outer activation "input_dq".
        branch = helper.make_graph(
            [helper.make_node("Identity", ["input_dq"], ["branch_out"])],
            "branch",
            [],
            [helper.make_tensor_value_info("branch_out", TensorProto.FLOAT, [1])],
        )
        q, dq = "QuantizeLinear", "DequantizeLinear"
        nodes = [
            helper.make_node(q, ["input", "s", "z"], ["input_q"]),
            helper.make_node(dq, ["input_q", "s", "z"], ["input_dq"]),
            helper.make_node("If", ["cond"], ["if_out"], then_branch=branch, else_branch=branch),
            helper.make_node(q, ["if_out", "s", "z"], ["if_out_q"]),
            helper.make_node(dq, ["if_out_q", "s", "z"], ["output"]),
        ]
        model = make_model(
            nodes,
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
            [
                helper.make_tensor("s", TensorProto.FLOAT, [], [0.1]),
                helper.make_tensor("z", TensorProto.UINT8, [], [0]),
                helper.make_tensor("cond", TensorProto.BOOL, [], [True]),
            ],
        )
        assert snooper._get_valid_intermediates([], model, True) == ["input_q", "input_dq", "if_out_q", "output"]
