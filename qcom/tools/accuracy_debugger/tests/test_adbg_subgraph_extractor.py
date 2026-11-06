# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Unit tests for subgraph_extractor."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import onnx
import pytest
import subgraph_extractor
from common import SubgraphExtractionError
from conftest import DEBUGGER_DIR
from onnx import TensorProto, helper, numpy_helper
from subgraph_extractor import OnnxSubgraphExtractor, SubgraphExtractionResult

FLOAT = TensorProto.FLOAT


def _vi(name, shape=(1, 4), elem=FLOAT):
    return helper.make_tensor_value_info(name, elem, list(shape) if shape is not None else None)


def _node_names(path: Path) -> list[str]:
    return [n.name for n in onnx.load(str(path)).graph.node]


def _input_names(path: Path) -> list[str]:
    return [vi.name for vi in onnx.load(str(path)).graph.input]


def _requantized(tensor: str) -> list[str]:
    """Names of the QuantizeLinear -> DequantizeLinear pair that quantizes subgraph input ``tensor`` again."""
    return [f"{tensor}_requant_QuantizeLinear", f"{tensor}_requant_DequantizeLinear"]


def _weights_first(qdq_chain_model: Path, tmp_path: Path) -> Path:
    """``qdq_chain_model`` with the weight's DequantizeLinear first, so it precedes every extraction window."""
    model = onnx.load(str(qdq_chain_model))
    nodes = sorted(model.graph.node, key=lambda node: node.name != "w_dq")
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    path = tmp_path / "weights_first" / "weights_first.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path


def _cpu_outputs(path: Path, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Runs the model on ORT CPU without graph optimizations and returns every node output."""
    ort = pytest.importorskip("onnxruntime")
    model = onnx.load(str(path))
    outputs = {vi.name for vi in model.graph.output}
    model.graph.output.extend(
        onnx.ValueInfoProto(name=o) for n in model.graph.node for o in n.output if o and o not in outputs
    )
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # no QLinear fusion on either side
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    names = [o.name for o in session.get_outputs()]
    return dict(zip(names, session.run(names, feeds), strict=True))


def _assert_matches_model(model_path: Path, result: SubgraphExtractionResult, feeds: dict[str, np.ndarray]) -> None:
    """The subgraph, fed with the model's tensors, reproduces the model's tensors."""
    expected = _cpu_outputs(model_path, feeds)
    got = _cpu_outputs(result.model_path, {name: expected.get(name, feeds.get(name)) for name in result.input_names})
    for name in result.output_names:
        np.testing.assert_array_equal(got[name], expected[name])


@pytest.fixture
def residual_model(tmp_path, make_model, save_model) -> Path:
    """x -> a -> b -> c(b + x1) -> d(c + a) -> e (graph output) -> f(e + x2) -> y (graph output)."""
    nodes = [
        helper.make_node("Relu", ["x"], ["a"], name="relu_a"),
        helper.make_node("Neg", ["a"], ["b"], name="neg_b"),
        helper.make_node("Add", ["b", "x1"], ["c"], name="add_c"),
        helper.make_node("Add", ["c", "a"], ["d"], name="add_d"),
        helper.make_node("Neg", ["d"], ["e"], name="neg_e"),
        helper.make_node("Add", ["e", "x2"], ["f"], name="add_f"),
        helper.make_node("Relu", ["f"], ["y"], name="relu_y"),
    ]
    model = make_model(nodes, [_vi("x"), _vi("x1"), _vi("x2")], [_vi("e"), _vi("y")])
    return save_model(model, tmp_path / "res" / "res.onnx")


# 1
def test_linear_chain(qdq_chain_model, tmp_path):
    result = OnnxSubgraphExtractor(qdq_chain_model).extract("input_dq", "conv1_out_dq", tmp_path / "out")
    assert isinstance(result, SubgraphExtractionResult)
    assert result.input_names == ["input_dq"]
    assert result.output_names == ["conv1_out_dq"]
    assert _node_names(result.model_path) == ["w_dq", "conv1", "conv1_q", "conv1_dq"]


# 2
def test_residual_add_needs_prefix_tensor(residual_model, tmp_path):
    result = OnnxSubgraphExtractor(residual_model).extract("b", "d", tmp_path / "out")
    assert result.input_names == ["b", "x1", "a"]
    assert result.output_names == ["d"]
    assert _node_names(result.model_path) == ["add_c", "add_d"]


# 3
def test_graph_input_in_window_included_outside_pruned(residual_model, tmp_path):
    result = OnnxSubgraphExtractor(residual_model).extract("a", "c", tmp_path / "out")
    assert result.input_names == ["a", "x1"]
    assert "x2" not in _input_names(result.model_path)
    assert "x" not in result.input_names


# 4
def test_graph_output_inside_window_included_after_end_excluded(residual_model, tmp_path):
    result = OnnxSubgraphExtractor(residual_model).extract("b", "f", tmp_path / "out")
    assert result.output_names == ["f", "e"]
    assert "y" not in result.output_names
    assert result.input_names == ["b", "x1", "a", "x2"]


# 4b
def test_side_branch_without_value_info_is_not_an_input(tmp_path, make_model, save_model):
    nodes = [
        helper.make_node("Relu", ["x"], ["a"], name="relu_a"),
        helper.make_node("CustomOp", ["a"], ["side"], name="side", domain="custom.domain"),
        helper.make_node("Neg", ["a"], ["b"], name="neg_b"),
        helper.make_node("Abs", ["b"], ["c"], name="abs_c"),
        helper.make_node("Relu", ["side"], ["side2"], name="side2"),
        helper.make_node("Add", ["c", "side2"], ["y"], name="add_y"),
    ]
    model = make_model(nodes, [_vi("x")], [_vi("y", None)])
    model.opset_import.append(helper.make_opsetid("custom.domain", 1))
    path = save_model(model, tmp_path / "side.onnx")
    extractor = OnnxSubgraphExtractor(path, symbolic_shape_fallback=False)
    result = extractor.extract("a", "c", tmp_path / "out")
    assert result.input_names == ["a"]
    assert _node_names(result.model_path) == ["neg_b", "abs_c"]


# 4c
def test_graph_output_of_other_branch_in_window_excluded(tmp_path, make_model, save_model):
    nodes = [
        helper.make_node("Relu", ["x"], ["a"], name="relu_a"),
        helper.make_node("Neg", ["a"], ["b"], name="neg_b"),
        helper.make_node("Abs", ["a"], ["c"], name="abs_c"),
        helper.make_node("Neg", ["b"], ["d"], name="neg_d"),
        helper.make_node("Add", ["d", "c"], ["y"], name="add_y"),
    ]
    path = save_model(make_model(nodes, [_vi("x")], [_vi("c"), _vi("y")]), tmp_path / "branches.onnx")
    extractor = OnnxSubgraphExtractor(path)
    result = extractor.extract("a", "d", tmp_path / "out")
    assert result.output_names == ["d"]
    assert _node_names(result.model_path) == ["neg_b", "neg_d"]
    # Requested, the other branch's graph output is extracted with its branch.
    result = extractor.extract("a", "d", tmp_path / "requested", additional_outputs=["c"])
    assert result.output_names == ["d", "c"]
    assert _node_names(result.model_path) == ["neg_b", "abs_c", "neg_d"]


# 5
def test_invalid_window_raises(residual_model, tmp_path):
    extractor = OnnxSubgraphExtractor(residual_model)
    with pytest.raises(SubgraphExtractionError, match=r"^Start tensor 'nope' is not produced by any node or graph"):
        extractor.extract("nope", "d", tmp_path / "out")
    with pytest.raises(SubgraphExtractionError, match=r"^End tensor 'a' is not produced after start tensor 'd'\.$"):
        extractor.extract("d", "a", tmp_path / "out")
    with pytest.raises(SubgraphExtractionError, match="End tensor 'd'"):
        extractor.extract("d", "d", tmp_path / "out")
    with pytest.raises(SubgraphExtractionError, match="End tensor 'missing'"):
        extractor.extract("a", "missing", tmp_path / "out")


def test_extractor_failure_wrapped(tmp_path, make_model, save_model):
    nodes = [
        helper.make_node("CustomOp", ["x"], ["a"], name="custom", domain="custom.domain"),
        helper.make_node("Relu", ["a"], ["y"], name="relu"),
    ]
    model = make_model(nodes, [_vi("x")], [_vi("y", None)])
    model.opset_import.append(helper.make_opsetid("custom.domain", 1))
    path = save_model(model, tmp_path / "m.onnx")
    with pytest.raises(SubgraphExtractionError, match="a") as exc:
        OnnxSubgraphExtractor(path, symbolic_shape_fallback=False).extract("a", "y", tmp_path / "out")
    assert exc.value.__cause__ is not None


# 6
def test_start_is_graph_input(qdq_chain_model, tmp_path):
    result = OnnxSubgraphExtractor(qdq_chain_model).extract("input", "input_dq", tmp_path / "out")
    assert result.input_names == ["input"]
    assert _node_names(result.model_path) == ["input_q", "input_dq"]


# 7
def test_long_chain_no_recursion_error(tmp_path, make_model, save_model):
    count = 3000
    nodes = [helper.make_node("Relu", ["x" if i == 0 else f"t{i - 1}"], [f"t{i}"], name=f"n{i}") for i in range(count)]
    model = make_model(nodes, [_vi("x")], [_vi(f"t{count - 1}")])
    path = save_model(model, tmp_path / "long.onnx")
    result = OnnxSubgraphExtractor(path).extract("t0", f"t{count - 1}", tmp_path / "out")
    assert result.input_names == ["t0"]
    assert len(_node_names(result.model_path)) == count - 1


# 8
def test_empty_optional_names(tmp_path, make_model, save_model):
    nodes = [
        helper.make_node("Relu", ["x"], ["a"], name="relu"),
        helper.make_node("Clip", ["a", "", "mx"], ["b"], name="clip"),
        helper.make_node("Dropout", ["b"], ["c", ""], name="drop"),
        helper.make_node("Neg", ["c"], ["y"], name="neg"),
    ]
    init = numpy_helper.from_array(np.array(1.0, dtype=np.float32), "mx")
    model = make_model(nodes, [_vi("x")], [_vi("y")], [init])
    path = save_model(model, tmp_path / "opt.onnx")
    result = OnnxSubgraphExtractor(path).extract("a", "y", tmp_path / "out")
    assert result.input_names == ["a"]
    assert _node_names(result.model_path) == ["clip", "drop", "neg"]


# 9
def test_graph_input_that_is_initializer_is_constant(tmp_path, make_model, save_model):
    init = numpy_helper.from_array(np.ones((1, 4), dtype=np.float32), "bias")
    nodes = [
        helper.make_node("Relu", ["x"], ["a"], name="relu"),
        helper.make_node("Add", ["a", "bias"], ["b"], name="add"),
        helper.make_node("Neg", ["b"], ["y"], name="neg"),
    ]
    model = make_model(nodes, [_vi("x"), _vi("bias")], [_vi("y")], [init])
    path = save_model(model, tmp_path / "ib.onnx")
    extractor = OnnxSubgraphExtractor(path)
    result = extractor.extract("a", "y", tmp_path / "out")
    assert result.input_names == ["a"]
    with pytest.raises(SubgraphExtractionError, match="Start tensor 'bias'"):
        extractor.extract("bias", "y", tmp_path / "out2")


def _ms_qdq_model(make_model, with_unknown: bool) -> onnx.ModelProto:
    ms = "com.microsoft"
    inits = [
        numpy_helper.from_array(np.array(0.1, dtype=np.float32), "s"),
        numpy_helper.from_array(np.array(0, dtype=np.uint8), "z"),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "s", "z"], ["q"], name="q0", domain=ms),
        helper.make_node("DequantizeLinear", ["q", "s", "z"], ["dq"], name="dq0", domain=ms),
        helper.make_node("Relu", ["dq"], ["r"], name="relu"),
        helper.make_node("QuantizeLinear", ["r", "s", "z"], ["q1"], name="q1", domain=ms),
        helper.make_node("DequantizeLinear", ["q1", "s", "z"], ["dq1"], name="dq1", domain=ms),
    ]
    inputs = [_vi("x")]
    if with_unknown:
        nodes.append(helper.make_node("Mystery", ["aux"], ["u"], name="mystery", domain="custom.x"))
        nodes.append(helper.make_node("Add", ["dq1", "u"], ["y"], name="add"))
        inputs.append(_vi("aux", None))
    else:
        nodes.append(helper.make_node("Identity", ["dq1"], ["y"], name="id"))
    model = make_model(nodes, inputs, [_vi("y", None)], inits)
    model.opset_import.extend([helper.make_opsetid(ms, 1), helper.make_opsetid("custom.x", 1)])
    return model


# 10
def test_com_microsoft_qdq_uses_symbolic_fallback(tmp_path, make_model, save_model):
    pytest.importorskip("onnxruntime.tools.symbolic_shape_infer")
    path = save_model(_ms_qdq_model(make_model, with_unknown=False), tmp_path / "ms.onnx")
    with pytest.raises(SubgraphExtractionError):
        OnnxSubgraphExtractor(path, symbolic_shape_fallback=False).extract("dq", "dq1", tmp_path / "no")
    result = OnnxSubgraphExtractor(path).extract("dq", "dq1", tmp_path / "out")
    assert result.input_names == ["dq"]
    assert _node_names(result.model_path) == ["relu", "q1", "dq1"]


# 10b
def test_partial_symbolic_harvest_writes_nothing_to_cwd(tmp_path, make_model, save_model, monkeypatch):
    pytest.importorskip("onnxruntime.tools.symbolic_shape_infer")
    path = save_model(_ms_qdq_model(make_model, with_unknown=True), tmp_path / "model" / "ms.onnx")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    extractor = OnnxSubgraphExtractor(path)
    result = extractor.extract("dq", "dq1", tmp_path / "out")
    assert result.input_names == ["dq"]
    assert list(cwd.iterdir()) == []
    assert not list(tmp_path.rglob("sym_shape_infer_temp.onnx*"))
    assert Path.cwd() == cwd


def test_symbolic_fallback_missing_module(tmp_path, make_model, save_model, monkeypatch, mock_logger):
    path = save_model(_ms_qdq_model(make_model, with_unknown=False), tmp_path / "ms.onnx")
    monkeypatch.setitem(sys.modules, "onnxruntime.tools.symbolic_shape_infer", None)
    extractor = OnnxSubgraphExtractor(path, logger=mock_logger)
    mock_logger.debug.assert_called()
    with pytest.raises(SubgraphExtractionError):
        extractor.extract("dq", "dq1", tmp_path / "out")


def test_symbolic_fallback_hard_failure_warns(tmp_path, make_model, save_model, monkeypatch, mock_logger):
    module = pytest.importorskip("onnxruntime.tools.symbolic_shape_infer")
    path = save_model(_ms_qdq_model(make_model, with_unknown=False), tmp_path / "ms.onnx")

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(module.SymbolicShapeInference, "infer_shapes", staticmethod(boom))
    OnnxSubgraphExtractor(path, logger=mock_logger)
    mock_logger.warning.assert_called_once()


def test_symbolic_fallback_not_called_when_complete(qdq_chain_model, monkeypatch):
    module = pytest.importorskip("onnxruntime.tools.symbolic_shape_infer")

    def boom(*args, **kwargs):
        raise AssertionError("must not run")

    monkeypatch.setattr(module.SymbolicShapeInference, "infer_shapes", staticmethod(boom))
    OnnxSubgraphExtractor(qdq_chain_model)


# 10c
def test_external_data_reshape_every_layer_extracts(tmp_path, make_model, monkeypatch):
    shape = numpy_helper.from_array(np.array([2, 8], dtype=np.int64), "shape")
    weight = numpy_helper.from_array(np.ones((8, 3), dtype=np.float32), "w")
    model = make_model(
        [
            helper.make_node("Relu", ["x"], ["r"], name="relu"),
            helper.make_node("Reshape", ["r", "shape"], ["rs"], name="reshape"),
            helper.make_node("MatMul", ["rs", "w"], ["y"], name="matmul"),
        ],
        [_vi("x", (4, 4))],
        [_vi("y", None)],
        [shape, weight],
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    path = model_dir / "ext.onnx"
    onnx.save(
        model, str(path), save_as_external_data=True, all_tensors_to_one_file=True, location="d.data", size_threshold=0
    )
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    extractor = OnnxSubgraphExtractor(path, symbolic_shape_fallback=False)
    for start, end in [("x", "r"), ("r", "rs"), ("rs", "y")]:
        result = extractor.extract(start, end, tmp_path / "out" / end)
        assert result.input_names == [start]
        onnx.checker.check_model(str(result.model_path))


# 11
def test_deterministic_across_hash_seeds(residual_model, tmp_path):
    script = textwrap.dedent(
        f"""
        import json
        from subgraph_extractor import OnnxSubgraphExtractor
        r = OnnxSubgraphExtractor({str(residual_model)!r}).extract("b", "f", {str(tmp_path / "det")!r})
        print(json.dumps([r.input_names, r.output_names]))
        """
    )
    results = []
    for seed in ("1", "2"):
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(DEBUGGER_DIR), os.environ.get("PYTHONPATH", "")]),
            "PYTHONHASHSEED": seed,
        }
        out = subprocess.run(
            [sys.executable, "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True, check=True
        )
        results.append(json.loads(out.stdout.strip().splitlines()[-1]))
    assert results[0] == results[1]
    assert results[0] == [["b", "x1", "a", "x2"], ["f", "e"]]


# 12
@pytest.mark.parametrize("weights_first", [False, True])
@pytest.mark.parametrize("requantize_inputs", [False, True])
def test_numeric_equivalence(qdq_chain_model, qdq_input_sample, tmp_path, requantize_inputs, weights_first):
    path = _weights_first(qdq_chain_model, tmp_path) if weights_first else qdq_chain_model
    extractor = OnnxSubgraphExtractor(path)
    for start, end in [("input_dq", "conv1_out_dq"), ("conv1_out_dq", "output"), ("input", "relu1_out")]:
        result = extractor.extract(start, end, tmp_path / end, requantize_inputs=requantize_inputs)
        _assert_matches_model(path, result, qdq_input_sample)


# 13
def test_file_name_and_check_model(qdq_chain_model, tmp_path):
    out_dir = tmp_path / "nested" / "dir"
    result = OnnxSubgraphExtractor(qdq_chain_model).extract("conv1_out_dq", "output", out_dir)
    assert OnnxSubgraphExtractor.FILE_NAME == "extracted_model.onnx"
    assert result.model_path == out_dir / "extracted_model.onnx"
    onnx.checker.check_model(str(result.model_path))
    assert result.input_names == _input_names(result.model_path)


def test_large_subgraph_saved_with_external_data(qdq_chain_model, tmp_path, monkeypatch):
    extractor = OnnxSubgraphExtractor(qdq_chain_model)
    calls = []
    real_save = onnx.save

    def spy_save(proto, path, *args, **kwargs):
        calls.append(kwargs)
        return real_save(proto, path, *args, **kwargs, size_threshold=0)  # the test tensors are below 1 KiB

    monkeypatch.setattr(onnx.checker, "MAXIMUM_PROTOBUF", 1)
    monkeypatch.setattr(subgraph_extractor.onnx, "save", spy_save)
    result = extractor.extract("input_dq", "conv1_out_dq", tmp_path / "out")
    assert calls == [
        {"save_as_external_data": True, "all_tensors_to_one_file": True, "location": "extracted_model.onnx.data"}
    ]
    assert (tmp_path / "out" / "extracted_model.onnx.data").exists()
    assert result.input_names == ["input_dq"]


def test_save_failure_wrapped(qdq_chain_model, tmp_path, monkeypatch):
    extractor = OnnxSubgraphExtractor(qdq_chain_model)

    def bad_check(*args, **kwargs):
        raise onnx.checker.ValidationError("broken")

    monkeypatch.setattr(subgraph_extractor.onnx.checker, "check_model", bad_check)
    with pytest.raises(SubgraphExtractionError, match=r"^Saving extracted model failed: broken$"):
        extractor.extract("input_dq", "conv1_out_dq", tmp_path / "out")


# 14
def test_ir_version_below_4(tmp_path, make_model, save_model):
    init = numpy_helper.from_array(np.ones((1, 4), dtype=np.float32), "bias")
    nodes = [
        helper.make_node("Relu", ["x"], ["a"], name="relu"),
        helper.make_node("Add", ["a", "bias"], ["b"], name="add"),
        helper.make_node("Neg", ["b"], ["y"], name="neg"),
    ]
    model = make_model(nodes, [_vi("x"), _vi("bias")], [_vi("y")], [init], opset=8, ir_version=3)
    path = save_model(model, tmp_path / "ir3.onnx")
    result = OnnxSubgraphExtractor(path).extract("a", "y", tmp_path / "out")
    assert result.input_names == ["a"]
    assert set(_input_names(result.model_path)) == {"a", "bias"}
    onnx.checker.check_model(str(result.model_path))


# 15
def test_requantize_inputs_quantizes_dequantized_input_again(qdq_chain_model, tmp_path):
    result = OnnxSubgraphExtractor(qdq_chain_model).extract(
        "conv1_out_dq", "output", tmp_path / "out", requantize_inputs=True
    )
    assert result.input_names == ["conv1_out_dq"]
    assert result.output_names == ["output"]
    graph = onnx.load(str(result.model_path)).graph
    assert [node.name for node in graph.node] == [*_requantized("conv1_out_dq"), "relu1", "relu1_q", "relu1_dq"]
    quantize, dequantize, relu = graph.node[:3]
    assert list(quantize.input) == ["conv1_out_dq", "s_conv", "zp_conv"]
    assert list(dequantize.input) == ["conv1_out_dq_requant_q", "s_conv", "zp_conv"]
    assert list(relu.input) == ["conv1_out_dq_requant_dq"]
    assert {"s_conv", "zp_conv"} <= {t.name for t in graph.initializer}


# 16
def test_requantize_inputs_computes_weights_before_window(qdq_chain_model, tmp_path):
    extractor = OnnxSubgraphExtractor(_weights_first(qdq_chain_model, tmp_path))
    legacy = extractor.extract("input_dq", "conv1_out_dq", tmp_path / "legacy")
    assert legacy.input_names == ["input_dq", "w_dq"]
    assert _node_names(legacy.model_path) == ["conv1", "conv1_q", "conv1_dq"]
    result = extractor.extract("input_dq", "conv1_out_dq", tmp_path / "out", requantize_inputs=True)
    assert result.input_names == ["input_dq"]
    assert _node_names(result.model_path) == ["w_dq", *_requantized("input_dq"), "conv1", "conv1_q", "conv1_dq"]


# 17
def test_requantize_inputs_computes_constants_and_feeds_random_tensors(tmp_path, make_model, save_model):
    nodes = [
        helper.make_node("Cast", ["c"], ["cc"], name="cast", to=FLOAT),
        helper.make_node("RandomUniformLike", ["c"], ["noise"], name="noise"),
        helper.make_node("Relu", ["x"], ["a"], name="relu"),
        helper.make_node("Add", ["a", "cc"], ["b"], name="add_b"),
        helper.make_node("Add", ["b", "noise"], ["y"], name="add_y"),
    ]
    init = numpy_helper.from_array(np.ones((1, 4), dtype=np.float32), "c")
    path = save_model(make_model(nodes, [_vi("x")], [_vi("y")], [init]), tmp_path / "const.onnx")
    extractor = OnnxSubgraphExtractor(path)
    assert extractor.extract("a", "y", tmp_path / "legacy").input_names == ["a", "cc", "noise"]
    result = extractor.extract("a", "y", tmp_path / "out", requantize_inputs=True)
    assert result.input_names == ["a", "noise"]
    assert _node_names(result.model_path) == ["cast", "add_b", "add_y"]


# 18
def test_requantize_inputs_constant_node_scale_and_int8_without_zero_point(tmp_path, make_model, save_model):
    scale = numpy_helper.from_array(np.array(0.1, dtype=np.float32))
    nodes = [
        helper.make_node("Constant", [], ["sc"], name="sc", value=scale),
        helper.make_node("QuantizeLinear", ["x", "sc", "zu"], ["qa"], name="qa"),
        helper.make_node("QuantizeLinear", ["x", "s8", "z8"], ["qb"], name="qb"),
        helper.make_node("DequantizeLinear", ["qb", "s8"], ["db"], name="db"),
        helper.make_node("DequantizeLinear", ["qa", "sc", "zu"], ["da"], name="da"),
        helper.make_node("Add", ["da", "db"], ["y"], name="add"),
    ]
    inits = [
        numpy_helper.from_array(np.array(0, dtype=np.uint8), "zu"),
        numpy_helper.from_array(np.array(0.05, dtype=np.float32), "s8"),
        numpy_helper.from_array(np.array(0, dtype=np.int8), "z8"),
    ]
    path = save_model(make_model(nodes, [_vi("x")], [_vi("y")], inits), tmp_path / "params.onnx")
    result = OnnxSubgraphExtractor(path).extract("da", "y", tmp_path / "out", requantize_inputs=True)
    assert result.input_names == ["da", "db"]
    graph = onnx.load(str(result.model_path)).graph
    # Each pair takes its DequantizeLinear's place; the scale's Constant node is copied in front.
    assert [node.name for node in graph.node] == ["sc", *_requantized("db"), *_requantized("da"), "add"]
    assert list(graph.node[1].input) == ["db", "s8", "db_requant_zero_point"]
    zero_point = next(t for t in graph.initializer if t.name == "db_requant_zero_point")
    assert zero_point.data_type == TensorProto.INT8
    assert numpy_helper.to_array(zero_point).tolist() == 0
    x = (np.random.default_rng(0).random((1, 4), dtype=np.float32) - 0.5) * 20
    _assert_matches_model(path, result, {"x": x})


# 19
def test_requantize_inputs_keeps_float_inputs_it_cannot_quantize(tmp_path, make_model, save_model, mock_logger):
    nodes = [
        helper.make_node("DequantizeLinear", ["acc", "s"], ["acc_dq"], name="acc_dq"),
        helper.make_node("ReduceMax", ["x"], ["s_rt"], name="scale", keepdims=0),
        helper.make_node("QuantizeLinear", ["x", "s_rt", "z"], ["q"], name="q"),
        helper.make_node("DequantizeLinear", ["q", "s_rt", "z"], ["d"], name="dq"),
        helper.make_node("Neg", ["d"], ["n"], name="neg"),
        helper.make_node("Add", ["n", "acc_dq"], ["y"], name="add"),
    ]
    inits = [
        numpy_helper.from_array(np.array(0.1, dtype=np.float32), "s"),
        numpy_helper.from_array(np.array(0, dtype=np.uint8), "z"),
    ]
    model = make_model(nodes, [_vi("x"), _vi("acc", elem=TensorProto.INT32)], [_vi("y")], inits)
    path = save_model(model, tmp_path / "float.onnx")
    extractor = OnnxSubgraphExtractor(path, logger=mock_logger)
    result = extractor.extract("d", "y", tmp_path / "out", requantize_inputs=True)
    assert result.input_names == ["d", "acc_dq"]
    assert _node_names(result.model_path) == ["neg", "add"]
    mock_logger.debug.assert_any_call("Subgraph input 'd' stays float: the parameters of 'dq' are computed at runtime.")
    mock_logger.debug.assert_any_call("Subgraph input 'acc_dq' stays float: QuantizeLinear cannot produce int32.")


# 20
def test_requantize_inputs_copies_axis_and_avoids_name_clashes(tmp_path, make_model, save_model):
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "s", "z"], ["q"], name="q", axis=1),
        helper.make_node("DequantizeLinear", ["q", "s", "z"], ["d"], name="dq", axis=1),
        helper.make_node("Relu", ["d"], ["d_requant_q"], name="d_requant_QuantizeLinear"),
    ]
    inits = [
        numpy_helper.from_array(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32), "s"),
        numpy_helper.from_array(np.array([0, 1, 2, 3], dtype=np.uint8), "z"),
    ]
    path = save_model(make_model(nodes, [_vi("x")], [_vi("d_requant_q")], inits), tmp_path / "axis.onnx")
    result = OnnxSubgraphExtractor(path).extract("d", "d_requant_q", tmp_path / "out", requantize_inputs=True)
    quantize, dequantize, relu = onnx.load(str(result.model_path)).graph.node
    assert [quantize.name, dequantize.name] == ["d_requant_QuantizeLinear_1", "d_requant_DequantizeLinear"]
    assert list(quantize.output) == ["d_requant_q_1"]
    assert list(relu.input) == ["d_requant_dq"]
    assert [(a.name, a.i) for a in quantize.attribute] == [("axis", 1)]
    assert [(a.name, a.i) for a in dequantize.attribute] == [("axis", 1)]
    _assert_matches_model(path, result, {"x": np.array([[-1.0, 0.5, 2.0, 30.0]], dtype=np.float32)})


# 21
def test_requantize_inputs_keeps_com_microsoft_domain(tmp_path, make_model, save_model):
    pytest.importorskip("onnxruntime.tools.symbolic_shape_infer")
    path = save_model(_ms_qdq_model(make_model, with_unknown=False), tmp_path / "ms.onnx")
    result = OnnxSubgraphExtractor(path).extract("dq", "dq1", tmp_path / "out", requantize_inputs=True)
    graph = onnx.load(str(result.model_path)).graph
    assert [node.name for node in graph.node] == [*_requantized("dq"), "relu", "q1", "dq1"]
    assert [node.domain for node in graph.node[:2]] == ["com.microsoft", "com.microsoft"]
    assert result.input_names == ["dq"]


# 22
def test_requantize_inputs_prunes_unused_start_first(tmp_path, make_model, save_model):
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "s", "z"], ["q"], name="q"),
        helper.make_node("DequantizeLinear", ["q", "s", "z"], ["d"], name="dq"),
        helper.make_node("Relu", ["x"], ["r"], name="relu"),
        helper.make_node("Add", ["r", "d"], ["y"], name="add"),
    ]
    inits = [
        numpy_helper.from_array(np.array(0.1, dtype=np.float32), "s"),
        numpy_helper.from_array(np.array(0, dtype=np.uint8), "z"),
    ]
    path = save_model(make_model(nodes, [_vi("x")], [_vi("y")], inits), tmp_path / "unused.onnx")
    result = OnnxSubgraphExtractor(path).extract("d", "r", tmp_path / "out", requantize_inputs=True)
    assert result.input_names == ["x"]
    assert _node_names(result.model_path) == ["relu"]
