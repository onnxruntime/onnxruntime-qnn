# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Unit tests for onnx_utils."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnx
import onnx_utils
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx_utils import (
    TensorSpec,
    add_graph_outputs,
    constant_tensors,
    get_input_specs,
    get_intermediate_outputs_info,
    infer_model_with_data,
    is_qdq_model,
    load_model_structure,
    session_model_source,
    validate_model,
)


def _relu_model(make_model, elem=TensorProto.FLOAT, shape=(1, 4)):
    return make_model(
        [helper.make_node("Relu", ["x"], ["y"], name="relu")],
        [helper.make_tensor_value_info("x", elem, list(shape))],
        [helper.make_tensor_value_info("y", elem, list(shape))],
    )


def _reshape_external_model(make_model, save_model, path: Path) -> Path:
    """Relu -> Reshape(shape stored externally) -> MatMul; the post-Reshape shape needs the constant data."""
    shape = numpy_helper.from_array(np.array([2, 8], dtype=np.int64), "shape")
    weight = numpy_helper.from_array(np.ones((8, 3), dtype=np.float32), "w")
    model = make_model(
        [
            helper.make_node("Relu", ["x"], ["r"], name="relu"),
            helper.make_node("Reshape", ["r", "shape"], ["rs"], name="reshape"),
            helper.make_node("MatMul", ["rs", "w"], ["y"], name="matmul"),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        [shape, weight],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx.data",
        size_threshold=0,
    )
    return path


# --------------------------------------------------------------------------- is_qdq_model / validate_model


def test_is_qdq_model_path_and_proto(qdq_chain_model, fp32_chain_model):
    assert is_qdq_model(qdq_chain_model)
    assert is_qdq_model(str(qdq_chain_model))
    assert is_qdq_model(onnx.load(str(qdq_chain_model)))
    assert not is_qdq_model(fp32_chain_model)


def test_is_qdq_model_matches_prefix(make_model):
    model = make_model(
        [helper.make_node("DequantizeLinearCustom", ["x"], ["y"], name="dq", domain="custom")],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    assert is_qdq_model(model)


# --------------------------------------------------------------------------- constant_tensors


def test_constant_tensors_follows_initializer_only_chains():
    branch = helper.make_graph(
        [helper.make_node("Identity", ["c"], ["branch_out"])],
        "branch",
        [],
        [helper.make_tensor_value_info("branch_out", TensorProto.FLOAT, [])],
    )
    nodes = [
        helper.make_node("Constant", [], ["c"], value=helper.make_tensor("value", TensorProto.FLOAT, [], [1.0])),
        helper.make_node("Identity", ["zp"], ["zp_copy"]),
        helper.make_node("Cast", ["c"], ["c_64"], to=TensorProto.DOUBLE),
        helper.make_node("DequantizeLinear", ["w_q", "s", "zp_copy"], ["w"]),
        # Not constant: a runtime input, a random op and a node with subgraphs.
        helper.make_node("Add", ["x", "c"], ["runtime"]),
        helper.make_node("Identity", ["runtime"], ["runtime_copy"]),
        helper.make_node("RandomNormalLike", ["c"], ["noise"]),
        helper.make_node("If", ["flag"], ["chosen"], then_branch=branch, else_branch=branch),
    ]
    initializers = ["w_q", "s", "zp", "flag"]
    assert constant_tensors(nodes, initializers) == {*initializers, "c", "zp_copy", "c_64", "w"}


@pytest.mark.parametrize("as_str", [False, True])
def test_validate_model_messages(tmp_path, qdq_chain_model, fp32_chain_model, as_str):
    conv = str if as_str else Path
    missing = tmp_path / "missing.onnx"
    with pytest.raises(FileNotFoundError) as exc:
        validate_model(conv(missing))
    assert str(exc.value) == f"Model path '{missing}' does not exist."

    wrong = tmp_path / "model.txt"
    wrong.write_text("x")
    with pytest.raises(ValueError, match=r"^Invalid model format specified\. Please provide a valid ONNX file\.$"):
        validate_model(conv(wrong))

    with pytest.raises(ValueError, match=r"^Model doesn't contain QDQ nodes\. Please provide a valid QDQ ONNX file\.$"):
        validate_model(conv(fp32_chain_model), validate_qdq=True)

    validate_model(conv(fp32_chain_model))
    validate_model(conv(qdq_chain_model), validate_qdq=True)


# --------------------------------------------------------------------------- get_input_specs


def test_get_input_specs_excludes_initializers_and_maps_dtype(tmp_path, make_model, save_model):
    init = numpy_helper.from_array(np.ones((2,), dtype=np.float32), "c")
    model = make_model(
        [
            helper.make_node("Add", ["a", "c"], ["s"], name="add"),
            helper.make_node("Cast", ["b"], ["bc"], name="cast", to=TensorProto.FLOAT),
            helper.make_node("Add", ["s", "bc"], ["y"], name="add2"),
        ],
        [
            helper.make_tensor_value_info("a", TensorProto.FLOAT16, [1, 2]),
            helper.make_tensor_value_info("c", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("b", TensorProto.INT64, ["batch", 2]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        [init],
    )
    path = save_model(model, tmp_path / "m.onnx")
    specs = get_input_specs(path, {"batch": 3})
    assert specs == [
        TensorSpec("a", (1, 2), np.dtype(np.float16)),
        TensorSpec("b", (3, 2), np.dtype(np.int64)),
    ]
    assert all(isinstance(s.dtype, np.dtype) for s in specs)

    with pytest.raises(ValueError, match=r"^Please supply value for onnx symbol: batch$"):
        get_input_specs(path)
    with pytest.raises(ValueError, match="batch"):
        get_input_specs(str(path), {"other": 1})


def test_get_input_specs_static_zero_dim_kept(tmp_path, make_model, save_model):
    path = save_model(_relu_model(make_model, shape=(0, 4)), tmp_path / "m.onnx")
    assert get_input_specs(path) == [TensorSpec("x", (0, 4), np.dtype(np.float32))]


# --------------------------------------------------------------------------- get_intermediate_outputs_info


def test_get_intermediate_outputs_info_fp16_and_dynamic(tmp_path, make_model, save_model):
    model = make_model(
        [
            helper.make_node("Relu", ["x"], ["/r/out"], name="relu"),
            helper.make_node("Neg", ["/r/out"], ["y"], name="neg"),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, ["n", 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, ["n", 4])],
    )
    path = save_model(model, tmp_path / "m.onnx")
    info = get_intermediate_outputs_info(path)
    assert set(info) == {"_r_out", "y"}
    assert info["_r_out"] == TensorSpec("/r/out", None, np.dtype(np.float16))
    assert info["y"].dtype == np.dtype(np.float16)
    assert info["y"].shape is None


def test_get_intermediate_outputs_info_static_and_unknown_dtype(tmp_path, make_model, save_model):
    model = make_model(
        [helper.make_node("Relu", ["x"], ["r"], name="relu"), helper.make_node("Neg", ["r"], ["y"], name="neg")],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.UNDEFINED, None)],
    )
    path = save_model(model, tmp_path / "m.onnx")
    info = get_intermediate_outputs_info(path)
    assert info["r"] == TensorSpec("r", (2, 3), np.dtype(np.float32))
    assert info["y"].dtype == np.dtype(np.float32)


def test_get_intermediate_outputs_info_external_reshape(tmp_path, make_model, save_model, monkeypatch):
    path = _reshape_external_model(make_model, save_model, tmp_path / "ext" / "model.onnx")
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    info = get_intermediate_outputs_info(path)
    assert info["rs"] == TensorSpec("rs", (2, 8), np.dtype(np.float32))
    assert info["y"].shape == (2, 3)


# --------------------------------------------------------------------------- infer_model_with_data


def test_infer_model_with_data_in_memory(tmp_path, make_model, save_model):
    path = save_model(_relu_model(make_model), tmp_path / "m.onnx")
    model = infer_model_with_data(path)
    assert isinstance(model, onnx.ModelProto)


def test_infer_model_with_data_external_loaded(tmp_path, make_model, save_model):
    path = _reshape_external_model(make_model, save_model, tmp_path / "ext" / "model.onnx")
    model = infer_model_with_data(str(path))
    assert "rs" in {vi.name for vi in model.graph.value_info}
    assert all(t.data_location != TensorProto.EXTERNAL for t in model.graph.initializer)


def _spy_infer_shapes_path(monkeypatch):
    calls = []
    real = onnx.shape_inference.infer_shapes_path

    def spy(src, dst, *args, **kwargs):
        calls.append((src, dst))
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr(onnx.shape_inference, "infer_shapes_path", spy)
    return calls


def test_infer_model_with_data_large_inlines_shape_constants(tmp_path, make_model, save_model, monkeypatch):
    path = _reshape_external_model(make_model, save_model, tmp_path / "ext" / "model.onnx")
    structure = load_model_structure(path)
    # Above the limit with all data, below it with only the int64 shape constant inlined.
    monkeypatch.setattr(onnx.checker, "MAXIMUM_PROTOBUF", structure.ByteSize() + 40)
    calls = _spy_infer_shapes_path(monkeypatch)
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    model = infer_model_with_data(path)
    assert calls == []
    vi = {v.name: v for v in model.graph.value_info}
    assert [d.dim_value for d in vi["rs"].type.tensor_type.shape.dim] == [2, 8]
    inits = {t.name: t for t in model.graph.initializer}
    assert inits["shape"].data_location == TensorProto.DEFAULT
    assert inits["w"].data_location == TensorProto.EXTERNAL
    onnx.load_external_data_for_model(model, str(path.parent))
    np.testing.assert_array_equal(numpy_helper.to_array(inits["w"]), np.ones((8, 3), dtype=np.float32))


def test_infer_model_with_data_huge_uses_infer_shapes_path(tmp_path, make_model, save_model, monkeypatch):
    path = _reshape_external_model(make_model, save_model, tmp_path / "ext" / "model.onnx")
    monkeypatch.setattr(onnx.checker, "MAXIMUM_PROTOBUF", 1)
    calls = _spy_infer_shapes_path(monkeypatch)
    model = infer_model_with_data(path)
    assert len(calls) == 1
    assert calls[0][0] == str(path)
    assert "r" in {v.name for v in model.graph.value_info}
    assert any(t.data_location == TensorProto.EXTERNAL for t in model.graph.initializer)
    onnx.load_external_data_for_model(model, str(path.parent))


def test_infer_model_with_data_failure_warns(tmp_path, make_model, save_model, monkeypatch, mock_logger):
    path = save_model(_relu_model(make_model), tmp_path / "m.onnx")

    def boom(model, *args, **kwargs):
        raise RuntimeError("bad")

    monkeypatch.setattr(onnx.shape_inference, "infer_shapes", boom)
    model = infer_model_with_data(path, mock_logger)
    assert [n.name for n in model.graph.node] == ["relu"]
    mock_logger.warning.assert_called_once_with("Shape inference failed: bad")


def test_infer_model_with_data_default_logger(tmp_path, make_model, save_model, monkeypatch, caplog):
    path = save_model(_relu_model(make_model), tmp_path / "m.onnx")
    monkeypatch.setattr(onnx.shape_inference, "infer_shapes", lambda *a, **k: (_ for _ in ()).throw(ValueError("x")))
    with caplog.at_level(logging.WARNING, logger=onnx_utils.__name__):
        infer_model_with_data(path)
    assert "Shape inference failed: x" in caplog.text


def test_load_model_structure_does_not_read_data(tmp_path, make_model, save_model):
    path = _reshape_external_model(make_model, save_model, tmp_path / "ext" / "model.onnx")
    model = load_model_structure(path)
    assert all(t.data_location == TensorProto.EXTERNAL for t in model.graph.initializer)


# --------------------------------------------------------------------------- add_graph_outputs


def test_add_graph_outputs_order_dedup_and_empty(make_model):
    model = make_model(
        [
            helper.make_node("Relu", ["x"], ["a"], name="r1"),
            helper.make_node("Neg", ["a"], ["b"], name="n1"),
            helper.make_node("Abs", ["b"], ["y"], name="a1"),
            helper.make_node("Neg", ["b"], ["z"], name="n2"),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("z", TensorProto.FLOAT, [1]),
        ],
    )
    result = add_graph_outputs(model, ["b", "", "a", "b", "z"])
    assert result == ["b", "a", "z", "y"]
    assert [o.name for o in model.graph.output] == result
    outputs = {o.name: o for o in model.graph.output}
    assert outputs["z"].type.tensor_type.elem_type == TensorProto.FLOAT
    assert outputs["y"].type.tensor_type.elem_type == TensorProto.FLOAT
    assert not outputs["a"].HasField("type")


def test_add_graph_outputs_empty_request_keeps_outputs(make_model):
    model = _relu_model(make_model)
    assert add_graph_outputs(model, []) == ["y"]
    assert [o.name for o in model.graph.output] == ["y"]


# --------------------------------------------------------------------------- session_model_source


def test_session_model_source_bytes(make_model):
    model = _relu_model(make_model)
    with session_model_source(model) as source:
        assert isinstance(source, bytes)
        assert onnx.load_from_string(source) == model


def test_session_model_source_path_branch(make_model, monkeypatch):
    weight = numpy_helper.from_array(np.arange(1024, dtype=np.float32).reshape(32, 32), "w")
    model = make_model(
        [helper.make_node("MatMul", ["x", "w"], ["y"], name="mm")],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 32])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 32])],
        [weight],
    )
    before = model.SerializeToString()
    monkeypatch.setattr(onnx.checker, "MAXIMUM_PROTOBUF", 1)
    with session_model_source(model) as source:
        assert isinstance(source, str)
        path = Path(source)
        assert path.name == "model.onnx"
        assert (path.parent / "model.onnx.data").exists()
        loaded = onnx.load(source)
        np.testing.assert_array_equal(numpy_helper.to_array(loaded.graph.initializer[0]), numpy_helper.to_array(weight))
    assert not path.parent.exists()
    assert model.SerializeToString() == before
