# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Unit tests for inputs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from inputs import InputProcessor, parse_input_list
from onnx import TensorProto, helper

_SPECS_PATCH = "inputs.get_input_specs"


def _spec(name: str, shape: tuple[int, ...] | None, dtype: str) -> SimpleNamespace:
    """A stand-in for ``onnx_utils.TensorSpec`` (same attribute contract)."""
    return SimpleNamespace(name=name, shape=shape, dtype=np.dtype(dtype))


@pytest.fixture
def dummy_model(tmp_path: Path) -> Path:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"")
    return path


# ---------------------------------------------------------------------------------------------- parse_input_list


def test_parse_input_list_comments_blanks_named_positional(tmp_path: Path) -> None:
    lst = tmp_path / "input_list.txt"
    lst.write_text(
        "# a comment\n\n   \n  a:=/abs/a.raw\tb.raw  \nrel/x.raw rel/y.raw\n  # indented comment\n n := v.raw\n",
        encoding="utf-8",
    )
    assert parse_input_list(lst) == [
        {"a": "/abs/a.raw", "1": "b.raw"},
        {"0": "rel/x.raw", "1": "rel/y.raw"},
        {"0": "n", "": "", "2": "v.raw"},
    ]


def test_parse_input_list_named_tokens(tmp_path: Path) -> None:
    lst = tmp_path / "input_list.txt"
    lst.write_text("in_0:=/data/a.raw in_1:=b/c.raw\n", encoding="utf-8")
    assert parse_input_list(lst) == [{"in_0": "/data/a.raw", "in_1": "b/c.raw"}]


def test_parse_input_list_splits_on_first_separator_only(tmp_path: Path) -> None:
    lst = tmp_path / "input_list.txt"
    lst.write_text("x:=a:=b.raw\n", encoding="utf-8")
    assert parse_input_list(lst) == [{"x": "a:=b.raw"}]


def test_parse_input_list_mixed_positional_index(tmp_path: Path) -> None:
    lst = tmp_path / "input_list.txt"
    lst.write_text("first.raw named:=second.raw third.raw\n", encoding="utf-8")
    assert parse_input_list(lst) == [{"0": "first.raw", "named": "second.raw", "2": "third.raw"}]


def test_parse_input_list_multiple_lines(tmp_path: Path) -> None:
    lst = tmp_path / "input_list.txt"
    lst.write_text("a.raw\n# skip\nb.raw\n\n", encoding="utf-8")
    assert parse_input_list(lst) == [{"0": "a.raw"}, {"0": "b.raw"}]


# ---------------------------------------------------------------------------------------------- InputProcessor init


def test_init_coerces_paths(tmp_path: Path, dummy_model: Path) -> None:
    proc = InputProcessor(str(tmp_path / "list.txt"), str(dummy_model), working_dir=str(tmp_path))
    assert proc.input_list == tmp_path / "list.txt"
    assert proc.model == dummy_model
    assert proc.working_dir == tmp_path
    assert InputProcessor(tmp_path / "list.txt", dummy_model).working_dir is None


@pytest.mark.parametrize("name", ["model.dlc", "model.pb", "model.ONNX.bin"])
def test_init_rejects_non_onnx_suffix(tmp_path: Path, name: str) -> None:
    model = tmp_path / name
    model.write_bytes(b"")
    with pytest.raises(ValueError, match=r"^Invalid source model\. Supported framework types are \['onnx'\]$"):
        InputProcessor(tmp_path / "list.txt", model)


def test_init_rejects_missing_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid source model"):
        InputProcessor(tmp_path / "list.txt", tmp_path / "missing.onnx")


# ---------------------------------------------------------------------------------------------- get_framework_inputs


def test_get_framework_inputs_dtype_reshape_named(tmp_path: Path, dummy_model: Path) -> None:
    a = np.arange(6, dtype=np.float32)
    b = np.arange(4, dtype=np.int64)
    a.tofile(tmp_path / "a.raw")
    b.tofile(tmp_path / "b.raw")
    lst = tmp_path / "list.txt"
    # Named tokens in the opposite order of the model inputs; b uses a relative path.
    lst.write_text(f'tok_b:=b.raw tok_a:="{tmp_path / "a.raw"}"\n', encoding="utf-8")
    specs = [_spec("tok_a", (2, 3), "float32"), _spec("tok_b", (2, 2), "int64")]
    with patch(_SPECS_PATCH, return_value=specs) as mocked:
        proc = InputProcessor(lst, dummy_model, onnx_symbols={"N": 1})
        out = proc.get_framework_inputs()
    mocked.assert_called_once_with(dummy_model, {"N": 1})
    assert len(out) == 1
    assert list(out[0]) == ["tok_a", "tok_b"]
    assert out[0]["tok_a"].dtype == np.float32
    assert out[0]["tok_a"].shape == (2, 3)
    np.testing.assert_array_equal(out[0]["tok_a"].ravel(), a)
    assert out[0]["tok_b"].dtype == np.int64
    assert out[0]["tok_b"].shape == (2, 2)
    np.testing.assert_array_equal(out[0]["tok_b"].ravel(), b)


def test_get_framework_inputs_positional_relative(tmp_path: Path, dummy_model: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    np.array([1, 2, 3], dtype=np.uint8).tofile(data_dir / "x.raw")
    np.array([True, False], dtype=np.bool_).tofile(data_dir / "y.raw")
    lst = tmp_path / "list.txt"
    lst.write_text("data/x.raw data/y.raw\ndata/x.raw data/y.raw\n", encoding="utf-8")
    specs = [_spec("x", (1, 3), "uint8"), _spec("y", (2,), "bool")]
    with patch(_SPECS_PATCH, return_value=specs):
        out = InputProcessor(lst, dummy_model).get_framework_inputs()
    assert len(out) == 2
    np.testing.assert_array_equal(out[1]["x"], np.array([[1, 2, 3]], dtype=np.uint8))
    np.testing.assert_array_equal(out[0]["y"], np.array([True, False]))


def test_get_framework_inputs_unknown_shape_stays_flat(tmp_path: Path, dummy_model: Path) -> None:
    np.arange(5, dtype=np.float32).tofile(tmp_path / "x.raw")
    lst = tmp_path / "list.txt"
    lst.write_text("x.raw\n", encoding="utf-8")
    with patch(_SPECS_PATCH, return_value=[_spec("x", None, "float32")]):
        out = InputProcessor(lst, dummy_model).get_framework_inputs()
    assert out[0]["x"].shape == (5,)


def test_get_framework_inputs_count_mismatch(tmp_path: Path, dummy_model: Path) -> None:
    lst = tmp_path / "list.txt"
    lst.write_text("# header\na.raw\na.raw b.raw c.raw\n", encoding="utf-8")
    np.zeros(1, np.float32).tofile(tmp_path / "a.raw")
    specs = [_spec("a", (1,), "float32"), _spec("b", (1,), "float32")]
    with (
        patch(_SPECS_PATCH, return_value=specs),
        pytest.raises(ValueError, match=r"^Input list line 1: expected 2 input\(s\), got 1\.$"),
    ):
        InputProcessor(lst, dummy_model).get_framework_inputs()


def test_get_framework_inputs_missing_name(tmp_path: Path, dummy_model: Path) -> None:
    lst = tmp_path / "list.txt"
    lst.write_text("a:=a.raw other:=b.raw\n", encoding="utf-8")
    np.zeros(1, np.float32).tofile(tmp_path / "a.raw")
    specs = [_spec("a", (1,), "float32"), _spec("b", (1,), "float32")]
    with (
        patch(_SPECS_PATCH, return_value=specs),
        pytest.raises(ValueError, match=r"^Input list line 1: no file for input 'b'\.$"),
    ):
        InputProcessor(lst, dummy_model).get_framework_inputs()


def test_get_framework_inputs_size_mismatch_message(tmp_path: Path, dummy_model: Path) -> None:
    np.zeros(5, np.float32).tofile(tmp_path / "a.raw")
    lst = tmp_path / "list.txt"
    lst.write_text("a:=a.raw\n", encoding="utf-8")
    with patch(_SPECS_PATCH, return_value=[_spec("a", (2, 3), "float32")]), pytest.raises(ValueError) as excinfo:
        InputProcessor(lst, dummy_model).get_framework_inputs()
    assert str(excinfo.value) == (
        f"Input file '{tmp_path / 'a.raw'}' has 5 elements of float32, expected 6 for input 'a' with shape (2, 3)."
    )


def test_get_framework_inputs_is_memoised(tmp_path: Path, dummy_model: Path) -> None:
    np.zeros(2, np.float32).tofile(tmp_path / "a.raw")
    lst = tmp_path / "list.txt"
    lst.write_text("a.raw\n", encoding="utf-8")
    with patch(_SPECS_PATCH, return_value=[_spec("a", (2,), "float32")]) as mocked:
        proc = InputProcessor(lst, dummy_model)
        first = proc.get_framework_inputs()
        second = proc.get_framework_inputs()
    assert first is second
    mocked.assert_called_once()


def test_get_framework_inputs_real_model(tmp_path: Path, make_model, save_model, write_input_list) -> None:
    """End to end with the real ``get_input_specs``: initializers are not inputs; dims come from the model."""
    model = make_model(
        [helper.make_node("Add", ["x", "w"], ["y"], name="add")],
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT16, [2, 3]),
            helper.make_tensor_value_info("w", TensorProto.FLOAT16, [2, 3]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [2, 3])],
        initializers=[helper.make_tensor("w", TensorProto.FLOAT16, [2, 3], np.zeros(6, np.float16).tolist())],
        ir_version=8,
    )
    path = save_model(model, tmp_path / "m" / "model.onnx")
    sample = {"x": np.arange(6, dtype=np.float16).reshape(2, 3)}
    lst = write_input_list(sample, tmp_path / "inputs")
    out = InputProcessor(lst, path).get_framework_inputs()
    assert list(out[0]) == ["x"]
    assert out[0]["x"].dtype == np.float16
    np.testing.assert_array_equal(out[0]["x"], sample["x"])
