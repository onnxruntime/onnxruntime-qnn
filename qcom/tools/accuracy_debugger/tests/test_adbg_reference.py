# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Unit tests for reference."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("onnxruntime")

import numpy as np
import onnx
import reference
from onnx import TensorProto, helper, numpy_helper
from reference import (
    dump_reference_outputs,
    generate_framework_outputs,
    get_all_node_outputs,
    load_data_from_directory,
    run_onnx_reference,
)

_MOD = "reference"


def _spec(name: str, shape: tuple[int, ...] | None, dtype: str) -> SimpleNamespace:
    """A stand-in for ``onnx_utils.TensorSpec`` (same attribute contract)."""
    return SimpleNamespace(name=name, shape=shape, dtype=np.dtype(dtype))


def _expected_chain(sample: dict[str, np.ndarray], model_path: Path) -> tuple[np.ndarray, np.ndarray]:
    model = onnx.load(str(model_path))
    w = numpy_helper.to_array(next(i for i in model.graph.initializer if i.name == "w_fp")).astype(np.float64)
    x = np.pad(sample["input"].astype(np.float64), ((0, 0), (0, 0), (1, 1), (1, 1)))
    conv = np.zeros((1, 4, 8, 8))
    for i in range(8):
        for j in range(8):
            conv[0, :, i, j] = np.tensordot(w, x[0, :, i : i + 3, j : j + 3], axes=([1, 2, 3], [0, 1, 2]))
    return conv, np.maximum(conv, 0)


# ---------------------------------------------------------------------------------------------- run_onnx_reference


def test_get_all_node_outputs_skips_empty(make_model) -> None:
    model = make_model(
        [
            helper.make_node("Split", ["x"], ["a", "", "b"], name="split", axis=0, num_outputs=3),
            helper.make_node("Relu", ["a"], ["r"], name="relu"),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])],
        [helper.make_tensor_value_info("r", TensorProto.FLOAT, [1])],
    )
    assert get_all_node_outputs(model) == ["a", "b", "r"]


def test_run_onnx_reference_all_node_outputs(fp32_chain_model: Path, qdq_input_sample, logger) -> None:
    out = run_onnx_reference(fp32_chain_model, qdq_input_sample, None, logger=logger)
    assert list(out) == ["output", "conv1_out"]
    conv, relu = _expected_chain(qdq_input_sample, fp32_chain_model)
    np.testing.assert_allclose(out["conv1_out"], conv, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(out["output"], relu, rtol=1e-4, atol=1e-5)
    assert out["output"].dtype == np.float32


def test_run_onnx_reference_empty_means_graph_outputs_only(fp32_chain_model: Path, qdq_input_sample) -> None:
    out = run_onnx_reference(fp32_chain_model, qdq_input_sample, [])
    assert list(out) == ["output"]


def test_run_onnx_reference_selected_intermediates(qdq_chain_model: Path, qdq_input_sample, mock_logger) -> None:
    out = run_onnx_reference(
        qdq_chain_model, qdq_input_sample, ["conv1_out_dq", "", "no_such_tensor", "relu1_out_q"], logger=mock_logger
    )
    assert list(out) == ["output", "conv1_out_dq", "relu1_out_q"]
    assert out["relu1_out_q"].dtype == np.uint8
    warnings = [str(c.args[0]) for c in mock_logger.warning.call_args_list]
    assert any("no_such_tensor" in w for w in warnings)


def test_run_onnx_reference_does_not_mutate_model_file(fp32_chain_model: Path, qdq_input_sample) -> None:
    before = fp32_chain_model.read_bytes()
    run_onnx_reference(fp32_chain_model, qdq_input_sample)
    assert fp32_chain_model.read_bytes() == before


def test_run_onnx_reference_missing_input(fp32_chain_model: Path) -> None:
    with pytest.raises(ValueError, match=r"^Missing input 'input' for reference model\.$"):
        run_onnx_reference(fp32_chain_model, {"other": np.zeros(1, np.float32)})


def test_run_onnx_reference_error_propagates_as_runtime_error(fp32_chain_model: Path) -> None:
    bad = {"input": np.zeros((1, 3, 8, 8), dtype=np.float64)}
    with pytest.raises(RuntimeError, match=r"^ONNX inference execution failed with error: "):
        run_onnx_reference(fp32_chain_model, bad)


def test_run_onnx_reference_external_data(tmp_path: Path, make_model, qdq_input_sample) -> None:
    w = np.full((1, 3, 8, 8), 2.0, dtype=np.float32)
    model = make_model(
        [helper.make_node("Mul", ["input", "w"], ["output"], name="mul")],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 8, 8])],
        initializers=[numpy_helper.from_array(w, "w")],
    )
    path = tmp_path / "ext" / "model.onnx"
    path.parent.mkdir()
    onnx.save(model, str(path), save_as_external_data=True, location="model.data", size_threshold=0)
    out = run_onnx_reference(path, qdq_input_sample, [])
    np.testing.assert_allclose(out["output"], qdq_input_sample["input"] * 2)


# ---------------------------------------------------------------------------------------------- generate_framework_outputs


def test_generate_framework_outputs_dumps(tmp_path: Path, fp32_chain_model: Path, qdq_input_sample, logger) -> None:
    out = generate_framework_outputs(
        logger, fp32_chain_model, qdq_input_sample, dump_output_tensors=True, output_dir=tmp_path / "wd"
    )
    assert set(out) == {"output", "conv1_out"}
    assert (tmp_path / "wd" / "reference_outputs" / "output.raw").is_file()
    assert (tmp_path / "wd" / "reference_outputs" / "conv1_out.raw").is_file()
    assert (tmp_path / "wd" / "reference_outputs" / "profile_info.json").is_file()


def test_generate_framework_outputs_forwards(tmp_path: Path, logger) -> None:
    sentinel = {"a": np.zeros(1, np.float32)}
    with (
        patch(f"{_MOD}.run_onnx_reference", return_value=sentinel) as run,
        patch(f"{_MOD}.dump_reference_outputs") as dump,
    ):
        assert generate_framework_outputs(logger, "m.onnx", {"x": 1}, intermediate_output_tensors=["t"]) is sentinel
        run.assert_called_once_with("m.onnx", {"x": 1}, ["t"], logger=logger)
        dump.assert_not_called()
        generate_framework_outputs(logger, "m.onnx", {}, dump_output_tensors=True, output_dir=tmp_path)
        dump.assert_called_once_with(sentinel, tmp_path / "reference_outputs", logger=logger)


def test_generate_framework_outputs_requires_output_dir(logger) -> None:
    with pytest.raises(ValueError, match="output_dir must be specified"):
        generate_framework_outputs(logger, "m.onnx", {}, dump_output_tensors=True)


# ---------------------------------------------------------------------------------------------- dump_reference_outputs


def test_dump_reference_outputs_files_and_profile(tmp_path: Path, mock_logger) -> None:
    outputs = {
        "/conv1/Conv_output_0": np.array([[1.23456, -2.0], [3.0, 0.5]], dtype=np.float32),
        "ints": np.array([1, 2, 3], dtype=np.int64),
        "flags": np.array([True, False]),
        "empty": np.zeros((0, 3), dtype=np.float32),
    }
    dump_reference_outputs(outputs, tmp_path / "out", logger=mock_logger)
    out_dir = tmp_path / "out"
    conv = np.fromfile(out_dir / "_conv1_Conv_output_0.raw", dtype=np.float32)
    np.testing.assert_array_equal(conv, outputs["/conv1/Conv_output_0"].ravel())
    ints = np.fromfile(out_dir / "ints.raw", dtype=np.float32)
    np.testing.assert_array_equal(ints, np.array([1, 2, 3], dtype=np.float32))
    np.testing.assert_array_equal(np.fromfile(out_dir / "flags.raw", dtype=np.float32), [1.0, 0.0])
    assert (out_dir / "empty.raw").stat().st_size == 0

    profile = json.loads((out_dir / "profile_info.json").read_text(encoding="utf-8"))
    assert list(profile) == ["_conv1_Conv_output_0", "ints", "flags", "empty"]
    assert profile["_conv1_Conv_output_0"] == [
        "type = float32",
        "shape = (2, 2)",
        "min = -2.0",
        "max = 3.0",
        "median = 0.867",
    ]
    assert profile["ints"] == ["type = int64", "shape = (3,)", "min = 1", "max = 3", "median = 2.0"]
    assert profile["flags"] == ["bool", [2], [True, False], [True, False], [True, False]]
    assert profile["empty"] == ["-", "-", "-", "-", "-"]
    mock_logger.warning.assert_not_called()


def test_dump_reference_outputs_profile_is_indented_utf8(tmp_path: Path) -> None:
    dump_reference_outputs({"a": np.ones(2, np.float32)}, tmp_path)
    text = (tmp_path / "profile_info.json").read_text(encoding="utf-8")
    assert text == json.dumps(
        {"a": ["type = float32", "shape = (2,)", "min = 1.0", "max = 1.0", "median = 1.0"]},
        ensure_ascii=False,
        indent=4,
    )


def test_dump_reference_outputs_skips_long_names(tmp_path: Path, mock_logger) -> None:
    if sys.platform == "win32":
        tmp_path = Path("\\\\?\\" + str(tmp_path.resolve()))
    long_name = "x" * 252  # 252 + ".raw" = 256 bytes
    ok_name = "y" * 251  # 255 bytes
    dump_reference_outputs(
        {long_name: np.ones(1, np.float32), ok_name: np.ones(1, np.float32)}, tmp_path, logger=mock_logger
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([f"{ok_name}.raw", "profile_info.json"])
    profile = json.loads((tmp_path / "profile_info.json").read_text(encoding="utf-8"))
    assert list(profile) == [ok_name]
    mock_logger.warning.assert_called_once()


def test_dump_reference_outputs_warns_on_sanitised_collision(tmp_path: Path, mock_logger) -> None:
    dump_reference_outputs(
        {"a/b": np.ones(1, np.float32), "a.b": np.full(1, 2.0, np.float32)}, tmp_path, logger=mock_logger
    )
    np.testing.assert_array_equal(np.fromfile(tmp_path / "a_b.raw", np.float32), [2.0])
    mock_logger.warning.assert_called_once()


# ---------------------------------------------------------------------------------------------- load_data_from_directory


def test_load_data_round_trip_with_specs(tmp_path: Path) -> None:
    np.arange(6, dtype=np.float32).tofile(tmp_path / "act.raw")
    np.array([1, 2, 3, 4], dtype=np.float32).tofile(tmp_path / "ints.raw")
    np.array([7, 8], dtype=np.float32).tofile(tmp_path / "plain.raw")
    info = {"act": _spec("act", (2, 3), "float32"), "ints": _spec("ints", (2, 2), "uint8")}
    data = load_data_from_directory(tmp_path, info)
    assert data["act"].shape == (2, 3)
    assert data["ints"].dtype == np.uint8
    np.testing.assert_array_equal(data["ints"], [[1, 2], [3, 4]])
    assert data["plain"].dtype == np.float32
    assert data["plain"].shape == (2,)


def test_load_data_sanitised_lookup_unsanitised_key(tmp_path: Path) -> None:
    # The file on disk carries a stem that sanitises to the info key; the result key is the raw stem.
    np.arange(4, dtype=np.float32).tofile(tmp_path / "conv-1.out.raw")
    data = load_data_from_directory(tmp_path, {"conv_1_out": _spec("conv-1.out", (2, 2), "float16")})
    assert list(data) == ["conv-1.out"]
    assert data["conv-1.out"].dtype == np.float16
    assert data["conv-1.out"].shape == (2, 2)


def test_load_data_ignores_target_and_subgraph_dirs(tmp_path: Path) -> None:
    np.array([1.0], np.float32).tofile(tmp_path / "t.raw")
    target = tmp_path / "target_outputs"
    target.mkdir()
    np.array([9.0], np.float32).tofile(target / "t.raw")
    np.array([9.0], np.float32).tofile(target / "only_target.raw")
    sub = tmp_path / "subgraph_data" / "layer" / "target_outputs"
    sub.mkdir(parents=True)
    np.array([9.0], np.float32).tofile(sub / "sub.raw")
    nested = tmp_path / "reference_outputs"
    nested.mkdir()
    np.array([3.0], np.float32).tofile(nested / "n.raw")
    data = load_data_from_directory(tmp_path)
    assert set(data) == {"t", "n"}
    np.testing.assert_array_equal(data["t"], [1.0])


def test_load_data_flat_on_mismatch_and_dynamic(tmp_path: Path, mock_logger) -> None:
    np.arange(5, dtype=np.float32).tofile(tmp_path / "a.raw")
    np.arange(4, dtype=np.float32).tofile(tmp_path / "b.raw")
    info = {"a": _spec("a", (2, 3), "float32"), "b": _spec("b", None, "int32")}
    data = load_data_from_directory(tmp_path, info, logger=mock_logger)
    assert data["a"].shape == (5,)
    assert data["b"].shape == (4,)
    assert data["b"].dtype == np.int32
    mock_logger.warning.assert_called_once()
    assert "'a'" in str(mock_logger.warning.call_args.args[0])


@pytest.mark.parametrize("extra_batches", [["batch_2"], ["batch_2", "batch_10"]])
def test_load_data_multi_sample_uses_batch_1(tmp_path: Path, mock_logger, extra_batches: list[str]) -> None:
    for i, name in enumerate(["batch_1", *extra_batches]):
        (tmp_path / name).mkdir()
        np.full(2, float(i + 1), np.float32).tofile(tmp_path / name / "out.raw")
    data = load_data_from_directory(tmp_path, logger=mock_logger)
    assert list(data) == ["out"]
    np.testing.assert_array_equal(data["out"], [1.0, 1.0])
    mock_logger.warning.assert_called_once_with(
        "Golden reference contains batch_<n> directories; only batch_1 is used."
    )


def test_load_data_batches_without_batch_1(tmp_path: Path) -> None:
    (tmp_path / "batch_2").mkdir()
    np.ones(1, np.float32).tofile(tmp_path / "batch_2" / "out.raw")
    with pytest.raises(ValueError, match=r"^Golden reference has batch_<n> dirs but no batch_1\.$"):
        load_data_from_directory(tmp_path)


def test_load_data_duplicate_stems_first_wins(tmp_path: Path, mock_logger) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    np.array([1.0], np.float32).tofile(tmp_path / "a" / "dup.raw")
    np.array([2.0], np.float32).tofile(tmp_path / "b" / "dup.raw")
    data = load_data_from_directory(tmp_path, logger=mock_logger)
    np.testing.assert_array_equal(data["dup"], [1.0])
    mock_logger.warning.assert_called_once_with(
        f"Ignoring duplicate golden tensor 'dup' at {tmp_path / 'b' / 'dup.raw'}"
    )


def test_load_data_dump_round_trip(tmp_path: Path, fp32_chain_model: Path, qdq_input_sample) -> None:
    out = run_onnx_reference(fp32_chain_model, qdq_input_sample)
    dump_reference_outputs(out, tmp_path)
    info = {name: _spec(name, arr.shape, str(arr.dtype)) for name, arr in out.items()}
    data = load_data_from_directory(tmp_path, info)
    assert set(data) == set(out)
    for name, arr in out.items():
        np.testing.assert_array_equal(data[name], arr)


def test_golden_shape_kept_after_reshape_in_external_data_model(
    tmp_path: Path, make_model, qdq_input_sample, logger: logging.Logger
) -> None:
    """End to end through the real ``get_intermediate_outputs_info``: Reshape's target shape is external data."""
    from onnx_utils import get_intermediate_outputs_info  # noqa: PLC0415

    model = make_model(
        [
            helper.make_node("Relu", ["input"], ["relu_out"], name="relu"),
            helper.make_node("Reshape", ["relu_out", "shape"], ["reshaped"], name="reshape"),
            helper.make_node("Identity", ["reshaped"], ["output"], name="ident"),
        ],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)],
        initializers=[numpy_helper.from_array(np.array([3, 64], dtype=np.int64), "shape")],
    )
    path = tmp_path / "ext" / "model.onnx"
    path.parent.mkdir()
    onnx.save(model, str(path), save_as_external_data=True, location="model.data", size_threshold=0)

    golden = tmp_path / "golden"
    generate_framework_outputs(logger, path, qdq_input_sample, dump_output_tensors=True, output_dir=golden)
    data = reference.load_data_from_directory(golden, get_intermediate_outputs_info(path), logger=logger)
    assert data["reshaped"].shape == (3, 64)
    assert data["relu_out"].shape == (1, 3, 8, 8)
    np.testing.assert_allclose(data["reshaped"], np.maximum(qdq_input_sample["input"], 0).reshape(3, 64))
