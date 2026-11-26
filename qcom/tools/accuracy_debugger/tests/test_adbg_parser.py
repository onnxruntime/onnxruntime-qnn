# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Tests for parser (ported from the legacy ORT/QNN snooper parser tests)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from common import Algorithm
from comparators import Comparator
from parser import PROG, OrtQnnSnooperParser, build_parser


@pytest.fixture
def files(tmp_path: Path) -> dict[str, str]:
    """Real dummy files for the existence checks."""
    reference_model = tmp_path / "reference_model.onnx"
    reference_model.write_text("dummy reference model content")
    qdq_model = tmp_path / "qdq_model.onnx"
    qdq_model.write_text("dummy qdq model content")
    input_list = tmp_path / "input_list.txt"
    input_list.write_text("dummy input list content")
    golden = tmp_path / "golden_reference"
    golden.mkdir()
    return {
        "reference_model": str(reference_model),
        "qdq_model": str(qdq_model),
        "input_list": str(input_list),
        "golden_reference": str(golden),
    }


@pytest.fixture
def base_args(files: dict[str, str]) -> list[str]:
    return [
        "--reference_model",
        files["reference_model"],
        "--qdq_model",
        files["qdq_model"],
        "--input_list",
        files["input_list"],
    ]


def _parse(args: list[str]) -> argparse.Namespace:
    return OrtQnnSnooperParser().parse(args)


def test_parser_initialization() -> None:
    parser = OrtQnnSnooperParser()
    assert parser.parser.prog == PROG
    assert parser.required is not None
    assert parser.optional is not None
    assert isinstance(build_parser(), argparse.ArgumentParser)


def test_parse_required_arguments_only(base_args: list[str], files: dict[str, str]) -> None:
    args = _parse(base_args)
    assert args.reference_model == files["reference_model"]
    assert args.qdq_model == files["qdq_model"]
    assert args.input_list == files["input_list"]
    assert args.algorithm == Algorithm.ONESHOT
    assert args.working_directory is None
    assert args.output_directory is None
    assert args.golden_reference is None
    assert args.set_intermediate_layers is None
    assert args.set_cpu_layers is None
    assert args.dump_elementwise_stats is None
    assert args.log_level == "INFO"
    assert [c.name for c in args.comparator] == ["mse"]


def test_parse_all_arguments(base_args: list[str], files: dict[str, str]) -> None:
    args = _parse(
        [
            *base_args,
            "--algorithm",
            "layerwise",
            "--comparator",
            "mse",
            "cosine",
            "--working_directory",
            "/tmp/working",
            "--output_directory",
            "/tmp/output",
            "--golden_reference",
            files["golden_reference"],
            "--set_intermediate_layers",
            "conv1,conv2,Gemm",
            "--set_cpu_layers",
            "fc1,fc2,MatMul",
            "--log_level",
            "DEBUG",
        ]
    )
    assert args.algorithm == Algorithm.LAYERWISE
    assert [c.name for c in args.comparator] == ["mse", "cosine"]
    assert args.working_directory == "/tmp/working"
    assert args.output_directory == "/tmp/output"
    assert args.golden_reference == files["golden_reference"]
    assert args.set_intermediate_layers == ["conv1", "conv2", "Gemm"]
    assert args.set_cpu_layers == ["fc1", "fc2", "MatMul"]
    assert args.log_level == "DEBUG"


def test_missing_required_argument_qdq_model(files: dict[str, str]) -> None:
    with pytest.raises(SystemExit):
        _parse(["--reference_model", files["reference_model"], "--input_list", files["input_list"]])


def test_missing_required_argument_input_list(files: dict[str, str]) -> None:
    with pytest.raises(SystemExit):
        _parse(["--reference_model", files["reference_model"], "--qdq_model", files["qdq_model"]])


def test_qdq_model_file_not_exists(files: dict[str, str]) -> None:
    missing = "/path/to/non_existent_qdq_model.onnx"
    with pytest.raises(argparse.ArgumentTypeError) as excinfo:
        _parse(["--qdq_model", missing, "--input_list", files["input_list"]])
    assert str(excinfo.value) == f"QDQ model file '{missing}' does not exist."


def test_input_list_file_not_exists(files: dict[str, str]) -> None:
    missing = "/path/to/non_existent_input.txt"
    with pytest.raises(argparse.ArgumentTypeError) as excinfo:
        _parse(["--qdq_model", files["qdq_model"], "--input_list", missing])
    assert str(excinfo.value) == f"Input list file '{missing}' does not exist."


@pytest.mark.parametrize(("value", "expected"), [("oneshot", Algorithm.ONESHOT), ("layerwise", Algorithm.LAYERWISE)])
def test_algorithm_choices(base_args: list[str], value: str, expected: Algorithm) -> None:
    args = _parse([*base_args, "--algorithm", value])
    assert args.algorithm == expected
    assert isinstance(args.algorithm, Algorithm)


def test_invalid_algorithm(base_args: list[str]) -> None:
    with pytest.raises(SystemExit):
        _parse([*base_args, "--algorithm", "INVALID_ALGORITHM"])


def test_algorithm_help_lists_values(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "oneshot" in out
    assert "layerwise" in out
    assert "--input_sample" not in out


def test_help_strings_have_separating_spaces(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "debugged. Can specify" in out
    assert "Execution-Provider. This option" in out


def test_set_intermediate_layers_parsing(base_args: list[str]) -> None:
    args = _parse([*base_args, "--set_intermediate_layers", "conv1_output, Gemm , MatMul,fc_layer_output"])
    assert args.set_intermediate_layers == ["conv1_output", "Gemm", "MatMul", "fc_layer_output"]


def test_set_cpu_layers_parsing(base_args: list[str]) -> None:
    args = _parse([*base_args, "--set_cpu_layers", "conv1_output , Gemm, MatMul ,fc_layer_output"])
    assert args.set_cpu_layers == ["conv1_output", "Gemm", "MatMul", "fc_layer_output"]


def test_empty_comma_items_dropped(base_args: list[str]) -> None:
    args = _parse([*base_args, "--set_intermediate_layers", "conv1,, ,Gemm,", "--set_cpu_layers", ",Relu"])
    assert args.set_intermediate_layers == ["conv1", "Gemm"]
    assert args.set_cpu_layers == ["Relu"]


def test_whitespace_in_layer_names(base_args: list[str]) -> None:
    args = _parse([*base_args, "--set_intermediate_layers", "  conv1  ,  conv2  ,  Gemm  "])
    assert args.set_intermediate_layers == ["conv1", "conv2", "Gemm"]


def test_many_intermediate_layers(base_args: list[str]) -> None:
    args = _parse([*base_args, "--set_intermediate_layers", ",".join(f"layer_{i}" for i in range(100))])
    assert len(args.set_intermediate_layers) == 100
    assert args.set_intermediate_layers[0] == "layer_0"
    assert args.set_intermediate_layers[99] == "layer_99"


def test_multiple_comparators(base_args: list[str]) -> None:
    args = _parse([*base_args, "--comparator", "mse", "cosine", "mae"])
    assert len(args.comparator) == 3
    assert all(isinstance(c, Comparator) for c in args.comparator)
    assert [c.name for c in args.comparator] == ["mse", "cosine", "mae"]


def test_single_comparator(base_args: list[str]) -> None:
    args = _parse([*base_args, "--comparator", "mse"])
    assert len(args.comparator) == 1


def test_duplicate_comparators_deduplicated(base_args: list[str]) -> None:
    args = _parse([*base_args, "--comparator", "cosine", "mse", "cosine", "mse"])
    assert [c.name for c in args.comparator] == ["cosine", "mse"]


def test_invalid_comparator(base_args: list[str]) -> None:
    with pytest.raises(SystemExit):
        _parse([*base_args, "--comparator", "not_a_comparator"])


def test_working_and_output_directory(base_args: list[str]) -> None:
    args = _parse([*base_args, "--working_directory", "/custom/wd", "--output_directory", "/custom/out"])
    assert args.working_directory == "/custom/wd"
    assert args.output_directory == "/custom/out"


def test_parse_returns_namespace(base_args: list[str]) -> None:
    args = _parse(base_args)
    assert isinstance(args, argparse.Namespace)
    for attr in ("reference_model", "qdq_model", "input_list", "algorithm", "comparator"):
        assert hasattr(args, attr)


def test_log_level_default(base_args: list[str]) -> None:
    assert _parse(base_args).log_level == "INFO"


@pytest.mark.parametrize("level", ["ERROR", "WARN", "INFO", "DEBUG", "VERBOSE"])
def test_all_log_levels(base_args: list[str], level: str) -> None:
    assert _parse([*base_args, "--log_level", level.lower()]).log_level == level


def test_log_level_invalid(base_args: list[str]) -> None:
    with pytest.raises(SystemExit):
        _parse([*base_args, "--log_level", "INVALID_LEVEL"])


def test_removed_input_sample_alias(files: dict[str, str]) -> None:
    with pytest.raises(SystemExit):
        _parse(["--qdq_model", files["qdq_model"], "--input_sample", files["input_list"]])


def test_removed_input_sample_alias_with_input_list(base_args: list[str]) -> None:
    with pytest.raises(SystemExit):
        _parse([*base_args, "--input_sample", "/ignored/path"])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("a=1;b=2;", {"a": 1, "b": 2}),
        ("conv1=2;Conv=0", {"conv1": 2, "Conv": 0}),
        (" conv1 = 3 ", {"conv1": 3}),
        ("a=b=4", {"a=b": 4}),
    ],
)
def test_dump_elementwise_stats_parsing(base_args: list[str], value: str, expected: dict[str, int]) -> None:
    assert _parse([*base_args, "--dump_elementwise_stats", value]).dump_elementwise_stats == expected


@pytest.mark.parametrize("value", ["conv1", "conv1=x", "conv1=-1", "=3", "conv1=1.5"])
def test_dump_elementwise_stats_invalid(base_args: list[str], value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="--dump_elementwise_stats value is not in valid format"):
        _parse([*base_args, "--dump_elementwise_stats", value])
