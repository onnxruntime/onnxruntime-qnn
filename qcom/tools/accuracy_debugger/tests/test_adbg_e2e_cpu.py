# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""End-to-end snooper runs on ORT CPU (no QNN needed)."""

from __future__ import annotations

import pytest

pytest.importorskip("onnxruntime")

import importlib.util
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import reference
from common import Algorithm
from comparators import get_comparator
from inputs import InputProcessor
from ort_qnn_snooper import (
    LOGGER_NAME,
    OrtQnnSnooper,
    OrtQnnSnooperInputConfig,
    execute_ort_qnn_snooper,
)

BASE_COLUMNS = [
    "Layer Name",
    "Layer Output",
    "Layer Type",
    "Layer Shape",
    "Source(Min, Max, Median)",
    "Target(Min, Max, Median)",
]
COMPARATORS = ["mse", "cosine"]
HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


@pytest.fixture(autouse=True)
def _ort_cpu_target(monkeypatch: pytest.MonkeyPatch, ort_cpu_target) -> None:
    # Test-only: run the reference unoptimised too, so ORT cannot fuse DQ-Conv-Q into QLinearConv on the reference
    # side only. Production reference numerics (default optimisations, design 1.9 / D6) are untouched.
    session_options = reference.ort.SessionOptions

    def _unoptimised_session_options() -> Any:
        options = session_options()
        options.graph_optimization_level = reference.ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        return options

    monkeypatch.setattr(reference.ort, "SessionOptions", _unoptimised_session_options)


def _expected_columns(algorithm: Algorithm) -> list[str]:
    if algorithm == Algorithm.ONESHOT:
        return [*BASE_COLUMNS, *COMPARATORS]
    return [*BASE_COLUMNS[:2], "Layer Status", *BASE_COLUMNS[2:], *COMPARATORS, "Exception"]


def _run(
    model: Path,
    sample: dict,
    write_input_list,
    tmp_path: Path,
    algorithm: Algorithm,
    isolate: bool = False,
    **config: Any,
) -> tuple[Path, pd.DataFrame, dict]:
    wd = tmp_path / "wd"
    wd.mkdir()
    input_list = write_input_list(sample, tmp_path / "inputs")
    snooper_config = OrtQnnSnooperInputConfig(
        qdq_model=model,
        input_processor=InputProcessor(input_list=input_list, model=model, working_dir=wd),
        algorithm=algorithm,
        comparators=[get_comparator(name) for name in COMPARATORS],
        working_directory=wd,
        dump_output_tensors=True,
        **config,
    )
    output = OrtQnnSnooper(logging.getLogger("adbg_e2e_cpu"), name=algorithm, isolate_inference=isolate).run(
        snooper_config
    )
    assert Path(output.csv_snooping_report) == wd / f"{algorithm.value}.csv"
    assert Path(output.json_snooping_report) == wd / f"{algorithm.value}.json"
    data_frame = pd.read_csv(output.csv_snooping_report, keep_default_na=False)
    report = json.loads(Path(output.json_snooping_report).read_text(encoding="utf-8"))
    return wd, data_frame, report


def _assert_common(wd: Path, data_frame: pd.DataFrame, report: dict, algorithm: Algorithm) -> None:
    assert data_frame.columns.tolist() == _expected_columns(algorithm)
    assert len(data_frame) > 0
    # Target and the (test-only unoptimised) reference run the same unfused graph on ORT CPU: identical numerics.
    assert all(float(v) <= 1e-12 for v in data_frame["mse"])
    assert report["header"]["algorithm"] == algorithm.value
    assert report["header"]["artifact_type"] == f"{algorithm.name}_SNOOPING_REPORT"
    assert len(report["layers"]) == len(data_frame)
    if algorithm == Algorithm.LAYERWISE:
        assert set(data_frame["Layer Status"]) == {"SUCCESS"}
        assert all(layer["layer_status"] == "SUCCESS" for layer in report["layers"])
    if HAS_MATPLOTLIB:
        assert (wd / "plots" / "mse.png").exists()
        assert (wd / "plots" / "cosine.png").exists()
    assert (wd / "reference_outputs" / "profile_info.json").exists()


CASES = [
    pytest.param(algorithm, isolate, intermediates, elementwise, id=f"{algorithm.value}-iso{int(isolate)}-{iid}-{eid}")
    for algorithm in (Algorithm.ONESHOT, Algorithm.LAYERWISE)
    for isolate in ((False,) if algorithm == Algorithm.ONESHOT else (False, True))
    for iid, intermediates in (("all", None), ("conv", ["Conv"]))
    for eid, elementwise in (("noew", None), ("ew", {"conv1": 0}))
]


@pytest.mark.parametrize(("algorithm", "isolate", "intermediates", "elementwise"), CASES)
def test_qdq_chain_model(
    qdq_chain_model: Path,
    qdq_input_sample,
    write_input_list,
    tmp_path: Path,
    algorithm: Algorithm,
    isolate: bool,
    intermediates: list[str] | None,
    elementwise: dict[str, int] | None,
) -> None:
    wd, data_frame, report = _run(
        qdq_chain_model,
        qdq_input_sample,
        write_input_list,
        tmp_path,
        algorithm,
        isolate,
        set_intermediate_layers=intermediates,
        dump_elementwise_stats=elementwise,
    )
    _assert_common(wd, data_frame, report, algorithm)
    outputs = data_frame["Layer Output"].tolist()
    if algorithm == Algorithm.ONESHOT:
        if intermediates:
            # Requested intermediates first, then the model's own output (kept by the engine, legacy behaviour).
            assert outputs == ["conv1_out_q", "conv1_out_dq", "output"]
        else:
            # The weight DQ "w_dq" is not an activation and is never compared.
            assert outputs == ["input_q", "input_dq", "conv1_out_q", "conv1_out_dq", "relu1_out_q", "output"]
        assert (wd / "target_outputs" / "conv1_out_dq.raw").exists()
    else:
        expected = ["conv1_out_q", "conv1_out_dq"] + ([] if intermediates else ["relu1_out_q", "output"])
        assert outputs == expected
        assert (wd / "subgraph_data" / "conv1" / "extracted_model.onnx").exists()
        assert (wd / "subgraph_data" / "conv1" / "target_outputs" / "conv1_out_dq.raw").exists()
        assert data_frame["Layer Shape"].iloc[0] == "(1, 4, 8, 8)"
        # Target equals reference, so no element is ever further than 0 steps and no elementwise log is written.
        assert not (wd / "elementwise_stats.log").exists()


@pytest.mark.parametrize("algorithm", [Algorithm.ONESHOT, Algorithm.LAYERWISE])
def test_quantized_model(quantized_model: Path, qdq_input_sample, write_input_list, tmp_path: Path, algorithm) -> None:
    wd, data_frame, report = _run(quantized_model, qdq_input_sample, write_input_list, tmp_path, algorithm)
    _assert_common(wd, data_frame, report, algorithm)
    if algorithm == Algorithm.LAYERWISE:
        assert data_frame["Layer Name"].unique().tolist() == ["conv1", "add1"]
        assert data_frame["Layer Output"].tolist()[-1] == "output"


def test_oneshot_with_fp32_reference(
    qdq_chain_model: Path, fp32_chain_model: Path, qdq_input_sample, write_input_list, tmp_path: Path
) -> None:
    _, data_frame, _ = _run(
        qdq_chain_model,
        qdq_input_sample,
        write_input_list,
        tmp_path,
        Algorithm.ONESHOT,
        reference_model=fp32_chain_model,
    )
    # DQ outputs only (FP32 reference); conv1_out_dq maps to conv1_out by prefix, output by exact name.
    assert data_frame["Layer Output"].tolist() == ["conv1_out_dq", "output"]
    assert data_frame["Layer Name"].tolist() == ["conv1", "relu1"]
    assert all(float(v) < 1e-2 for v in data_frame["mse"])


def test_cli_layerwise(qdq_chain_model: Path, qdq_input_sample, write_input_list, tmp_path: Path) -> None:
    input_list = write_input_list(qdq_input_sample, tmp_path / "inputs")
    wd = tmp_path / "wd"
    wd.mkdir()
    try:
        execute_ort_qnn_snooper(
            [
                "--qdq_model",
                str(qdq_chain_model),
                "--input_list",
                str(input_list),
                "--algorithm",
                "layerwise",
                "--comparator",
                "mse",
                "--working_directory",
                str(wd),
                "--output_directory",
                "run",
                "--set_intermediate_layers",
                "conv1",
            ]
        )
    finally:
        snooper_logger = logging.getLogger(LOGGER_NAME)
        for handler in list(snooper_logger.handlers):
            snooper_logger.removeHandler(handler)
            handler.close()
    run_dir = wd / "layerwise_snooping" / "run"
    data_frame = pd.read_csv(run_dir / "layerwise.csv", keep_default_na=False)
    assert data_frame["Layer Status"].tolist() == ["SUCCESS", "SUCCESS"]
    assert (run_dir / "layerwise.json").exists()
    log_text = (run_dir / "accuracy_debugger_user_info.log").read_text()
    assert "Snooping Completed. CSV Report generated at:" in log_text


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"algorithm": Algorithm.LAYERWISE, "reference_model": "SELF"},
            "reference_model is supported only with oneshot algorithm.",
        ),
        ({"set_cpu_layers": ["conv1"]}, "set_cpu_layers option is supported only with layerwise algorithm."),
        (
            {"reference_model": "FP32", "dump_elementwise_stats": {"conv1": 0}},
            "dump_elementwise_stats option is supported only when reference_model is a quantized model.",
        ),
    ],
)
def test_negative_validator_messages(
    qdq_chain_model: Path,
    fp32_chain_model: Path,
    qdq_input_sample,
    write_input_list,
    tmp_path: Path,
    overrides: dict[str, Any],
    message: str,
) -> None:
    overrides = dict(overrides)
    if overrides.get("reference_model") == "SELF":
        overrides["reference_model"] = qdq_chain_model
    elif overrides.get("reference_model") == "FP32":
        overrides["reference_model"] = fp32_chain_model
    input_list = write_input_list(qdq_input_sample, tmp_path / "inputs")
    with pytest.raises(ValueError, match=message):
        OrtQnnSnooperInputConfig(
            qdq_model=qdq_chain_model,
            input_processor=InputProcessor(input_list=input_list, model=qdq_chain_model),
            working_directory=tmp_path,
            **overrides,
        )
