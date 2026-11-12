# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""End-to-end snooper runs on the real QNN Execution Provider (ResNet-18 QDQ), ported from the legacy QNN-EP E2E test.

Requirements (otherwise every case is skipped):
    * ``onnxruntime-qnn`` installed and a QNN EP device available (``requires_qnn_ep()``);
    * ``ADBG_E2E_ASSET_DIR`` pointing to a directory with ``r18.qdq.onnx``, ``dog.raw`` and the reference reports
      ``oneshot_full.csv``, ``oneshot_partial.csv``, ``layerwise_full.csv``, ``layerwise_partial.csv``.
"""

from __future__ import annotations

import itertools
import logging
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("onnxruntime")

import onnx
import pandas as pd
from common import Algorithm
from comparators import get_comparator
from conftest import requires_qnn_ep
from inputs import InputProcessor
from ort_qnn_snooper import HOST_DEQUANTIZE_REASON, OrtQnnSnooper, OrtQnnSnooperInputConfig

pytestmark = pytest.mark.qnn

ASSET_ENV = "ADBG_E2E_ASSET_DIR"
ASSET_FILES = (
    "r18.qdq.onnx",
    "dog.raw",
    "oneshot_full.csv",
    "oneshot_partial.csv",
    "layerwise_full.csv",
    "layerwise_partial.csv",
)
R18_INPUT_NAME = "t.1"
R18_INTERMEDIATE_LAYERS = ["MaxPool", "layer4.1.conv2", "layer2.1.conv1"]
R18_CPU_LAYERS = ["MaxPool", "layer4.1.conv2", "layer2.1.conv1"]
R18_ELEMENTWISE_STATS = {"layer4.1.conv2": 0, "layer2.1.conv1": 1}
R18_REFERENCE_MODEL: Path | None = None  # the QDQ model is its own reference
COMPARATOR_NAMES = ("mse", "cosine")
COMPARED_COLUMNS = ("Layer Name", "Layer Output", "Layer Type", "Layer Shape")
CPU_RETRY_MESSAGE = "ORT retried this model on CPU"
# The legacy reports name and select Q/DQ outputs by node position: the node before a Q, and two nodes before a DQ,
# is the layer an output belongs to, and the Q and DQ right after a layer are its outputs. The snooper follows data
# edges (design.md F30), so the reports differ where branches interleave in node order:
# * The input Q/DQ follow a weight DQ, which oneshot names them after. They are named after themselves.
# * Stages 2-4 start with conv1 and the downsample Conv, then their Q/DQ nodes. Oneshot names conv1's Q after the
#   downsample Conv and the other three Q/DQ outputs after Q nodes. Layerwise has a single row for both layers,
#   conv1's Q under the downsample name.
R18_INPUT_QDQ = {
    "t.1_QuantizeLinear_Output": ("t.1_QuantizeLinear", "QuantizeLinear"),
    "t.1_DequantizeLinear_Output": ("t.1_DequantizeLinear", "DequantizeLinear"),
}
R18_STAGE_STARTS = {2: ("100", "108"), 3: ("145", "153"), 4: ("190", "198")}  # stage -> conv1, downsample tensors

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def asset_dir() -> Path:
    """Skips unless the QNN EP is usable and the ResNet-18 assets are available."""
    if not requires_qnn_ep():
        pytest.skip("QNN Execution Provider (onnxruntime-qnn) is not available.")
    value = os.environ.get(ASSET_ENV)
    if not value:
        pytest.skip(f"{ASSET_ENV} is not set.")
    directory = Path(value)
    missing = [name for name in ASSET_FILES if not (directory / name).is_file()]
    if missing:
        pytest.skip(f"{ASSET_ENV}={directory} is missing {missing}.")
    return directory


def _snooper_config(
    asset_dir: Path,
    work_dir: Path,
    algorithm: Algorithm,
    set_intermediates: bool,
    set_cpu_layers: bool,
    dump_elementwise_stats: bool,
    **trace_options: Any,
) -> OrtQnnSnooperInputConfig:
    qdq_model = asset_dir / "r18.qdq.onnx"
    input_list = work_dir / "input_list.txt"
    input_list.write_text(f'{R18_INPUT_NAME}:="{asset_dir / "dog.raw"}"\n', encoding="utf8")
    return OrtQnnSnooperInputConfig(
        reference_model=R18_REFERENCE_MODEL,
        qdq_model=qdq_model,
        input_processor=InputProcessor(input_list=input_list, model=qdq_model, working_dir=work_dir),
        algorithm=algorithm,
        working_directory=work_dir,
        golden_reference_path=None,
        set_intermediate_layers=R18_INTERMEDIATE_LAYERS if set_intermediates else None,
        set_cpu_layers=R18_CPU_LAYERS if set_cpu_layers else None,
        dump_elementwise_stats=R18_ELEMENTWISE_STATS if dump_elementwise_stats else None,
        comparators=[get_comparator(name) for name in COMPARATOR_NAMES],
        **trace_options,
    )


def _stage_start_layers(stage: int) -> dict[str, str]:
    """Maps the Q/DQ outputs of the conv1 and downsample Conv that start a ResNet-18 stage to their layer names."""
    conv1, downsample = R18_STAGE_STARTS[stage]
    return {
        f"{tensor}_{op}_Output": layer
        for tensor, layer in ((conv1, f"layer{stage}.0.conv1"), (downsample, f"layer{stage}.0.downsample.0"))
        for op in ("QuantizeLinear", "DequantizeLinear")
    }


def _read_reference(reference_csv: Path) -> pd.DataFrame:
    """Reads a legacy reference report without its weight and bias rows, which the snooper does not capture.

    ResNet-18 dequantizes each weight and bias initializer with one DequantizeLinear node. The rows that legacy
    names or selects by node position are corrected (see ``R18_STAGE_STARTS``).

    Args:
        reference_csv: Report in the asset directory, next to ``r18.qdq.onnx``.

    Returns:
        The remaining rows, re-indexed from 0.
    """
    graph = onnx.load(reference_csv.parent / "r18.qdq.onnx", load_external_data=False).graph
    initializers = {initializer.name for initializer in graph.initializer}
    weights = {
        node.output[0] for node in graph.node if node.op_type == "DequantizeLinear" and node.input[0] in initializers
    }
    reference = pd.read_csv(reference_csv)
    reference = reference[~reference["Layer Output"].isin(weights)]

    stage_starts = {next(iter(layers)): layers for layers in map(_stage_start_layers, R18_STAGE_STARTS)}
    layer_names = {output: layer for layers in stage_starts.values() for output, layer in layers.items()}
    layerwise = reference_csv.name.startswith(Algorithm.LAYERWISE.value)
    rows = []
    for row in reference.to_dict("records"):
        output = row["Layer Output"]
        if output in R18_INPUT_QDQ:
            name, layer_type = R18_INPUT_QDQ[output]
            rows.append({**row, "Layer Name": name, "Layer Type": layer_type})
        elif layerwise and output in stage_starts:
            # The four rows share the layer type and shape; the scores of the legacy row are not compared.
            rows.extend({**row, "Layer Name": name, "Layer Output": out} for out, name in stage_starts[output].items())
        elif output in layer_names:
            rows.append({**row, "Layer Name": layer_names[output], "Layer Type": "Conv"})
        else:
            rows.append(row)
    return pd.DataFrame(rows, columns=reference.columns)


def assert_cpu_retry_reported(output_csv: Path, reference_csv: Path, log_text: str) -> None:
    """Asserts that zero scores where the reference has error are explained by a reported ORT retry on CPU.

    ORT reruns a failing QNN session on CPU, so the target equals the reference. Such rows must carry the reason
    in ``Exception`` (layerwise) or ``CPU_Fallback_Reason`` (traced oneshot); otherwise the log must warn.
    """
    output_df = pd.read_csv(output_csv)
    merged = output_df.merge(pd.read_csv(reference_csv), on="Layer Output", suffixes=("", "_ref"))
    suspicious = merged[(merged["mse"] == 0) & (merged["mse_ref"] > 0)]
    if suspicious.empty:
        return
    column = next((name for name in ("Exception", "CPU_Fallback_Reason") if name in output_df.columns), None)
    if column is None:
        assert CPU_RETRY_MESSAGE in log_text, "Zero scores were not explained by a CPU retry warning"
        return
    unflagged = suspicious[~suspicious[column].fillna("").astype(str).str.contains(CPU_RETRY_MESSAGE)]
    assert unflagged.empty, f"Zero-score rows without a CPU retry flag: {unflagged['Layer Output'].tolist()}"


def assert_csv_report(output_csv: Path, reference_csv: Path, columns_to_compare: tuple[str, ...] = COMPARED_COLUMNS):
    """Asserts that the generated CSV has the reference columns, row count and per-row layer columns.

    Args:
        output_csv: Generated report.
        reference_csv: Known-good report. Its weight and bias Q/DQ rows are ignored.
        columns_to_compare: Columns compared value by value (NaN == NaN counts as a match).
    """
    reference_df = _read_reference(reference_csv)
    output_df = pd.read_csv(output_csv)

    expected_columns = reference_df.columns.tolist()
    found_columns = output_df.columns.tolist()
    assert expected_columns == found_columns, (
        f"Column names do not match. Expected: {expected_columns}, Found: {found_columns}"
    )
    assert len(output_df) == len(reference_df), (
        f"Number of rows do not match. Expected: {len(reference_df)}, Found: {len(output_df)}"
    )
    for column in columns_to_compare:
        comparison = (reference_df[column] == output_df[column]) | (
            reference_df[column].isna() & output_df[column].isna()
        )
        mismatches = ~comparison
        assert mismatches.sum() == 0, (
            f"{mismatches.sum()} mismatches found in column '{column}' at row indices: "
            f"{reference_df[mismatches].index.tolist()}"
        )


CASES = list(
    itertools.product(
        [Algorithm.ONESHOT, Algorithm.LAYERWISE],  # algorithm
        [True, False],  # set_intermediates
        [True, False],  # set_cpu_layers
        [True, False],  # dump_elementwise_stats
    )
)


@pytest.mark.parametrize(("algorithm", "set_intermediates", "set_cpu_layers", "dump_elementwise_stats"), CASES)
def test_qnn_ep_snooping(
    asset_dir: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    algorithm: Algorithm,
    set_intermediates: bool,
    set_cpu_layers: bool,
    dump_elementwise_stats: bool,
) -> None:
    params: dict[str, Any] = {
        "algorithm": algorithm,
        "set_intermediates": set_intermediates,
        "set_cpu_layers": set_cpu_layers,
        "dump_elementwise_stats": dump_elementwise_stats,
    }

    # Negative configurations.
    if set_cpu_layers and algorithm != Algorithm.LAYERWISE:
        with pytest.raises(ValueError, match="set_cpu_layers option is supported only with layerwise algorithm."):
            _snooper_config(asset_dir, tmp_path, **params)
        return
    if R18_REFERENCE_MODEL is not None:
        if algorithm != Algorithm.ONESHOT:
            with pytest.raises(ValueError, match="reference_model is supported only with oneshot algorithm."):
                _snooper_config(asset_dir, tmp_path, **params)
            return
        if dump_elementwise_stats:
            with pytest.raises(
                ValueError,
                match="dump_elementwise_stats option is supported only when reference_model is a quantized model.",
            ):
                _snooper_config(asset_dir, tmp_path, **params)
            return

    config = _snooper_config(asset_dir, tmp_path, **params)
    snooper = OrtQnnSnooper(logger, name=algorithm)
    with patch.object(snooper, "_dump_elementwise_stats", wraps=snooper._dump_elementwise_stats) as stats:
        output = snooper.run(config)

    assert Path(output.csv_snooping_report).exists(), "CSV snooping report was not generated"
    assert Path(output.json_snooping_report).exists(), "JSON snooping report was not generated"

    csv_type = "partial" if set_intermediates else "full"
    if not set_cpu_layers:  # no reference CSVs exist for the mixed-backend runs (legacy TODO)
        reference_csv = asset_dir / f"{algorithm.value}_{csv_type}.csv"
        assert_csv_report(Path(output.csv_snooping_report), reference_csv)
        assert_cpu_retry_reported(Path(output.csv_snooping_report), reference_csv, caplog.text)
    else:
        rows = pd.read_csv(output.csv_snooping_report)
        cpu_rows = rows[rows["Layer Name"].isin(R18_CPU_LAYERS) | rows["Layer Type"].isin(R18_CPU_LAYERS)]
        assert not cpu_rows.empty
        assert set(cpu_rows["Layer Status"]) == {"SUCCESS"}

    if dump_elementwise_stats and not set_cpu_layers:
        assert stats.called
        # Legacy requires the log. It is written only when some element differs, which a CPU retry prevents.
        stats_log = tmp_path / "elementwise_stats.log"
        rows = pd.read_csv(output.csv_snooping_report)
        flags = rows.get("Exception", pd.Series(dtype=str)).fillna("").astype(str)
        stats_rows = flags[rows["Layer Name"].isin(R18_ELEMENTWISE_STATS)]
        retried = (not stats_rows.empty and stats_rows.str.contains(CPU_RETRY_MESSAGE).all()) or (
            "Exception" not in rows and CPU_RETRY_MESSAGE in caplog.text
        )
        if not retried:
            assert stats_log.exists(), "elementwise_stats.log was not generated"
        if stats_log.exists():
            assert "Difference=" in stats_log.read_text()


def test_qnn_ep_negative_reference_model_with_layerwise(asset_dir: Path, tmp_path: Path) -> None:
    qdq_model = asset_dir / "r18.qdq.onnx"
    input_list = tmp_path / "input_list.txt"
    input_list.write_text(f'{R18_INPUT_NAME}:="{asset_dir / "dog.raw"}"\n', encoding="utf8")
    with pytest.raises(ValueError, match="reference_model is supported only with oneshot algorithm."):
        OrtQnnSnooperInputConfig(
            reference_model=qdq_model,
            qdq_model=qdq_model,
            input_processor=InputProcessor(input_list=input_list, model=qdq_model),
            algorithm=Algorithm.LAYERWISE,
            working_directory=tmp_path,
        )


def _assert_trace_report(rows: pd.DataFrame) -> None:
    """Trace columns are present and never name EP-inserted layout ops."""
    assert {"QNN_Op_Name", "QNN_Op_Type", "Fusion_Pattern", "Original_ONNX_Nodes", "CPU_Fallback_Reason"} <= set(
        rows.columns
    )
    assert not rows["Layer Name"].astype(str).str.contains("_token_").any()
    assert not rows["QNN_Op_Name"].fillna("").astype(str).str.contains("Transpose_token_").any()


@pytest.mark.parametrize("set_intermediates", [False, True])
def test_qnn_ep_oneshot_op_trace(
    asset_dir: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture, set_intermediates: bool
) -> None:
    config = _snooper_config(
        asset_dir,
        tmp_path,
        Algorithm.ONESHOT,
        set_intermediates,
        set_cpu_layers=False,
        dump_elementwise_stats=False,
        enable_framework_op_trace=True,
        dump_qnn_ep_input_graph=True,
    )
    output = OrtQnnSnooper(logger, name=Algorithm.ONESHOT).run(config)

    trace_files = list((tmp_path / "op_trace").glob("run_*/qnn_op_trace.json"))
    assert len(trace_files) == 1
    assert list(trace_files[0].parent.glob("*_qnn_ep_input_graph.json"))

    csv_type = "partial" if set_intermediates else "full"
    reference_csv = asset_dir / f"oneshot_{csv_type}.csv"
    rows = pd.read_csv(output.csv_snooping_report)
    _assert_trace_report(rows)
    # Trace columns are inserted, so compare only the shared layer columns.
    reference = _read_reference(reference_csv)
    assert rows[list(COMPARED_COLUMNS)].equals(reference[list(COMPARED_COLUMNS)])
    # Captured Q/DQ pairs are split between QNN and CPU, so ORT does not rerun the model on CPU (design.md F27).
    assert CPU_RETRY_MESSAGE not in caplog.text
    if set_intermediates:
        dq_rows = rows[rows["Layer Output"].str.endswith("_DequantizeLinear_Output")]
        assert len(dq_rows) == 3
        assert (dq_rows["CPU_Fallback_Reason"] == HOST_DEQUANTIZE_REASON).all()
        assert (rows["mse"] > 0).all()
    if not set_intermediates:
        conv_rows = rows[rows["Layer Type"] == "Conv"]
        assert not conv_rows.empty
        assert conv_rows["QNN_Op_Type"].str.contains("Conv2d").all()

    # A supplied trace gives the same layer columns and QNN metadata.
    (tmp_path / "supplied").mkdir()
    supplied = _snooper_config(
        asset_dir,
        tmp_path / "supplied",
        Algorithm.ONESHOT,
        set_intermediates,
        set_cpu_layers=False,
        dump_elementwise_stats=False,
        op_trace_file=trace_files[0],
    )
    supplied_rows = pd.read_csv(OrtQnnSnooper(logger, name=Algorithm.ONESHOT).run(supplied).csv_snooping_report)
    columns = [*COMPARED_COLUMNS, "QNN_Op_Name", "QNN_Op_Type", "Original_ONNX_Nodes"]
    assert supplied_rows[columns].fillna("").equals(rows[columns].fillna(""))


def test_qnn_ep_layerwise_op_trace(asset_dir: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    config = _snooper_config(
        asset_dir,
        tmp_path,
        Algorithm.LAYERWISE,
        set_intermediates=False,
        set_cpu_layers=False,
        dump_elementwise_stats=False,
        enable_framework_op_trace=True,
        dump_qnn_ep_input_graph=True,
    )
    output = OrtQnnSnooper(logger, name=Algorithm.LAYERWISE).run(config)

    # Each layer compiles once, so its subgraph directory holds exactly one trace run.
    subgraph_dirs = list((tmp_path / "subgraph_data").iterdir())
    assert subgraph_dirs
    assert all(len(list(path.glob("op_trace/run_*"))) == 1 for path in subgraph_dirs)
    layer_traces = list((tmp_path / "subgraph_data").glob("*/op_trace/run_*/qnn_op_trace.json"))
    assert layer_traces
    assert all(list(path.parent.glob("*_qnn_ep_input_graph.json")) for path in layer_traces)

    reference_csv = asset_dir / "layerwise_full.csv"
    rows = pd.read_csv(output.csv_snooping_report)
    _assert_trace_report(rows)
    assert set(rows["Layer Status"]) == {"SUCCESS"}
    # Trace columns are inserted, so compare only the shared layer columns.
    reference = _read_reference(reference_csv)
    assert rows[list(COMPARED_COLUMNS)].equals(reference[list(COMPARED_COLUMNS)])
    assert_cpu_retry_reported(Path(output.csv_snooping_report), reference_csv, caplog.text)
    assert CPU_RETRY_MESSAGE not in caplog.text
