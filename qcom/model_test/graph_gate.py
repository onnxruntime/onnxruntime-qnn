# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Graph-diff gating: skip accuracy verification when QNN graph is unchanged.

When the ORT/QAIRT versions haven't changed and the compiled QNN graph JSON
matches the golden (generated from main branch), inference behavior is
guaranteed identical. This module implements the comparison logic used by
model_zoo_test.py to skip redundant accuracy checks.
"""

import json
import logging
from pathlib import Path
from typing import NamedTuple

import onnxruntime_qnn

GOLDEN_MANIFEST_FILENAME = "golden_manifest.json"

logger = logging.getLogger(__name__)


class GateResult(NamedTuple):
    skip_accuracy: bool
    reason: str


def get_current_versions() -> dict[str, str]:
    """Return current ORT + QAIRT versions from the installed wheel."""
    return {
        "ort_version": onnxruntime_qnn.build_and_package_info.__version__,
        "qnn_version": onnxruntime_qnn.build_and_package_info.qnn_version,
    }


def load_golden_manifest(golden_dir: Path) -> dict | None:
    """Load golden_manifest.json containing versions and graph file list."""
    manifest_path = golden_dir / GOLDEN_MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text())


def normalize_graph_json(graph: dict) -> dict:
    """Strip non-deterministic fields (tensor IDs) for stable comparison.

    Mirrors the NormalizeQnnJSONGraph logic from snapshot.h (PR #399).
    """
    if "graph" in graph and "tensors" in graph["graph"]:
        for tensor in graph["graph"]["tensors"].values():
            if isinstance(tensor, dict):
                tensor.pop("id", None)
    return graph


def compare_graphs(dumped_dir: Path, golden_dir: Path) -> tuple[bool, str]:
    """Compare all dumped graph JSONs against goldens.

    Returns (match: bool, detail: str).
    Ignores tensor_log files (metadata only, not graph structure).
    """
    dumped_files = sorted(f for f in dumped_dir.glob("*.json") if "_tensor_log" not in f.name)
    if not dumped_files:
        return False, "no graph JSONs were dumped"

    golden_files = sorted(
        f for f in golden_dir.glob("*.json") if f.name != GOLDEN_MANIFEST_FILENAME and "_tensor_log" not in f.name
    )
    golden_names = {f.name for f in golden_files}
    dumped_names = {f.name for f in dumped_files}

    # Check for new/removed subgraphs
    if dumped_names != golden_names:
        added = dumped_names - golden_names
        removed = golden_names - dumped_names
        parts = []
        if added:
            parts.append(f"new subgraphs: {sorted(added)}")
        if removed:
            parts.append(f"removed subgraphs: {sorted(removed)}")
        return False, "; ".join(parts)

    # Compare each graph file
    for dumped_file in dumped_files:
        golden_file = golden_dir / dumped_file.name
        dumped = normalize_graph_json(json.loads(dumped_file.read_text()))
        golden = normalize_graph_json(json.loads(golden_file.read_text()))
        if dumped != golden:
            return False, f"graph diff in {dumped_file.name}"

    return True, "all graphs match"


def check_gate(model_root: Path, dumped_dir: Path) -> GateResult:
    """Main gate check: version match + graph diff.

    Args:
        model_root: Path to the model directory (contains goldens/ subdirectory).
        dumped_dir: Directory where the current run's graph JSONs were dumped.

    Returns:
        GateResult indicating whether accuracy can be skipped.
    """
    golden_dir = model_root / "goldens"

    # Check golden exists
    manifest = load_golden_manifest(golden_dir)
    if manifest is None:
        return GateResult(skip_accuracy=False, reason="no golden manifest found")

    # Check version match
    current = get_current_versions()
    golden_ort = manifest.get("ort_version")
    golden_qnn = manifest.get("qnn_version")

    if current["ort_version"] != golden_ort:
        return GateResult(
            skip_accuracy=False,
            reason=f"ORT version changed: {golden_ort} -> {current['ort_version']}",
        )
    if current["qnn_version"] != golden_qnn:
        return GateResult(
            skip_accuracy=False,
            reason=f"QAIRT version changed: {golden_qnn} -> {current['qnn_version']}",
        )

    # Compare graphs
    match, detail = compare_graphs(dumped_dir, golden_dir)
    if match:
        return GateResult(skip_accuracy=True, reason="graph unchanged")

    return GateResult(skip_accuracy=False, reason=f"graph diff detected: {detail}")
