#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Fail-open consumer for published HTP ModelZoo graph snapshots."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graph_snapshot import QNN_JSON_GRAPH_SCHEMA_VERSION


@dataclass(frozen=True)
class SnapshotGateResult:
    skip_real_execution: bool
    reason: str


def _load_manifest(golden_root: Path) -> dict[str, Any]:
    manifest_path = golden_root / "manifest.json"
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("Snapshot manifest must be a JSON object.")
    return manifest


def _compare_model_snapshots(current_dir: Path, golden_dir: Path) -> str | None:
    current_files = sorted(path.relative_to(current_dir) for path in current_dir.rglob("*.json"))
    golden_files = sorted(path.relative_to(golden_dir) for path in golden_dir.rglob("*.json"))
    if current_files != golden_files:
        return "snapshot file set differs"
    for relative_path in current_files:
        current = (current_dir / relative_path).read_bytes()
        golden = (golden_dir / relative_path).read_bytes()
        if current != golden:
            return f"snapshot differs: {relative_path.as_posix()}"
    return None


def check_snapshot_gate(
    current_dir: Path,
    golden_root: Path,
    relative_model_dir: Path,
    modelzoo_platform: str,
    htp_arch: str,
    ort_version: str,
    qairt_version: str,
) -> SnapshotGateResult:
    """Return a skip decision, treating every validation problem as unverified.

    The caller must run normal ModelZoo execution whenever this returns False.
    """
    try:
        manifest = _load_manifest(golden_root)
        expected = {
            "snapshot_schema_version": QNN_JSON_GRAPH_SCHEMA_VERSION,
            "modelzoo_platform": modelzoo_platform,
            "htp_arch": htp_arch,
            "ort_version": ort_version,
            "qairt_version": qairt_version,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                return SnapshotGateResult(False, f"manifest {key} mismatch")
        difference = _compare_model_snapshots(current_dir, golden_root / relative_model_dir)
        if difference is not None:
            return SnapshotGateResult(False, difference)
        return SnapshotGateResult(True, "snapshot matches compatible golden")
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as error:
        return SnapshotGateResult(False, f"golden is unavailable or invalid: {error}")
