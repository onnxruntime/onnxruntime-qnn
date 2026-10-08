#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Normalization helpers for opt-in QNN ModelZoo graph snapshots."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

QNN_JSON_GRAPH_SCHEMA_VERSION = 1


def normalize_qnn_graph(graph: dict[str, Any]) -> dict[str, Any]:
    """Return the stable representation of one QNN graph dump.

    Tensor IDs are allocated by QNN and are not part of the graph contract.
    Every other field, including static tensor byte hashes and HTP graph
    settings, remains part of the snapshot.
    """
    if graph.get("qnn_json_graph_schema_version") != QNN_JSON_GRAPH_SCHEMA_VERSION:
        actual_version = graph.get("qnn_json_graph_schema_version")
        raise ValueError(
            f"Unsupported QNN graph snapshot schema: expected {QNN_JSON_GRAPH_SCHEMA_VERSION}, got {actual_version!r}."
        )

    normalized = deepcopy(graph)
    tensors = normalized.get("graph", {}).get("tensors", {})
    if isinstance(tensors, dict):
        for tensor in tensors.values():
            if isinstance(tensor, dict):
                tensor.pop("id", None)

    op_types = normalized.get("op_types")
    if isinstance(op_types, list) and all(isinstance(op_type, str) for op_type in op_types):
        op_types.sort()

    return normalized


def normalize_qnn_graph_dump_dir(dump_dir: Path) -> list[Path]:
    """Normalize graph dumps in place and return their deterministic file list."""
    graph_files = sorted(path for path in dump_dir.glob("*.json") if not path.name.endswith("_tensor_log.json"))
    if not graph_files:
        raise RuntimeError(f"No QNN graph JSON files were created in {dump_dir}.")

    for graph_file in graph_files:
        normalized = normalize_qnn_graph(json.loads(graph_file.read_text(encoding="utf-8")))
        graph_file.write_text(
            json.dumps(normalized, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    return graph_files
