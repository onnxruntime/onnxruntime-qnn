#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Helpers for producing canonical QNN graph goldens for ModelZoo validation."""

import json
import platform
from copy import deepcopy
from pathlib import Path
from typing import Any

QNN_GRAPH_GOLDEN_SCHEMA_VERSION = 1
QNN_GRAPH_GOLDEN_MANIFEST = "qnn_graph_golden_manifest.json"
QNN_GRAPH_GOLDEN_METADATA_SUFFIX = ".golden_metadata.json"


def normalize_qnn_graph(graph: dict[str, Any]) -> dict[str, Any]:
    """Return a stable QNN Netron-compatible graph representation."""
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


def _normalize_cpu_arch(machine: str) -> str:
    normalized = machine.lower()
    if normalized in {"aarch64", "arm64"}:
        return "arm64"
    if normalized in {"amd64", "x86_64"}:
        return "x86_64"
    return normalized


def _load_graph_metadata(metadata_path: Path, graph_file: str) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual_version = metadata.get("schema_version")
    if actual_version != QNN_GRAPH_GOLDEN_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported QNN graph golden metadata schema: "
            f"expected {QNN_GRAPH_GOLDEN_SCHEMA_VERSION}, got {actual_version!r}."
        )
    if metadata.get("graph_file") != graph_file:
        raise ValueError(f"QNN graph golden metadata {metadata_path} does not describe {graph_file}.")
    if metadata.get("backend") != "htp":
        raise ValueError(f"QNN graph golden metadata {metadata_path} is not for the HTP backend.")

    htp_arch = metadata.get("htp_arch")
    if not isinstance(htp_arch, int) or htp_arch <= 0:
        raise ValueError(f"QNN graph golden metadata {metadata_path} has invalid HTP arch {htp_arch!r}.")
    return metadata


def finalize_qnn_graph_golden_dir(dump_dir: Path) -> tuple[list[Path], Path]:
    """Normalize QNN graphs and create one ORT-owned golden manifest."""
    graph_files = sorted(
        path
        for path in dump_dir.glob("*.json")
        if not path.name.endswith("_tensor_log.json")
        and not path.name.endswith(QNN_GRAPH_GOLDEN_METADATA_SUFFIX)
        and path.name != QNN_GRAPH_GOLDEN_MANIFEST
    )
    if not graph_files:
        raise RuntimeError(f"No QNN graph JSON files were created in {dump_dir}.")

    manifest_graphs: dict[str, Any] = {}
    target_backend: str | None = None
    target_htp_arch: int | None = None
    metadata_files: list[Path] = []

    for graph_file in graph_files:
        metadata_path = graph_file.with_suffix(QNN_GRAPH_GOLDEN_METADATA_SUFFIX)
        if not metadata_path.exists():
            raise RuntimeError(f"Missing QNN graph golden metadata for {graph_file}.")

        normalized = normalize_qnn_graph(json.loads(graph_file.read_text(encoding="utf-8")))
        graph_file.write_text(
            json.dumps(normalized, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

        metadata = _load_graph_metadata(metadata_path, graph_file.name)
        backend = metadata["backend"]
        htp_arch = metadata["htp_arch"]
        if target_backend is not None and backend != target_backend:
            raise ValueError(f"QNN graph golden metadata mixes backends {target_backend!r} and {backend!r}.")
        if target_htp_arch is not None and htp_arch != target_htp_arch:
            raise ValueError(f"QNN graph golden metadata mixes HTP arches v{target_htp_arch} and v{htp_arch}.")
        target_backend = backend
        target_htp_arch = htp_arch

        manifest_graphs[graph_file.name] = {
            "backend_config": metadata.get("backend_config", {}),
            "static_tensor_hashes": metadata.get("static_tensor_hashes", {}),
        }
        metadata_files.append(metadata_path)

    manifest = {
        "schema_version": QNN_GRAPH_GOLDEN_SCHEMA_VERSION,
        "target": {
            "os": platform.system().lower(),
            "cpu_arch": _normalize_cpu_arch(platform.machine()),
            "backend": target_backend,
            "htp_arch": f"v{target_htp_arch}",
        },
        "graphs": manifest_graphs,
    }
    manifest_path = dump_dir / QNN_GRAPH_GOLDEN_MANIFEST
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    # Diagnostic sidecars are retained until the complete golden is built so a
    # failure leaves enough information to investigate the producer output.
    for metadata_file in metadata_files:
        metadata_file.unlink()
    for tensor_log_file in dump_dir.glob("*_tensor_log.json"):
        tensor_log_file.unlink()

    return graph_files, manifest_path
