# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import json
from pathlib import Path

import pytest
from graph_golden import QNN_GRAPH_GOLDEN_MANIFEST, finalize_qnn_graph_golden_dir, normalize_qnn_graph


def _graph_dump() -> dict:
    return {
        "op_types": ["Relu", "Conv2d"],
        "graph": {
            "tensors": {"weights": {"id": 77, "params_count": "4"}},
            "nodes": {},
        },
    }


def _gating_metadata(graph_file: str, *, schema_version: int = 1, htp_arch: int = 73) -> dict:
    return {
        "schema_version": schema_version,
        "graph_file": graph_file,
        "backend": "htp",
        "htp_arch": htp_arch,
        "backend_config": {"vtcm_size_in_mb": 0},
        "static_tensor_hashes": {
            "graph_tensors": {
                "weights": {
                    "algorithm": "sha256",
                    "byte_count": 4,
                    "digest": "a" * 64,
                }
            }
        },
    }


def test_normalize_qnn_graph_preserves_qnn_format() -> None:
    normalized = normalize_qnn_graph(_graph_dump())

    assert "id" not in normalized["graph"]["tensors"]["weights"]
    assert normalized["op_types"] == ["Conv2d", "Relu"]
    assert "qnn_json_graph_schema_version" not in normalized
    assert "qnn_ep_graph_configs" not in normalized


def test_finalize_qnn_graph_golden_dir_creates_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    graph_path = tmp_path / "subgraph.json"
    metadata_path = tmp_path / "subgraph.golden_metadata.json"
    tensor_log_path = tmp_path / "subgraph_tensor_log.json"
    graph_path.write_text(json.dumps(_graph_dump()), encoding="utf-8")
    metadata_path.write_text(json.dumps(_gating_metadata(graph_path.name)), encoding="utf-8")
    tensor_log_path.write_text('{"id": 99}', encoding="utf-8")
    monkeypatch.setattr("graph_golden.platform.system", lambda: "Windows")
    monkeypatch.setattr("graph_golden.platform.machine", lambda: "ARM64")

    graph_files, manifest_path = finalize_qnn_graph_golden_dir(tmp_path)

    assert graph_files == [graph_path]
    assert manifest_path == tmp_path / QNN_GRAPH_GOLDEN_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["target"] == {
        "os": "windows",
        "cpu_arch": "arm64",
        "backend": "htp",
        "htp_arch": "v73",
    }
    assert manifest["graphs"][graph_path.name]["backend_config"] == {"vtcm_size_in_mb": 0}
    assert manifest["graphs"][graph_path.name]["static_tensor_hashes"]["graph_tensors"]["weights"]["digest"] == "a" * 64
    assert "static_data_hash" not in json.loads(graph_path.read_text(encoding="utf-8"))["graph"]["tensors"]["weights"]
    assert not metadata_path.exists()
    assert not tensor_log_path.exists()


def test_finalize_qnn_graph_golden_dir_rejects_unknown_metadata_schema(tmp_path: Path) -> None:
    graph_path = tmp_path / "subgraph.json"
    metadata_path = tmp_path / "subgraph.golden_metadata.json"
    graph_path.write_text(json.dumps(_graph_dump()), encoding="utf-8")
    metadata_path.write_text(json.dumps(_gating_metadata(graph_path.name, schema_version=2)), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported QNN graph golden metadata schema"):
        finalize_qnn_graph_golden_dir(tmp_path)

    assert metadata_path.exists()
