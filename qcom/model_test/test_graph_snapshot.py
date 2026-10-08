# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import json
from pathlib import Path

import pytest
from graph_snapshot import normalize_qnn_graph, normalize_qnn_graph_dump_dir


def _graph_dump() -> dict:
    return {
        "qnn_json_graph_schema_version": 1,
        "op_types": ["Relu", "Conv2d"],
        "graph": {
            "tensors": {
                "weights": {
                    "id": 77,
                    "static_data_hash": {
                        "algorithm": "sha256",
                        "byte_count": 4,
                        "digest": "a" * 64,
                    },
                }
            },
            "nodes": {},
        },
    }


def test_normalize_qnn_graph_removes_only_nondeterministic_fields() -> None:
    normalized = normalize_qnn_graph(_graph_dump())

    assert "id" not in normalized["graph"]["tensors"]["weights"]
    assert normalized["graph"]["tensors"]["weights"]["static_data_hash"]["digest"] == "a" * 64
    assert normalized["op_types"] == ["Conv2d", "Relu"]


def test_normalize_qnn_graph_rejects_unknown_schema() -> None:
    with pytest.raises(ValueError, match="Unsupported QNN graph snapshot schema"):
        normalize_qnn_graph({"qnn_json_graph_schema_version": 2})


def test_normalize_qnn_graph_dump_dir_excludes_tensor_logs(tmp_path: Path) -> None:
    graph_path = tmp_path / "subgraph.json"
    tensor_log_path = tmp_path / "subgraph_tensor_log.json"
    graph_path.write_text(json.dumps(_graph_dump()), encoding="utf-8")
    tensor_log_path.write_text('{"id": 99}', encoding="utf-8")

    graph_files = normalize_qnn_graph_dump_dir(tmp_path)

    assert graph_files == [graph_path]
    normalized = json.loads(graph_path.read_text(encoding="utf-8"))
    assert "id" not in normalized["graph"]["tensors"]["weights"]
    assert json.loads(tensor_log_path.read_text(encoding="utf-8")) == {"id": 99}
