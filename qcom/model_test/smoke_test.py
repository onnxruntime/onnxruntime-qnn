# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import json
import os
import warnings
from pathlib import Path

import onnxruntime_qnn
import pytest
from model_test import ModelTestCase, ModelTestDef, ModelTestSuite
from model_zoo_test import get_xfails

import onnxruntime

SMOKE_TESTS = list(
    ModelTestSuite(
        Path(os.getenv("ORT_WHEEL_SMOKE_TEST_ROOT", "HopefullyBogusPath")),
        backend_type="htp",
        rtol=None,
        atol=None,
        cosine_similarity=None,
        enable_context=True,
        enable_cpu_fallback=False,
    ).tests
)

SMOKE_TEST_IDS = [str(st) for st in SMOKE_TESTS]


@pytest.mark.parametrize("test_def", SMOKE_TESTS, ids=SMOKE_TEST_IDS)
def test_models(test_def: ModelTestDef) -> None:
    xfails = get_xfails("ORT_WHEEL_SMOKE_TEST_XFAILS")
    if test_def.model_root.name in xfails:
        pytest.xfail(xfails[test_def.model_root.name])
    ModelTestCase(test_def).run()


def test_qnn_version() -> None:
    assert onnxruntime_qnn.build_and_package_info.qnn_version is not None


def test_json_graph_dump(tmp_path: Path) -> None:
    """Verify that dump_json_qnn_graph produces valid JSON graph files."""
    if not SMOKE_TESTS:
        pytest.skip("No smoke test models available")

    # Log device info for CI visibility (warnings bypass pytest capture)
    ep_devices = [ed for ed in onnxruntime.get_ep_devices() if ed.ep_name == onnxruntime_qnn.get_ep_names()[0]]
    for ed in ep_devices:
        warnings.warn(
            f"[Device Info] ep={ed.ep_name}, type={ed.device.type},"
            f" device_id={ed.device.device_id} (hex={hex(ed.device.device_id)})",
            stacklevel=1,
        )

    test_def = SMOKE_TESTS[0]

    # Use a persistent path for CI artifact upload if available, otherwise tmp_path
    artifact_dir = Path(os.getenv("ORT_JSON_DUMP_ARTIFACT_DIR", str(tmp_path / "json_dump")))
    dump_dir = artifact_dir / test_def.model_root.name
    dump_dir.mkdir(parents=True, exist_ok=True)

    ModelTestCase(test_def, json_dump_dir=dump_dir).dump_graph_only()

    graph_jsons = [f for f in dump_dir.glob("*.json") if "_tensor_log" not in f.name]
    assert len(graph_jsons) > 0, f"No graph JSONs dumped to {dump_dir}"

    for graph_file in graph_jsons:
        data = json.loads(graph_file.read_text())
        assert "graph" in data, f"Missing 'graph' key in {graph_file.name}"
        assert "nodes" in data["graph"], f"Missing 'nodes' in {graph_file.name}"
        assert "tensors" in data["graph"], f"Missing 'tensors' in {graph_file.name}"
        assert len(data["graph"]["nodes"]) > 0, f"Empty nodes in {graph_file.name}"
