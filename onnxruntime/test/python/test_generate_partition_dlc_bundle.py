# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Tiny self-checks for generate_partition_dlc_bundle.

Run: python test_generate_partition_dlc_bundle.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

REPO_ROOT = Path(__file__).resolve().parents[3]
HELPER = (
    REPO_ROOT / "onnxruntime" / "python" / "tools" / "qnn" / "partition_dlc_bundle" / "generate_partition_dlc_bundle.py"
)


def _make_two_partition_model(path: Path):
    # Input -> Add -> [boundary T_a] -> Mul -> Output
    a = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])
    out = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])
    one = helper.make_tensor("one", TensorProto.FLOAT, [4], [1, 1, 1, 1])
    two = helper.make_tensor("two", TensorProto.FLOAT, [4], [2, 2, 2, 2])
    add = helper.make_node("Add", ["X", "one"], ["T_a"])
    mul = helper.make_node("Mul", ["T_a", "two"], ["Y"])
    g = helper.make_graph([add, mul], "g", [a], [out], initializer=[one, two])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, str(path))


def _fake_manifest():
    return {
        "bundle_version": 1,
        "partitions": [
            {
                "name": "qnn_0",
                "dlc_path": "partitions/qnn_0.dlc",
                "inputs": [{"name": "X", "dtype": "float32", "shape": [4]}],
                "outputs": [{"name": "T_a", "dtype": "float32", "shape": [4]}],
            },
            {
                "name": "qnn_1",
                "dlc_path": "partitions/qnn_1.dlc",
                "inputs": [{"name": "T_a", "dtype": "float32", "shape": [4]}],
                "outputs": [{"name": "Y", "dtype": "float32", "shape": [4]}],
            },
        ],
        "edges": [{"producer_partition": "qnn_0", "consumer_partition": "qnn_1", "tensor_name": "T_a"}],
    }


def test_bundle_fill():
    with tempfile.TemporaryDirectory() as td_str:
        td = Path(td_str)
        model_path = td / "m.onnx"
        _make_two_partition_model(model_path)

        bundle = td / "bundle"
        bundle.mkdir()
        (bundle / "manifest.json").write_text(json.dumps(_fake_manifest()))

        # Fake the compile-time DLCs the helper consolidates into each partition folder.
        (bundle / "partitions").mkdir()
        for p in ("qnn_0", "qnn_1"):
            (bundle / "partitions" / f"{p}.dlc").write_bytes(b"DLC" + p.encode())

        x = np.array([10, 20, 30, 40], dtype=np.float32)
        x_path = td / "x.raw"
        x.tofile(x_path)

        rc = subprocess.call(
            [
                sys.executable,
                str(HELPER),
                "--bundle-dir",
                str(bundle),
                "--model",
                str(model_path),
                "--inputs",
                f"X={x_path}",
            ]
        )
        assert rc == 0, "helper exited non-zero"

        for p, kind, name, expected in [
            ("qnn_0", "inputs", "X", x),
            ("qnn_0", "goldens", "T_a", x + 1),
            ("qnn_1", "inputs", "T_a", x + 1),
            ("qnn_1", "goldens", "Y", (x + 1) * 2),
        ]:
            data = np.fromfile(bundle / p / kind / f"{name}.raw", dtype=np.float32)
            assert np.allclose(data, expected), f"{p}/{kind}/{name} mismatch: {data} != {expected}"

        # DLC, inputs and goldens now live together under <bundle>/<partition>/.
        for p in ("qnn_0", "qnn_1"):
            assert (bundle / p / f"{p}.dlc").is_file(), f"{p}.dlc not consolidated"
        assert not (bundle / "partitions").exists(), "empty partitions/ should be removed"
        manifest_after = json.loads((bundle / "manifest.json").read_text())
        for p in manifest_after["partitions"]:
            assert p["dlc_path"] == f"{p['name']}/{p['name']}.dlc", p["dlc_path"]
        assert manifest_after["goldens_source"] == "cpu"


if __name__ == "__main__":
    test_bundle_fill()
    print("ok")
