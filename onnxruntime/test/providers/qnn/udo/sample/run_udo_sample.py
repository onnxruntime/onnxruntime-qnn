# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""
Python reference sample: run MyAdd UDO on QNN EP.

MyAdd computes: output = input + constant
  input  : shape [1, 32], float32 values in [-1, 1]
  output : shape [1, 32], float32

Before running, export the custom-op domain so the QNN EP factory can register
the schema automatically (no separate schema library needed):

  export ORT_QNN_CUSTOM_OP_DOMAINS="example:MyAdd"

Usage:
  # QNN CPU backend (fp32 model)
  LD_LIBRARY_PATH=<qnn_sdk>/lib/x86_64-linux-clang:<ort_lib> \\
  python run_udo_sample.py cpu myadd_fp32.onnx \\
      --op-package ../libMyAddOpPackage_cpu.so \\
      --qnn-ep-lib <path/to/libonnxruntime_providers_qnn.so>

  # QNN HTP backend (QDQ model, x86 simulator or on-device)
  LD_LIBRARY_PATH=<qnn_sdk>/lib/x86_64-linux-clang:<ort_lib> \\
  python run_udo_sample.py htp myadd_qdq.onnx \\
      --op-package ../libMyAddOpPackage_htp.so \\
      --qnn-ep-lib <path/to/libonnxruntime_providers_qnn.so>

Domain registration:
  ORT_QNN_CUSTOM_OP_DOMAINS is read by the QNN EP factory at library-load time.
  It registers a placeholder schema for the custom domain so ORT can load the
  model.  The actual QNN hardware kernel comes from the --op-package library.

QNN EP registration (v2 plugin API):
  The QNN EP is registered as a plugin library via
  ort.register_execution_provider_library() + ort.get_ep_devices() +
  so.add_provider_for_devices(), matching the C++ AppendExecutionProvider_V2 path.
"""

import argparse
import os
import sys
import numpy as np
import onnxruntime as ort

CONSTANT = 2.0
INPUT_SHAPE = (1, 32)
QNN_EP_NAME = "QNNExecutionProvider"


def build_input() -> np.ndarray:
    n = INPUT_SHAPE[1]
    return np.linspace(-1.0, 1.0, n, dtype=np.float32).reshape(INPUT_SHAPE)


def append_qnn_ep(so: ort.SessionOptions, qnn_ep_lib: str, ep_options: dict) -> None:
    """Register the QNN EP plugin and append it to session options (v2 API).

    qnn_ep_lib must be an absolute path to libonnxruntime_providers_qnn.so.
    """
    ort.register_execution_provider_library(QNN_EP_NAME, qnn_ep_lib)
    devices = [d for d in ort.get_ep_devices() if d.ep_name == QNN_EP_NAME]
    if not devices:
        raise RuntimeError("No QNN EP device found after registration.")
    so.add_provider_for_devices(devices, ep_options)


def run_cpu(model_path: str, op_package: str, qnn_ep_lib: str) -> None:
    print("\n=== QNN CPU backend ===")
    so = ort.SessionOptions()

    op_packages_str = f"MyAdd:{op_package}:MyAddOpPackageInterfaceProvider"
    append_qnn_ep(so, qnn_ep_lib, {"backend_type": "cpu", "op_packages": op_packages_str})

    sess = ort.InferenceSession(model_path, sess_options=so)
    x = build_input()
    [output] = sess.run(["output"], {"input": x})

    expected = x + CONSTANT
    max_err = float(np.max(np.abs(output - expected)))
    print(f"Max absolute error vs (input + {CONSTANT}): {max_err:.2e}")
    if max_err > 1e-4:
        print("FAIL: error exceeds threshold")
        sys.exit(1)
    print("PASS")


def run_htp(model_path: str, op_package: str, qnn_ep_lib: str) -> None:
    print("\n=== QNN HTP backend ===")
    so = ort.SessionOptions()

    op_packages_str = f"MyAdd:{op_package}:MyAddOpPackageInterfaceProvider:CPU"
    append_qnn_ep(so, qnn_ep_lib, {
        "backend_type": "htp",
        "offload_graph_io_quantization": "0",
        "op_packages": op_packages_str,
    })

    sess = ort.InferenceSession(model_path, sess_options=so)
    x = build_input()
    [output] = sess.run(["output"], {"input": x})

    expected = x + CONSTANT
    max_err = float(np.max(np.abs(output - expected)))
    # QDQ tolerance: 2x the output quantization scale (4/255, covering [0,4] output range).
    qdq_tol = 4.0 / 255.0 * 2
    print(f"Max absolute error vs (input + {CONSTANT}): {max_err:.4f}  (QDQ tol: {qdq_tol:.4f})")
    if max_err > qdq_tol:
        print("FAIL: error exceeds QDQ tolerance")
        sys.exit(1)
    print("PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MyAdd UDO on QNN EP")
    parser.add_argument("backend", choices=["cpu", "htp"], help="QNN backend to use")
    parser.add_argument("model", help="Path to ONNX model (myadd_fp32.onnx or myadd_qdq.onnx)")
    parser.add_argument("--op-package", required=True,
                        help="Path to libMyAddOpPackage_<backend>.so")
    parser.add_argument("--qnn-ep-lib", required=True,
                        help="Absolute path to libonnxruntime_providers_qnn.so")
    args = parser.parse_args()

    if args.backend == "cpu":
        run_cpu(args.model, args.op_package, args.qnn_ep_lib)
    else:
        run_htp(args.model, args.op_package, args.qnn_ep_lib)


if __name__ == "__main__":
    main()
