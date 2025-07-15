# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT


import argparse
import onnxruntime
import os
import sys

GENERATE_DLC = True

def main():
    parser = argparse.ArgumentParser(description="Converts a QDQ model using ORT-QNN")
    parser.add_argument("--model", required=True, help="Path to the model file to convert")
    args = parser.parse_args()

    model_path = args.model

    try:
        print(f"Converting QDQ model corresponding to: {model_path}")

        qdq_path = os.path.join(os.path.dirname(model_path), f"{model_path.replace('.aimet', '')}_qdq.onnx")
        if not os.path.exists(qdq_path):
            print(f"No QDQ directory at {qdq_path}")
            sys.exit(1)

        qdq_dir_contents = os.listdir(qdq_path)
        if len(qdq_dir_contents) != 2:
            print(f"Expected 2 files in {qdq_path}, found {len(qdq_dir_contents)}")
            sys.exit(1)

        # Convert the model using ORT-QNN

        sess_options = onnxruntime.SessionOptions()
        sess_options.add_session_config_entry("ep.context_enable", "1")
        ep_context_onnx_model_path = os.path.join(
            os.path.dirname(model_path), model_path.replace('.aimet', '_ep_converted.onnx')
        )
        sess_options.add_session_config_entry("ep.context_file_path", ep_context_onnx_model_path)
        sess_options.log_severity_level = 0
        sess_options.log_verbosity_level = 1
        print(f"Saving EP context binary to {ep_context_onnx_model_path}")

        artifact_dir = model_path.replace('.aimet', '_onnx')
        if not GENERATE_DLC:
            provider_options = [
                {
                    "backend_path": "libQnnHtp.so",
                    "htp_graph_finalization_optimization_mode": 3,
                    "soc_model": "60",
                    "htp_arch": "73",
                    "vtcm_mb": "8",
                    "dump_json_qnn_graph": "1",
                    "json_qnn_graph_dir": artifact_dir,
                }
            ]
        else:
            provider_options = [
                {
                    "dump_qnn_ir_dlc": 1,
                    "dump_qnn_ir_dlc_dir": artifact_dir
                }
            ]


        sess = onnxruntime.InferenceSession(
            os.path.join(qdq_path, "model.onnx"),
            sess_options=sess_options,
            providers=["QNNExecutionProvider"],
            provider_options=provider_options,
        )

        print("Successfully generated EP context binary.")

    except Exception as e:
        print(f"Failed to convert QDQ model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
