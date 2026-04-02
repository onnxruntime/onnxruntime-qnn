# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

# The following script demonstrates how to use the QNNExecutionProvider with an ONNX protobuf which
# wraps a Genie-prepared DLC. Execution will take place via Genie for one execution of an LLM's 
# auto-regressive loop. This portion of model execution takes in tokens as input and gives logits as
# output. Additional functionalities are enabled via the use of onnxruntime-genai.

# Setup Instructions:
# 1. Build the EP
# 2. Create a virtual environment
# 3. Install the EP wheel

import argparse
import onnxruntime

from pathlib import Path

parser = argparse.ArgumentParser(description="Executes a Genie model with the onnxruntime-qnn execution provider")

parser.add_argument('--onnx_network', type=str, required=True, help='Path to the .onnx network file which wraps a Genie-compatible DLC')

args = parser.parse_args()

onnx_network_path = Path(args.onnx_network).resolve()

sess_options = onnxruntime.SessionOptions()
sess_options.log_severity_level = 0

sess = onnxruntime.InferenceSession(
    onnx_network_path, 
    sess_options=sess_options, 
    providers=['QNNExecutionProvider'],
    provider_options=[{"backend_type": "genie"}]
)