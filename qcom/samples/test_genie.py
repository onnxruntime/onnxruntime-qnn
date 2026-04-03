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
import numpy as np
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

from pathlib import Path

parser = argparse.ArgumentParser(description="Executes a Genie model with the onnxruntime-qnn execution provider")

parser.add_argument('--onnx-network', type=str, required=True, help='Path to the .onnx network file which wraps a Genie-compatible DLC')

args = parser.parse_args()

onnx_network_path = Path(args.onnx_network).resolve()

# Path to the plugin EP library
ep_lib_path = qnn_ep.get_library_path()
# Registration name can be anything the application chooses
ep_registration_name = "QNNExecutionProvider"

# Register plugin EP library with ONNX Runtime
ort.register_execution_provider_library(ep_registration_name, ep_lib_path)

# Get EP name(s) from the plugin EP library
ep_names = qnn_ep.get_ep_names()
# For this example we'll use the first one
ep_name = ep_names[0]

# Select an OrtEpDevice
# For this example, we'll use any OrtEpDevices matching our EP name
all_ep_devices = ort.get_ep_devices()
selected_ep_devices = [ep_device for ep_device in all_ep_devices if ep_device.ep_name == ep_name]

assert len(selected_ep_devices) > 0

sess_options = ort.SessionOptions()
sess_options.log_severity_level = 0

# EP-specific options
ep_options = {"backend_type": "genie"}

# Equivalent to the C API's SessionOptionsAppendExecutionProvider_V2 that appends the plugin EP to the session options
sess_options.add_provider_for_devices(selected_ep_devices, ep_options)

assert sess_options.has_providers()

# Create ORT session with the plugin EP
sess = ort.InferenceSession(onnx_network_path, sess_options=sess_options)

# Print detected input metadata
print("\nModel inputs:")
for inp in sess.get_inputs():
    print(f"  {inp.name}: shape={inp.shape}, dtype={inp.type}")

# Build input feed dynamically from the model's input metadata
input_feed = {}
for inp in sess.get_inputs():
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    input_feed[inp.name] = np.zeros(shape, dtype=np.float32)

# Run inference
outputs = sess.run(None, input_feed)

print("\nInference completed successfully!")
print(f"Number of outputs: {len(outputs)}")
for i, output in enumerate(outputs):
    print(f"Output {i} shape: {output.shape}, dtype: {output.dtype}")

del sess

# Unregister the library using the same registration name specified earlier
# Must only unregister a library after all sessions that use the library have been released
ort.unregister_execution_provider_library(ep_registration_name)

print(f"Unregister {ep_registration_name} successfully!")