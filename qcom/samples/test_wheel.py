# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import argparse

import numpy as np
import onnxruntime_qnn as qnn_ep

import onnxruntime as ort

parser = argparse.ArgumentParser()
parser.add_argument(
    "--disable-cpu-fallback",
    action="store_true",
    help="Fail session creation if the model cannot run with QNN EP.",
)
args = parser.parse_args()

# Path to the plugin EP library
ep_lib_path = qnn_ep.get_library_path()
# Registration name can be anything the application chooses
ep_registration_name = "QNNExecutionProvider"

# Register plugin EP library with ONNX Runtime
ort.register_execution_provider_library(ep_registration_name, ep_lib_path)

# Create ORT session with explicit OrtEpDevice(s)

# Get EP name(s) from the plugin EP library
ep_names = qnn_ep.get_ep_names()
# For this example we'll use the first one
ep_name = ep_names[0]

# Select OrtEpDevice(s) exposed by the QNN EP. Generic wheel smoke tests may
# run on hosts without an NPU, so they retain all advertised QNN devices.
all_ep_devices = ort.get_ep_devices()
selected_ep_devices = [ep_device for ep_device in all_ep_devices if ep_device.ep_name == ep_name]

# Strict validation must exercise the packaged HTP backend on an NPU and must
# not silently pass by falling back to the CPU EP.
if args.disable_cpu_fallback:
    selected_ep_devices = [
        ep_device for ep_device in selected_ep_devices if ep_device.device.type == ort.OrtHardwareDeviceType.NPU
    ]

assert len(selected_ep_devices) > 0

sess_options = ort.SessionOptions()

if args.disable_cpu_fallback:
    sess_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

# EP-specific options
ep_options = {"backend_path": qnn_ep.get_qnn_htp_path()}

# Equivalent to the C API's SessionOptionsAppendExecutionProvider_V2 that appends the plugin EP to the session options
sess_options.add_provider_for_devices(selected_ep_devices, ep_options)

assert sess_options.has_providers()

# Create ORT session with the plugin EP
model_path = "cmake/external/onnx/onnx/backend/test/data/node/test_abs/model.onnx"
sess = ort.InferenceSession(model_path, sess_options=sess_options)

# Create input data for the model
# Input "x" with shape [3, 4, 5] and dtype float32
x = np.random.randn(3, 4, 5).astype(np.float32)

# Run inference
outputs = sess.run(
    None,  # Get all outputs
    {"x": x},
)

print("\nInference completed successfully!")
print(f"Number of outputs: {len(outputs)}")
for i, output in enumerate(outputs):
    print(f"Output {i} shape: {output.shape}, dtype: {output.dtype}")

# Use `sess`
# ...

del sess

# Unregister the library using the same registration name specified earlier
# Must only unregister a library after all sessions that use the library have been released
ort.unregister_execution_provider_library(ep_registration_name)

print(f"Unregister {ep_registration_name} successfully!")
