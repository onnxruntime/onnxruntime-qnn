import numpy as np
import onnxruntime as ort

qnn_options = {
    "backend_path": "QnnCpu.dll"
}

ort.register_execution_provider_library(
    "QnnAbiTestProvider",
    "onnxruntime_providers_qnn_abi.dll"
)
session_option = ort.SessionOptions()
# Use [ort.get_ep_devices()[1]]. Otherwise, it raises 
# "All OrtEpDevice values in ep_devices must have the same execution provider"
session_option.add_provider_for_devices(
    [ort.get_ep_devices()[1]],
    qnn_options
)

# Do not pass providers and provider_options as they override the one in SessionOptions
session = ort.InferenceSession(
    "layout_transform_reshape.onnx",
    sess_options=session_option
)
assert "QnnAbiTestProvider" in session.get_providers()

ort.unregister_execution_provider_library("QnnAbiTestProvider")
