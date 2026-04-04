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
from pathlib import Path

import numpy as np
import onnxruntime_qnn as qnn_ep

import onnxruntime as ort

parser = argparse.ArgumentParser(description="Executes a Genie model with the onnxruntime-qnn execution provider")

parser.add_argument(
    "--onnx-network", type=str, required=True, help="Path to the .onnx network file which wraps a Genie-compatible DLC"
)
parser.add_argument(
    "--genie-log-level",
    type=str,
    default="error",
    choices=["error", "warn", "info", "verbose"],
    help="Genie log level (default: error)",
)
parser.add_argument(
    "--input-shape",
    type=int,
    nargs="+",
    help="Custom input shape (e.g., --input-shape 1 128). If not provided, uses shape from model with dynamic dims set to 1",
)
parser.add_argument("--vocab-size", type=int, help="Vocabulary size for generating random token IDs")
parser.add_argument(
    "--input-tokens-file",
    type=str,
    help="Path to file containing input tokens (one token per line or space/comma separated)",
)

args = parser.parse_args()

onnx_network_path = Path(args.onnx_network).resolve()


# Helper function to convert ONNX type string to NumPy dtype
def onnx_type_to_numpy_dtype(onnx_type_str):
    # Extract the type from strings like "tensor(int32)" or "tensor(float)"
    type_map = {
        "float": np.float32,
        "float32": np.float32,
        "float64": np.float64,
        "double": np.float64,
        "int32": np.int32,
        "int64": np.int64,
        "int8": np.int8,
        "int16": np.int16,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
        "bool": np.bool_,
    }

    # Extract type name from "tensor(type)" format
    if "tensor(" in onnx_type_str:
        type_name = onnx_type_str.split("(")[1].rstrip(")")
    else:
        type_name = onnx_type_str

    return type_map.get(type_name, np.float32)  # Default to float32 if unknown


# Helper function to load tokens from file
def load_tokens_from_file(file_path):
    """Load tokens from a file. Supports space-separated, comma-separated, or newline-separated tokens."""
    with open(file_path) as f:
        content = f.read().strip()

    # Try to parse as space or comma separated
    tokens = []
    for line in content.split("\n"):
        stripped_line = line.strip()
        if "," in stripped_line:
            tokens.extend([int(t.strip()) for t in stripped_line.split(",") if t.strip()])
        else:
            tokens.extend([int(t.strip()) for t in stripped_line.split() if t.strip()])

    return np.array(tokens)


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
ep_options = {"genie_log_level": args.genie_log_level}

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
    dtype = onnx_type_to_numpy_dtype(inp.type)
    if args.input_shape:
        shape = args.input_shape
    else:
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    # Load tokens from file if provided
    if args.input_tokens_file:
        tokens = load_tokens_from_file(args.input_tokens_file)
        # Reshape tokens to match shape
        total_size = np.prod(shape)
        if len(tokens) < total_size:
            # Pad with zeros if not enough tokens
            tokens = np.pad(tokens, (0, total_size - len(tokens)), mode="constant")
        elif len(tokens) > total_size:
            # Truncate if too many tokens
            tokens = tokens[:total_size]
        input_feed[inp.name] = tokens.reshape(shape).astype(dtype)
    else:
        if np.issubdtype(dtype, np.integer):
            # For integer types, generate random token IDs within vocab size if provided
            if args.vocab_size:
                input_feed[inp.name] = np.random.randint(0, args.vocab_size, size=shape, dtype=dtype)
            else:
                # Default to small random integers if vocab_size not specified
                input_feed[inp.name] = np.zeros(shape, dtype=onnx_type_to_numpy_dtype(inp.type))
        else:
            # Fallback to zeros for other types
            input_feed[inp.name] = np.zeros(shape, dtype=dtype)

print(f"  Generated input '{inp.name}': shape={input_feed[inp.name].shape}, dtype={input_feed[inp.name].dtype}")

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
