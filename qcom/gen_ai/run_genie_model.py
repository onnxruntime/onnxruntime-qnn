import argparse
import onnxruntime

from pathlib import Path

parser = argparse.ArgumentParser(description="Executes a Genie model with the onnxruntime-qnn execution provider")

parser.add_argument('--onnx_network', type=str, required=True, help='Path to the .onnx network file')

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
