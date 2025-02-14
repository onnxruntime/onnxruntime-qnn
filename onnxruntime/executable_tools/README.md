# Onnxruntime Executable Tools
The tool runs onnxruntime session inference on inputs and save the outputs. The inputs/outputs can be .pb or .raw format.

## Model and Inputs Data Directory Structure
The tool expect the onnx model and inputs data to be arranged in the following directory structure:
```bash
resnet18-v1-7
├── resnet18-v1-7.onnx
├── test_data_set_0   
│   └── input_0.pb (input_0.raw)
└── test_data_set_1   
    └── input_0.pb (input_0.raw)
```

## Command Line Usage
1. The following command serves as an example to run the tool
    ```ps1
    # executable_tools.exe <model_dir> <backend_path> <"pb"/"raws">
    .\executable_tools.exe resnet18-v1-7 QnnCpu.dll pb
    ```

2. The executable will produce .pb / .raw under the corresponding directory
    ```bash
    resnet18-v1-7
    ├── resnet18-v1-7.onnx
    ├── test_data_set_0   
    │   ├── input_0.pb (input_0.raw)
    │   └── out_0.pb (out_0.raw)
    └── test_data_set_1   
        ├── input_0.pb (input_0.raw)
        └── out_0.pb (out_0.raw)
    ```