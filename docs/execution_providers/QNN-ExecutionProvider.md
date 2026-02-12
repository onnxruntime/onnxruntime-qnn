# QNN Execution Provider

The QNN Execution Provider (QNN EP) for ONNX Runtime enables hardware accelerated execution on Qualcomm chipsets.
It uses the Qualcomm AI Runtime SDK (QAIRT SDK) to construct a QNN graph from an ONNX model which can
be executed by a supported accelerator backend library.
ONNX Runtime QNN EP can be used on Windows devices with Qualcomm Snapdragon SOC's.

## Contents

- [Install Pre-requisites (Build from Source Only)](#install-pre-requisites-build-from-source-only)
- [Build (Windows)](#build-windows)
- [Pre-built Packages (Windows Only)](#pre-built-packages-windows-only)
- [Qualcomm AI Hub](#qualcomm-ai-hub)
- [Configuration Options](#configuration-options)
- [Supported ONNX operators](#supported-onnx-operators)
- [Running a model with QNN EP's HTP backend (Python)](#running-a-model-with-qnn-eps-htp-backend-python)
- [Running a model with QNN EP's GPU backend](#running-a-model-with-qnn-eps-gpu-backend)
- [QNN context binary cache feature](#qnn-context-binary-cache-feature)
- [QNN EP Profiling](#qnn-ep-profiling)
- [QNN EP weight sharing](#qnn-ep-weight-sharing)
- [Usage](#usage)
- [Error handling](#error-handling)
- [Add new operator support in QNN EP](#add-new-operator-support-in-qnn-ep)
- [Mixed precision support](#mixed-precision-support)
- [LoRAv2 support](#lorav2-support)

## Install Pre-requisites (Build from Source Only)

If you build QNN EP from source, you should first
download the Qualcomm AI Runtime SDK (QAIRT SDK) from [https://qpm.qualcomm.com/#/main/tools/details/Qualcomm_AI_Runtime_SDK](https://qpm.qualcomm.com/#/main/tools/details/Qualcomm_AI_Runtime_SDK) and the ONNX Runtime SDK from [https://github.com/microsoft/onnxruntime/releases](https://github.com/microsoft/onnxruntime/releases).

### QNN Execution Provider Version Requirements

ONNX Runtime QNN EP has been built and tested with the following SDK version combinations on Windows:
| QNN EP Version | QAIRT SDK Version | ONNX Runtime Version |
|----------------|-------------------|----------------------|
| v2.0.0         | v2.42.0           | v1.24.1              |

> **Note**: ONNX Runtime QNN EP is built and tested by using the arm64 ONNX Runtime SDK (ex: onnxruntime-win-arm64-1.24.0.zip).

## Build (Windows)
For build instructions, please see the [BUILD page](./build.md).

## Pre-built Packages (Windows Only)
- [NuGet package](https://www.nuget.org/packages/Qualcomm.ML.OnnxRuntime.QNN)
  - **Note**: The NuGet package only supports Windows ARM64 platform
- [Python package](https://pypi.org/project/onnxruntime-qnn/)
  - Requirements:
    - Windows ARM64 (for inferencing on local device with Qualcomm NPU)
    - Windows X64 (for quantizing models. see [Generating a quantized model](./QNN-ExecutionProvider.md#generating-a-quantized-model-x64-only))
    - Python 3.11.x
    - Numpy 1.25.2 or >= 1.26.4
  - Install: `pip install onnxruntime-qnn`

## Qualcomm AI Hub
Qualcomm AI Hub can be used to optimize and run models on Qualcomm hosted devices.
OnnxRuntime QNN Execution Provider is a supported runtime in [Qualcomm AI Hub](https://aihub.qualcomm.com/)

## Configuration Options
The QNN Execution Provider supports a number of configuration options. These provider options are specified as key-value string pairs.
### EP Provider Options

|`"backend_type"`|Description|
|---|-----|
|'cpu'|Enable CPU backend. Useful for integration testing. The CPU backend is a reference implementation of QNN operators.|
|'gpu'|Enable GPU backend.|
|'htp'|Enable HTP backend. Offloads compute to NPU. Default.|
|'saver'|Enable Saver backend.|
|'ir'|Enable IR backend.|

|`"backend_path"`|Description|
|---|-----|
|'libQnnCpu.so' or 'QnnCpu.dll'|Enable CPU backend. See `backend_type` 'cpu'.|
|'libQnnHtp.so' or 'QnnHtp.dll'|Enable HTP backend. See `backend_type` 'htp'.|
|'libQnnGpu.so' or 'QnnGpu.dll'|Enable GPU backend. See `backend_type` 'gpu'.|
|'libQnnSaver.so' or 'QnnSaver.dll'|Enable Saver backend. See `backend_type` 'saver'.|
|'libQnnIr.so' or 'QnnIr.dll'|Enable IR backend. See `backend_type` 'ir'.|

**Note:** `backend_path` is an alternative to `backend_type`. At most one of the two should be specified.
`backend_path` requires a platform-specific path (e.g., `libQnnCpu.so` vs. `QnnCpu.dll`) but also allows one to specify an arbitrary path.

|`"ep_select_backend_path"`|Description|
|---|-----|
|Backend library path (string)|Alternative way to specify the backend path. Same as `backend_path`.|

|`"profiling_level"`|Description|
|---|---|
|'off'|Default. Profiling disabled.|
|'basic'|Basic profiling enabled.|
|'detailed'|Detailed profiling enabled.|
|'optrace'|Optrace profiling enabled. Requires QAIRT 2.39 or later.|

|`"profiling_file_path"`|Description|
|---|---|
|File path (string)|Specify the CSV file path to dump the QNN profiling events.|

See [profiling-tools](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html) for more info on profiling
Alternatively to setting profiling_level at compile time, profiling can be enabled dynamically with ETW (Windows). See [tracing](https://onnxruntime.ai/docs/performance/tune-performance/logging_tracing.html) for more details

|`"rpc_control_latency"`|Description|
|---|---|
|Microseconds (string)|Allows client to set up RPC control latency in microseconds. Defaults to "0" (not set).|

|`"vtcm_mb"`|Description|
|---|---|
|Size in MB (string)|QNN VTCM size in MB. Defaults to "0" (not set). Must be > 0 to take effect.|

|`"enable_vtcm_backup_buffer_sharing"`|Description|
|---|---|
|'0'|Default. Disabled.|
|'1'|Enable VTCM backup buffer sharing across sessions. Requires QNN API version >= 2.26. Conflicts with `ep.context_embed_mode`.|

|`"htp_performance_mode"`|Description|
|---|---|
|'burst'|Burst performance mode.|
|'balanced'|Balanced performance mode.|
|'default'|Default performance mode.|
|'high_performance'|High performance mode.|
|'high_power_saver'|High power saver mode.|
|'low_balanced'|Low balanced mode.|
|'low_power_saver'|Low power saver mode.|
|'power_saver'|Power saver mode.|
|'extreme_power_saver'|Extreme power saver mode.|
|'sustained_high_performance'|Sustained high performance mode.|

|`"qnn_saver_path"`|Description|
|---|---|
|File path (string)|File path to the QNN Saver backend library ('QnnSaver.dll' or 'libQnnSaver.so'). Dumps QNN API calls to disk for replay/debugging.|

|`"qnn_context_priority"`|[Description](https://docs.qualcomm.com/doc/80-63442-10/topic/htp_yielding.html)|
|---|---|
|'low'|Low priority.|
|'normal'|Normal priority. Default.|
|'normal_high'|Normal high priority.|
|'high'|High priority.|

|`"htp_graph_finalization_optimization_mode"`|Description|
|---|---|
|'0'|Default optimization mode.|
|'1'|Faster preparation time, less optimal graph.|
|'2'|Longer preparation time, more optimal graph.|
|'3'|Longest preparation time, most optimal graph.|

|`"soc_model"`|Description|
|---|---|
|Model number (string)|The SoC model number. Refer to the [QAIRT SDK documentation](https://docs.qualcomm.com/doc/80-63442-10/topic/QNN_general_overview.html#supported-snapdragon-devices) for valid values. Defaults to "0" (unknown).|

|`"htp_arch"`|Description|
|---|---|
|Hardware architecture (string)|HTP Architecture number. Refer to the [QAIRT SDK documentation](https://docs.qualcomm.com/doc/80-63442-10/topic/enum_QnnHtpDevice_8h_1a0ed976142af98a86143459dfd326f717.html) for valid values. Defaults to "0" (none).|

|`"device_id"`|Description|
|---|---|
|Device ID (string)|The ID of the device to use when setting `htp_arch`. Defaults to "0" (for single device). Must be >= 0.|

|`"enable_htp_fp16_precision"`|Description [Example](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx/QNN_EP/mobilenetv2_classification)|
|---|---|
|'0'|Disabled. Inference with fp32 precision if it's fp32 model.|
|'1'|Default. Enable the float32 model to be inferenced with fp16 precision.|

|`"enable_htp_spill_fill_buffer"`|Description|
|---|---|
|'0'|Default. Disabled.|
|'1'|Enable HTP spill fill buffer for context cache.|

|`"offload_graph_io_quantization"`|Description|
|---|---|
|'0'|Disabled. QNN EP will handle quantization and dequantization of graph I/O.|
|'1'|Default. Enabled. Offload quantization and dequantization of graph I/O to CPU EP.|

|`"enable_htp_shared_memory_allocator"`|Description|
|---|---|
|'0'|Default. Disabled.|
|'1'|Enable the QNN HTP shared memory allocator. Requires libcdsprpc.so/dll to be available. [Code example](https://github.com/microsoft/onnxruntime/blob/544bdd60730270f49f6a5baafdff54065f626776/onnxruntime/test/shared_lib/test_inference.cc#L2262-L2354)|

|`"op_packages"`|Description|
|---|---|
|Op package config (string)|Register custom op packages. Format: `<OpType>:<PackagePath>:<InterfaceSymbolName>[:<Target>]`. Multiple packages can be separated by commas.|

|`"dump_json_qnn_graph"`|Description|
|---|---|
|'0'|Default. Disabled.|
|'1'|Enable dumping QNN graphs to JSON format for debugging.|

|`"json_qnn_graph_dir"`|Description|
|---|---|
|Directory path (string)|Directory path for dumping QNN JSON graphs. Only effective when `dump_json_qnn_graph` is enabled.|

|`"dump_qnn_ir_dlc"`|Description|
|---|---|
|'0'|Default. Disabled.|
|'1'|Enable dumping QNN graphs to DLC (Deep Learning Container) format using IR backend.|

|`"dump_qnn_ir_dlc_dir"`|Description|
|---|---|
|Directory path (string)|Directory path for dumping QNN IR DLC files. Only effective when `dump_qnn_ir_dlc` is enabled.|

|`"qnn_ir_backend_path"`|Description|
|---|---|
|Backend path (string)|Path to the QNN IR backend library. Defaults to 'libQnnIr.so' or 'QnnIr.dll'. Only effective when `dump_qnn_ir_dlc` is enabled.|

|`"skip_qnn_version_check"`|Description|
|---|---|
|'0'|Default. Version check enabled.|
|'1'|Skip QNN API interface version check to use other QNN library versions.|

### Run Options

Run options can be set dynamically at runtime using the ORT Run API. These options allow you to configure QNN EP behavior on a per-inference basis.

|Option Key|Description|
|---|---|
|`"qnn.perf_mode"`|HTP performance mode for this inference run. Valid values: 'burst', 'balanced', 'default', 'high_performance', 'high_power_saver', 'low_balanced', 'low_power_saver', 'power_saver', 'extreme_power_saver', 'sustained_high_performance'. Overrides the EP provider option `htp_performance_mode` for this run.|
|`"qnn.rpc_control_latency"`|RPC control latency in microseconds for this inference run. Overrides the EP provider option `rpc_control_latency` for this run.|
|`"qnn.lora_config"`|LoRAv2 config file path. Format: `<graph name>;<adapter binary section path>`. See **LoRAv2 support** section for details.|

**Example usage (Python):**
```python
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

# Register QNN EP library
ep_lib_path = qnn_ep.get_library_path()
ep_registration_name = "QnnExecutionProvider"
ort.register_execution_provider_library(ep_registration_name, ep_lib_path)

# Select QNN EP device
all_ep_devices = ort.get_ep_devices()
selected_ep_devices = [ep_device for ep_device in all_ep_devices if ep_device.ep_name == ep_registration_name]

# Configure and create session
ep_options = {'backend_path': qnn_ep.get_qnn_htp_path()}
session_options = ort.SessionOptions()
session_options.add_provider_for_devices(selected_ep_devices, ep_options)
session = ort.InferenceSession("model.onnx", sess_options=session_options)

# Set run options for this specific inference
run_options = ort.RunOptions()
run_options.add_run_config_entry("qnn.perf_mode", "burst")
run_options.add_run_config_entry("qnn.rpc_control_latency", "100")

result = session.run(None, {"input": input_data}, run_options)

# Clean up
del session
ort.unregister_execution_provider_library(ep_registration_name)
```

## Supported ONNX operators

|Operator|Notes|
|---|---|
|ai.onnx:Abs||
|ai.onnx:Add||
|ai.onnx:And||
|ai.onnx:ArgMax||
|ai.onnx:ArgMin||
|ai.onnx:Asin||
|ai.onnx:Atan||
|ai.onnx:AveragePool||
|ai.onnx:BatchNormalization|fp16 supported since 1.18.0|
|ai.onnx:Cast||
|ai.onnx:Ceil||
|ai.onnx:Clip|fp16 supported since 1.18.0|
|ai.onnx:Concat||
|ai.onnx:Conv|3d supported since 1.18.0|
|ai.onnx:ConvTranspose|3d supported since 1.18.0|
|ai.onnx:Cos||
|ai.onnx:CumSum||
|ai.onnx:DepthToSpace||
|ai.onnx:DequantizeLinear||
|ai.onnx:Div||
|ai.onnx:Einsum||
|ai.onnx:Elu||
|ai.onnx:Equal||
|ai.onnx:Exp||
|ai.onnx:Expand||
|ai.onnx:Flatten||
|ai.onnx:Floor||
|ai.onnx:Gather|Only supports positive indices|
|ai.onnx:GatherElements||
|ai.onnx:GatherND||
|ai.onnx:Gelu||
|ai.onnx:Gemm||
|ai.onnx:GlobalAveragePool||
|ai.onnx:GlobalMaxPool||
|ai.onnx:Greater||
|ai.onnx:GreaterOrEqual||
|ai.onnx:GridSample||
|ai.onnx:HardSigmoid||
|ai.onnx:HardSwish||
|ai.onnx:InstanceNormalization||
|ai.onnx:Inverse||
|ai.onnx:LRN||
|ai.onnx:LSTM||
|ai.onnx:LayerNormalization||
|ai.onnx:LeakyRelu||
|ai.onnx:Less||
|ai.onnx:LessOrEqual||
|ai.onnx:Log||
|ai.onnx:LogSoftmax||
|ai.onnx:LpNormalization|p == 2|
|ai.onnx:MatMul|Supported input data types on HTP backend: (uint8, uint8), (uint8, uint16), (uint16, uint8)|
|ai.onnx:Max||
|ai.onnx:MaxPool||
|ai.onnx:Mean||
|ai.onnx:Min||
|ai.onnx:Mod||
|ai.onnx:Mul||
|ai.onnx:Neg||
|ai.onnx:Not||
|ai.onnx:Or||
|ai.onnx:PRelu|fp16, int32 supported since 1.18.0|
|ai.onnx:Pad||
|ai.onnx:Pow||
|ai.onnx:QuantizeLinear||
|ai.onnx:RandomUniformLike||
|ai.onnx:Reciprocal||
|ai.onnx:ReduceL2||
|ai.onnx:ReduceMax||
|ai.onnx:ReduceMean||
|ai.onnx:ReduceMin||
|ai.onnx:ReduceProd||
|ai.onnx:ReduceSum||
|ai.onnx:Relu||
|ai.onnx:Resize||
|ai.onnx:Round||
|ai.onnx:STFT||
|ai.onnx:ScatterElements||
|ai.onnx:ScatterND||
|ai.onnx:Sigmoid||
|ai.onnx:Sign||
|ai.onnx:Sin||
|ai.onnx:Slice||
|ai.onnx:Softmax||
|ai.onnx:SpaceToDepth||
|ai.onnx:Split||
|ai.onnx:Sqrt||
|ai.onnx:Squeeze||
|ai.onnx:Sub||
|ai.onnx:Sum||
|ai.onnx:Tanh||
|ai.onnx:ThresholdedRelu||
|ai.onnx:Tile||
|ai.onnx:TopK||
|ai.onnx:Transpose||
|ai.onnx:Unsqueeze||
|ai.onnx:Upsample||
|ai.onnx:Where||
|com.microsoft:DequantizeLinear|Provides 16-bit integer dequantization support|
|com.microsoft:Gelu||
|com.microsoft:QuantizeLinear|Provides 16-bit integer quantization support|

Supported data types vary by operator and QNN backend. Refer to the [QAIRT SDK documentation](https://docs.qualcomm.com/doc/80-63442-10/topic/operations.html) for more information.

## Running a model with QNN EP's HTP backend (Python)
<p align="center"><img width="100%" src="../images/qnn_ep_quant_workflow.png" alt="Offline workflow for quantizing an ONNX model for use on QNN EP"/></p>

The QNN HTP backend supports both floating-point models and quantized models. Models with 32-bit floating-point activations and weights can be quantized to use a lower integer precision (e.g., 8-bit or 16-bit integers) to get better performance.

This section provides instructions for quantizing a model and then running the quantized model on QNN EP's HTP backend using Python APIs. Please refer to the [quantization page](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) for a broader overview of quantization concepts.

### Model requirements
QNN EP does not support models with dynamic shapes (e.g., a dynamic batch size). Dynamic shapes must be fixed to a specific value. Refer to the documentation for [making dynamic input shapes fixed](https://onnxruntime.ai/docs/tutorials/mobile/helpers/make-dynamic-shape-fixed.html) for more information.

Additionally, QNN EP supports a subset of ONNX operators (e.g., Loops and Ifs are not supported). Refer to the [list of supported ONNX operators](./QNN-ExecutionProvider.md#supported-onnx-operators).

### Generating a quantized model (x64 only)
The ONNX Runtime python package provides utilities for quantizing ONNX models via the `onnxruntime.quantization` import. The quantization utilities are currently only supported on x86_64 due to issues installing the `onnx` package on ARM64.
Therefore, it is recommended to either use an x64 machine to quantize models or, alternatively, use a separate x64 python installation on Windows ARM64 machines.

Install the ONNX Runtime x64 python package. (please note, you must use x64 package for quantizing the model. use the arm64 package for inferencing and utilizing the HTP/NPU)
```shell
python -m pip install onnxruntime
```

Quantization for QNN EP requires the use of calibration input data. Using a calibration dataset that is representative of typical model inputs is crucial in generating an accurate quantized model.

The following snippet defines a sample `DataReader` class that generates random float32 input data. Note that using random input data will most likely produce an inaccurate quantized model.
Refer to the [implementation of a Resnet data reader](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/resnet50_data_reader.py) for one example of how to create a `CalibrationDataReader` that provides input from image files on disk.

```python
# data_reader.py

import numpy as np
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader


class DataReader(CalibrationDataReader):
    def __init__(self, model_path: str):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        inputs = session.get_inputs()

        self.data_list = []

        # Generate 10 random float32 inputs
        # TODO: Load valid calibration input data for your model
        for _ in range(10):
            input_data = {inp.name : np.random.random(inp.shape).astype(np.float32) for inp in inputs}
            self.data_list.append(input_data)

        self.datasize = len(self.data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                self.data_list
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None

```

The following snippet pre-processes the original model and then quantizes the pre-processed model to use `uint16` activations and `uint8` weights.
Although the quantization utilities expose the `uint8`, `int8`, `uint16`, and `int16` quantization data types, QNN operators typically support the `uint8` and `uint16` data types.
Refer to the [QAIRT SDK operator documentation](https://docs.qualcomm.com/doc/80-63442-10/topic/HtpOpDefSupplement.html) for the data type requirements of each QNN operator.

```python
# quantize_model.py

import data_reader
import numpy as np
import onnx
from onnxruntime.quantization import QuantType, quantize
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model

if __name__ == "__main__":
    input_model_path = "model.onnx"  # TODO: Replace with your actual model
    output_model_path = "model.qdq.onnx"  # Name of final quantized model
    my_data_reader = data_reader.DataReader(input_model_path)

    # Pre-process the original float32 model.
    preproc_model_path = "model.preproc.onnx"
    model_changed = qnn_preprocess_model(input_model_path, preproc_model_path)
    model_to_quantize = preproc_model_path if model_changed else input_model_path

    # Generate a suitable quantization configuration for this model.
    # Note that we're choosing to use uint16 activations and uint8 weights.
    qnn_config = get_qnn_qdq_config(model_to_quantize,
                                    my_data_reader,
                                    activation_type=QuantType.QUInt16,  # uint16 activations
                                    weight_type=QuantType.QUInt8)       # uint8 weights

    # Quantize the model.
    quantize(model_to_quantize, output_model_path, qnn_config)
```

Running `python quantize_model.py` will generate a quantized model called `model.qdq.onnx` that can be run on Windows ARM64 devices via ONNX Runtime's QNN EP.

Refer to the following pages for more information on usage of the quantization utilities:
- [Quantization example for mobilenet on CPU EP](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/quantization/image_classification/cpu)
- [quantization/execution_providers/qnn/preprocess.py](https://github.com/microsoft/onnxruntime/blob/23996bbbbe0406a5c8edbf6b7dbd71e5780d3f4b/onnxruntime/python/tools/quantization/execution_providers/qnn/preprocess.py#L16)
- [quantization/execution_providers/qnn/quant_config.py](https://github.com/microsoft/onnxruntime/blob/23996bbbbe0406a5c8edbf6b7dbd71e5780d3f4b/onnxruntime/python/tools/quantization/execution_providers/qnn/quant_config.py#L20-L27)

### Running a quantized model on Windows ARM64 (onnxruntime-qnn version >= 1.18.0)
Install the ONNX Runtime ARM64 python package for QNN EP (requires Python 3.11.x and Numpy 1.25.2 or >= 1.26.4):
```shell
python -m pip install onnxruntime-qnn
```

The following Python snippet creates an ONNX Runtime session with QNN EP and runs the quantized model `model.qdq.onnx` on the HTP backend.

```python
# run_qdq_model.py

import onnxruntime as ort
import onnxruntime_qnn as qnn_ep
import numpy as np

# Register QNN EP library
# The registration name is used as the EP name for QNN EP
ep_lib_path = qnn_ep.get_library_path()
ep_registration_name = "QnnExecutionProvider"
ort.register_execution_provider_library(ep_registration_name, ep_lib_path)

# Select OrtEpDevice(s) matching the registration name (which is the EP name for QNN EP)
all_ep_devices = ort.get_ep_devices()
selected_ep_devices = [ep_device for ep_device in all_ep_devices if ep_device.ep_name == ep_registration_name]

if len(selected_ep_devices) == 0:
    raise RuntimeError("QNN EP device not found")

# Create session options
options = ort.SessionOptions()

# (Optional) Enable configuration that raises an exception if the model can't be
# run entirely on the QNN HTP backend.
options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

# Configure QNN EP options
ep_options = {"backend_path": qnn_ep.get_qnn_htp_path()}  # Use HTP backend

# Add QNN EP to session
options.add_provider_for_devices(selected_ep_devices, ep_options)

# Create an ONNX Runtime session.
# TODO: Provide the path to your ONNX model
session = ort.InferenceSession("model.qdq.onnx", sess_options=options)

# Run the model with your input.
# TODO: Use numpy to load your actual input from a file or generate random input.
input0 = np.ones((1,3,224,224), dtype=np.float32)
result = session.run(None, {"input": input0})

# Print output.
print(result)

# Clean up
del session

# Unregister the library after all sessions using it have been released
ort.unregister_execution_provider_library(ep_registration_name)
```

Running `python run_qdq_model.py` will execute the quantized `model.qdq.onnx` model on the QNN HTP backend.

Notice that the session has been optionally configured to raise an exception if the entire model cannot be executed on the QNN HTP backend. This is useful for verifying that the quantized model is fully supported by QNN EP.
Available session configurations include:
- [session.disable_cpu_ep_fallback](https://github.com/microsoft/onnxruntime/blob/a4cfdc1c28ac95ec6fd0667e856b6a6b8dd1020c/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h#L229): Disables fallback of unsupported operators to the CPU EP.
- [ep.context_enable](https://github.com/microsoft/onnxruntime/blob/a4cfdc1c28ac95ec6fd0667e856b6a6b8dd1020c/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h#L243): [Enable QNN context cache](./QNN-ExecutionProvider.md#qnn-context-binary-cache-feature) feature to dump a cached version of the model in order to decrease session creation time.

The above snippet only specifies the `backend_path` provider option. Refer to the [Configuration options section](./QNN-ExecutionProvider.md#configuration-options) for a list of all available QNN EP provider options.

## Running a model with QNN EP's GPU backend

The QNN GPU backend can run models with 32-bit/16-bit floating-point activations and weights as such without prior quantization. A 16-bit floating-point model generally can run inference faster on the GPU compared to its 32-bit version. To help reduce the size of large models, quantizing weights to `uint8`, while keeping activations in float is also supported.

Other than the quantized model requirement mentioned in the above HTP backend section, all other requirements are valid for the GPU backend also. So is the model inference sample code except for the portion where you specify the backend.

```python
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

# Register QNN EP library
# The registration name is used as the EP name for QNN EP
ep_lib_path = qnn_ep.get_library_path()
ep_registration_name = "QnnExecutionProvider"
ort.register_execution_provider_library(ep_registration_name, ep_lib_path)

# Select OrtEpDevice(s) matching the registration name (which is the EP name for QNN EP)
all_ep_devices = ort.get_ep_devices()
selected_ep_devices = [ep_device for ep_device in all_ep_devices if ep_device.ep_name == ep_registration_name]

if len(selected_ep_devices) == 0:
    raise RuntimeError("QNN EP device not found")

# Configure QNN EP options for GPU backend
ep_options = {"backend_path": "QnnGpu.dll"}  # Use GPU backend

# Add QNN EP to session
options = ort.SessionOptions()
options.add_provider_for_devices(selected_ep_devices, ep_options)

# Create an ONNX Runtime session.
# TODO: Provide the path to your ONNX model
session = ort.InferenceSession("model.onnx", sess_options=options)

# Use session for inference
# ...

# Clean up
del session

# Unregister the library after all sessions using it have been released
ort.unregister_execution_provider_library(ep_registration_name)
```

## QNN context binary cache feature
There's a QNN context which contains QNN graphs after converting, compiling, finalizing the model. QNN can serialize the context into binary file, so that user can use it for futher inference directly (without the QDQ model) to improve the model loading cost.
The QNN Execution Provider supports a number of session options to configure this.

### Dump QNN context binary
1. Create session option, set "ep.context_enable" to "1" to enable QNN context dump. The key "ep.context_enable" is defined as kOrtSessionOptionEpContextEnable in [onnxruntime_session_options_config_keys.h](https://github.com/microsoft/onnxruntime/blob/8931854528b1b2a3f320d012c78d37186fbbdab8/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h#L239-L252).
2. Create the session with the QDQ model using session options created in step 1, and use HTP backend
A Onnx model with QNN context binary will be created once the session is created/initialized. No need to run the session.
The QNN context binary generation can be done on the QualComm device which has HTP using Arm64 build. It can also be done on x64 machine using x64 build (not able to run it since there's no HTP device).

The generated Onnx model which has QNN context binary can be deployed to production/real device to run inference. This Onnx model is treated as a normal model by QNN Execution Provider. Inference code keeps same as inference with QDQ model on HTP backend.

[Code example](https://github.com/microsoft/onnxruntime-inference-examples/blob/733ce6f3e8dd2ede8b67a8465684bca2f62a4a33/c_cxx/QNN_EP/mobilenetv2_classification/main.cpp#L90-L97)
```
#include "onnxruntime_session_options_config_keys.h"

// C++
Ort::SessionOptions so;
so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");

// C
const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
OrtSessionOptions* session_options;
CheckStatus(g_ort, g_ort->CreateSessionOptions(&session_options));
g_ort->AddSessionConfigEntry(session_options, kOrtSessionOptionEpContextEnable, "1");
```

```python
# Python
import onnxruntime

options = onnxruntime.SessionOptions()
options.add_session_config_entry("ep.context_enable", "1")
```

### Configure the context binary file path
The generated Onnx model with QNN context binary is default to [input_QDQ_model_name]_ctx.onnx in case user does not specify the path. User can to set the path in the session option with the key "ep.context_file_path". Example code below:

```
// C++
so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, "./model_a_ctx.onnx");

// C
g_ort->AddSessionConfigEntry(session_options, kOrtSessionOptionEpContextFilePath, "./model_a_ctx.onnx");
```

```python
# Python
options.add_session_config_entry("ep.context_file_path", "./model_a_ctx.onnx")
```

### Enable the embed mode
The QNN context binary content is not embedded in the generated Onnx model by default. A bin file will be generated separately. The file name looks like [input_model_file_name]_QNN_[hash_id].bin. The name is provided by Ort and tracked in the generated Onnx model. It will cause problems if any changes are made to the bin file. This bin file needs to sit together with the generated Onnx file. User can enable it by setting "ep.context_embed_mode" to "1". In that case the content of the context binary is embedded inside the Onnx model.

```
// C++
so.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "1");

// C
g_ort->AddSessionConfigEntry(session_options, kOrtSessionOptionEpContextEmbedMode, "1");
```

```python
# Python
options.add_session_config_entry("ep.context_embed_mode", "1")
```

## QNN EP Profiling
Profiling data is available with the HTP backend. Enabling QNN profiling will generate a user-readable .csv file that will contain information from initialization, execution, and de-initialization.

If onnxruntime is compiled with a more recent QAIRT SDK (2.39 or later), then a _qnn.log file will also be generated alongside the .csv file. This .log file is parsable by [qnn-profile-viewer](https://docs.qualcomm.com/doc/80-63442-10/topic/general_tools.html#qnn-profile-viewer), which is provided in the SDK.

### General Usage
To utilize QNN profiling, simply set the EP option profiling_level to basic, detailed, or optrace. Additionally, the EP option profiling_file_path must also be set to the output .csv filepath you would like to write data to:
```python
# Python on Windows on Snapdragon device
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep
import numpy as np

# Register QNN EP library
ep_lib_path = qnn_ep.get_library_path()
ep_registration_name = "QnnExecutionProvider"
ort.register_execution_provider_library(ep_registration_name, ep_lib_path)

# Select QNN EP device
all_ep_devices = ort.get_ep_devices()
selected_ep_devices = [ep_device for ep_device in all_ep_devices if ep_device.ep_name == ep_registration_name]

# Configure EP options with profiling
ep_options = {
    "backend_path": "path/to/QnnHtp.dll",  # Use libQnnHtp.so if on Linux
    "htp_performance_mode": "burst",
    "device_id": "0",
    "htp_graph_finalization_optimization_mode": "3",
    "soc_model": "60",
    "htp_arch": "73",
    "vtcm_mb": "8",
    "profiling_level": "basic",
    "profiling_file_path": "output.csv"
}

sess_options = ort.SessionOptions()
sess_options.add_provider_for_devices(selected_ep_devices, ep_options)

session = ort.InferenceSession("model.onnx", sess_options=sess_options)

input0 = np.ones((1,2,3,4), dtype=np.float32)
result = session.run(None, {"input": input0})

# Clean up
del session
ort.unregister_execution_provider_library(ep_registration_name)
```

With the example above, a file "output.csv" will be generated containing the profiling data. Additionally, if using QAIRT 2.39 SDK or later, another file "output_qnn.log" will be generated.

"output_qnn.log" can then be parsed with the appropriate qnn-profile-viewer binary:
```console
> qnn-profile-viewer.exe --input_log .\output_qnn.log --output output_2.csv
```

The above will output basic information, such as the profiling data for the fastest and slowest execution as well as the average case. A .csv file can also be generated this way, too, though the information will likely not differ from the "output.csv".

Additionally, if the profiling_level is set to "detailed" or "optrace", additional data will be shown per-network-layer.

### Optrace-Level Profiling
[Optrace-level profiling](https://docs.qualcomm.com/doc/80-63442-10/topic/htp_backend.html#qnn-htp-profiling) generates a profiling .log file that contains [Qualcomm Hexagon Tensor Processor Analaysis Summary (QHAS)](https://docs.qualcomm.com/doc/80-63442-10/topic/htp_backend.html#qnn-htp-analysis-summary-qhas-) data. This data can be used to generate chrometraces and provide a web browser-friendly UI to visualize data.

**This feature is only available with the QAIRT 2.39 SDK and later.**

### Optrace Setup
To utilize this feature, a context binary must be generated prior to execution:
```python
# Python on Windows on Snapdragon device
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep
import numpy as np

# Register QNN EP library
ep_lib_path = qnn_ep.get_library_path()
ep_registration_name = "QnnExecutionProvider"
ort.register_execution_provider_library(ep_registration_name, ep_lib_path)

# Select QNN EP device
all_ep_devices = ort.get_ep_devices()
selected_ep_devices = [ep_device for ep_device in all_ep_devices if ep_device.ep_name == ep_registration_name]

# Configure EP options with optrace profiling
ep_options = {
    "backend_path": "path/to/QnnHtp.dll",  # Use libQnnHtp.so if on Linux
    "htp_performance_mode": "burst",
    "device_id": "0",
    "htp_graph_finalization_optimization_mode": "3",
    "soc_model": "60",
    "htp_arch": "73",
    "vtcm_mb": "8",
    "profiling_level": "optrace",  # Set profiling_level to optrace
    "profiling_file_path": "optrace.csv"
}

sess_options = ort.SessionOptions()

# Enable context bin generation
sess_options.add_session_config_entry("ep.context_embed_mode", "0")
sess_options.add_session_config_entry("ep.context_enable", "1")

sess_options.add_provider_for_devices(selected_ep_devices, ep_options)

session = ort.InferenceSession("model.onnx", sess_options=sess_options)
```

Upon successful session creation, three files will be generated:
- model_ctx.onnx
- model_qnn.bin
- QNNExecutionProvider_QNN_\<number\>_schematic.bin

model_ctx.onnx is an onnx model with a node that points to the model_qnn.bin context binary, which will be used by the HTP backend for execution. The _schematic.bin file will be used by qnn-profile-viewer to generate QHAS data.

### Generating QHAS Data
Previously for general profiling data, a session was created and executed with "model.onnx". However, now there is a new _ctx.onnx model that utilizes a newly generated context binary. As such, a new inference session must be created with the new _ctx.onnx model:
```python
# Continuing from Optrace Setup:
sess_options = ort.SessionOptions()
sess_options.add_session_config_entry("ep.context_enable", "0")
sess_options.add_provider_for_devices(selected_ep_devices, ep_options)

optrace_session = ort.InferenceSession("model_ctx.onnx", sess_options=sess_options)

input0 = np.ones((1,2,3,4), dtype=np.float32)
result = optrace_session.run(None, {"input": input0})

# Clean up
del optrace_session
ort.unregister_execution_provider_library(ep_registration_name)
```

As before under "General Usage", a .csv file (optrace.csv) and a _qnn.log file (optrace_qnn.log) are generated.

qnn-profile-viewer can be used with different parameters and files to parse all the data written to optrace_qnn.log:
```console
> qnn-profile-viewer.exe --config .\config.json --reader .\QnnHtpOptraceProfilingReader.dll --input_log .\optrace_qnn.log  --schematic .\QNNExecutionProvider_QNN_12345_schematic.bin --output optrace.json
```

Please note:
- Three new files are used:
  - config.json: Please refer to the "Post Process (Chrometrace Generation)" section [on this page](https://docs.qualcomm.com/doc/80-63442-10/topic/htp_backend.html#qnn-htp-optrace-profiling).
  - QnnHtpOptraceProfilingReader.dll: Provided as part of the QAIRT SDK. The corresponding file for Linux is libQnnHtpOptraceProfilingReader.so.
  - QNNExecutionProvider_QNN_12345_schematic.bin: The name will vary. This file must be the same one generated alongside the context binary under "Optrace Setup".
- The output file is now a .json file containing chrometrace data. This .json file can be opened with either [Perfetto Trace Vizualizer](https://ui.perfetto.dev/) or with chrome://tracing.

After running qnn-profile-viewer, you should see a handful of .json files generated with the same prefix as the --output filename parameter. You should also see an .html file generated as well. This .html file can be opened by Chrome to view the chrometrace in a more user-friendly GUI.

### Additional References
For more information how to interpret QHAS data, please refer to [this page](https://docs.qualcomm.com/doc/80-63442-10/topic/htp_backend.html#qnn-htp-analysis-summary-qhas-).

For more information on the data collected with optrace profiling, please refer to [this page](https://docs.qualcomm.com/doc/80-63442-10/topic/htp_backend.html#qnn-htp-optrace-profiling).

## QNN EP weight sharing
Refers to the [EPContext design doc](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html#epcontext-with-weight-sharing)

Note: QNN EP requires **Linux x86_64** or **Windows x86_64** platform.

Additionally, if user creates the QNN context binary (`qnn_ctx.bin`) with weight sharing using the QNN toolchain (`qnn-context-binary-generator`), they can use a script to generate the wrapper Onnx model from the context:  [gen_qnn_ctx_onnx_model.py](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/qnn/gen_qnn_ctx_onnx_model.py). The script creates multiple `model_x_ctx.onnx` files, each containing an `EPContext` node that references the shared `qnn_ctx.bin` file. Each `EPContext` node specifies a unique node name, referring to different Qnn graph from the QNN context.


## Usage

### Using QNN EP as a Plugin EP

QNN EP can be used as a plugin execution provider. This is the recommended approach as it follows the standardized plugin EP pattern described in the [EP Context Design documentation](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html).
> **Note**: QNN EP requires the ep registartion name as "QnnExecutionProvider", and the ep_device name is the same as the registration name.

### C++
C API details are [here](https://onnxruntime.ai/docs/get-started/with-c.html).

```cpp
#include "onnxruntime_cxx_api.h"

const char* lib_registration_name = "QnnExecutionProvider";
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};

// Register QNN EP library with ONNX Runtime
env.RegisterExecutionProviderLibrary(
    lib_registration_name,           // Registration name
    ORT_TSTR("onnxruntime_providers_qnn.dll")  // Path to QNN EP library
);

{
    // Get available EP devices
    std::vector<Ort::ConstEpDevice> ep_devices = env.GetEpDevices();

    // Find the QNN EP device
    std::array<Ort::ConstEpDevice, 1> selected_ep_devices = { nullptr };
    for (Ort::ConstEpDevice ep_device : ep_devices) {
        if (std::strcmp(ep_device.GetName(), lib_registration_name) == 0) {
            selected_ep_devices[0] = ep_device;
            break;
        }
    }

    if (selected_ep_devices[0] == nullptr) {
        throw std::runtime_error("QNN EP device not found");
    }

    // Configure QNN EP options
    Ort::KeyValuePairs ep_options;
    ep_options.Append("backend_path", "QnnHtp.dll");

    // Create session with QNN EP
    Ort::SessionOptions session_options;
    session_options.AppendExecutionProvider_V2(env, selected_ep_devices, ep_options);

    Ort::Session session(env, ORT_TSTR("model.onnx"), session_options);

    // Run model ...
}

// Unregister the library after all sessions using it have been released
env.UnregisterExecutionProviderLibrary(lib_registration_name);
```

### Python
```python
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

# Register QNN EP library
# The registration name is used as the EP name for QNN EP
ep_lib_path = qnn_ep.get_library_path()
ep_registration_name = "QnnExecutionProvider"
ort.register_execution_provider_library(ep_registration_name, ep_lib_path)

# Select OrtEpDevice(s) matching the registration name (which is the EP name for QNN EP)
all_ep_devices = ort.get_ep_devices()
selected_ep_devices = [ep_device for ep_device in all_ep_devices if ep_device.ep_name == ep_registration_name]

if len(selected_ep_devices) == 0:
    raise RuntimeError("QNN EP device not found")

# Configure QNN EP options
ep_options = {'backend_path': qnn_ep.get_qnn_htp_path()}  # Use HTP backend

# Create session options and add QNN EP
session_options = ort.SessionOptions()
session_options.add_provider_for_devices(selected_ep_devices, ep_options)

# Create a session with QNN EP using HTP (NPU) backend
sess = ort.InferenceSession(model_path, sess_options=session_options)

# Use session for inference
# ...

# Clean up
del sess

# Unregister the library after all sessions using it have been released
ort.unregister_execution_provider_library(ep_registration_name)
```

### Inference example

[Image classification with Mobilenetv2 in CPP using QNN Execution Provider with QNN CPU & HTP Backend](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx/QNN_EP/mobilenetv2_classification)


## Error handling
### HTP SubSystem Restart - [SSR](https://docs.qualcomm.com/doc/80-63442-10/topic/htp_backend.html#subsystem-restart-ssr-)
QNN EP returns StatusCode::ENGINE_ERROR regarding QNN HTP SSR issue. Uppper level framework/application should recreate Onnxruntime session if this error detected during session run.


## Add new operator support in QNN EP
To enable new operator support in EP, areas to visit:
- QDQ script support this Op? [code example](https://github.com/microsoft/onnxruntime/pull/14867/files#diff-b1ea073c326fef46054382117c256f106d39bd7c34539d44c6e6d9e9eacc059c)
- Onnxruntime QDQ node unit support this Op? [code example](https://github.com/microsoft/onnxruntime/pull/14867/files#diff-ce0281aaf63e03ecadd592240e41f18742bf8eb095b3725c0e55e589c890946f)
- Is it layout sensitive operator?
  - Registered in LayoutTransformer?
    [code example](https://github.com/microsoft/onnxruntime/blob/6d464748ba7fed2275ecba3a7406298cabc93438/onnxruntime/core/optimizer/transpose_optimizer/transpose_optimizer.cc#L2168)
  - NHWC op schema registered?
    Example error message: <lambda_acc29b18d21b7c13448c4952cd957a60>::operator ()] Model face_det_qdq failed to load:Fatal error: com.ms.internal.nhwc:BatchNormalization(9) is not a registered function/op
    [Example PR](https://github.com/microsoft/onnxruntime/pull/15278)

### Example PRs to enable new operators:
- Non-layout sensitive operator. [Enable Hardsigmoid for QNN EP using SDK support direct support](https://github.com/microsoft/onnxruntime/pull/20956)

- Layout sensitive operator. [Add InstanceNormalization operator to QNN EP](https://github.com/microsoft/onnxruntime/pull/14867)


## Mixed precision support
The following figure demonstrates an example of mixed precision model.
<p align="center"><img width="100%" src="../images/quantization_mixed_precision_1.png" alt="mixed precision model"/></p>
A mixed precision QDQ model consists of regions with different activation/weight quantization data types. The boundary between regions converts between activation quantization data types (e.g., uint8 to uint16) using a DQ to Q sequence.

The ability to specify regions with different quantization data types enables exploring the tradeoffs between accuracy and latency. A higher integer precision may improve accuracy at the expense of latency, so selectively promoting certain regions to a higher precision can aid in achieving a desirable balance in key metrics.

The following figure shows a model with a region that has been promoted to 16-bit from the default 8-bit activation type.
<p align="center"><img width="60%" src="../images/quantization_mixed_precision_2.png" alt="mixed precision layers"/></p>

This model is quantized to uint8 precision, but tensor "Op4_out" is quantized to 16-bit. This can be achieved by specifying the following initial tensor quantization overrides:

```
# Op4_out could be an inaccurate tensor that should be upgraded to 16bit
initial_overrides = {"Op4_out": [{"quant_type": QuantType.QUInt16}]}

qnn_config = get_qnn_qdq_config(
    float_model_path,
    data_reader,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QUInt8,
    init_overrides=initial_overrides,  # These initial overrides will be "fixed"
)
```

The above snippet generates the following "fixed" overrides (get via qnn_config.extra_options["TensorQuantOverrides"]):

```
overrides = {
  “Op2_out”: [{“quant_type”: QUInt8, “convert”: {“quant_type”: QUInt16, “recv_nodes”: {“Op4”}}}],
  “Op3_out”: [{“quant_type”: QUInt8, “convert”: {“quant_type”: QUInt16, “recv_nodes”: {“Op5”}}}],
  “Op4_out”: [{“quant_type”: QUInt16}],
  “Op5_out”: [{“quant_type”: QUInt16, “convert”: {“quant_type”: QUInt8, “recv_nodes”: {“Op6”}}}]
}
```

After the override, the model works like this:

- Op2’s output is consumed by Op4, Op7, and Op8. Op4 consumes the converted u16 type, while Op7 and Op8 consume the original u8 type.
- Op3’s output is converted from u8 to u16. Op5 consumes the converted u16 type.
- Op4’s output is just u16 (not converted).
- Op5’s output is converted from u16 to u8. Op6 consumes the u8 type.

## LoRAv2 support
Currently, only pre-compiled models with EPContext nodes are supported. The  example script for reference [gen_qnn_ctx_onnx_model.py](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/qnn/gen_qnn_ctx_onnx_model.py). After applying the model LoRAv2 using the [QAIRT SDK](https://docs.qualcomm.com/doc/80-63442-10/topic/general_tutorials.html#lora-low-rank-adaptation), a main qnn context binary and several adapter binary sections will be generated. We use the LoRAv2 config and place it into RunOptions for inference.

* The format of the LoRAv2 config:
  * graph name: QNN graph in QNN pre-built context binary.
  * adapter binary section path: binary section generated by qnn-context-binary-generator
```
<graph name>;<adapter binary section path>
```
