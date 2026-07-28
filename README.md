<p align="center"><img width="489px" src="docs/images/header.png" /></p>

**ONNX Runtime QNN** is a plugin execution provider that brings Qualcomm hardware acceleration to ONNX Runtime — enabling high-performance AI inference on Qualcomm Snapdragon SoCs via the [Qualcomm AI Runtime SDK (QAIRT)](https://qpm.qualcomm.com/#/main/tools/details/Qualcomm_AI_Runtime_SDK).

This repository is maintained by Qualcomm. For the general ONNX Runtime project, visit [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime).

---

## What is a Plugin Execution Provider?

ONNX Runtime supports hardware acceleration through **Execution Providers (EPs)**. The QNN EP is a *plugin* EP — a separately distributed shared library that plugs into a standard ONNX Runtime installation at runtime, without requiring a custom ORT build.

> **QNN EP 2.4.0 is the Plugin QNN EP.** Starting with version 2.0.0, the QNN EP ships as a standalone plugin package (`onnxruntime-qnn>=2.0.0`) that works with any standard ORT installation — no custom build required. [Learn more about Plugin EPs →](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/)

<br/>
<p align="center"><img width="80%" src="docs/images/PluginEP-final.png" /></p>
<br/>

| | Provider Bridge EP (QNN) | Plugin QNN EP |
|---|---|---|
| Distribution | Bundled with ORT | Separate package |
| ORT build required | Yes | No |
| Install | `pip install onnxruntime-qnn==1.x.x` | `pip install onnxruntime-qnn==`**`2.x.x`** |

---

## Platform Support

| Package | Windows ARM64 | Windows ARM64 (ARM64x) | Windows x86_64 | Linux ARM64 | Linux x86_64 | Android ARM64 |
|---|---|---|---|---|---|---|
| Python Wheel | Inference | — | AOT compilation | Inference | AOT compilation | — |
| NuGet | Inference | Inference | AOT compilation | — | — | — |
| ZIP | Inference | Inference | AOT compilation | — | AOT compilation | — |
| tgz | — | — | — | Inference | — | — |
| Maven | — | — | — | — | — | Inference |

---

## Getting Started with the Plugin QNN EP

The Plugin QNN EP workflow is different from the classic built-in EP. Follow these steps to migrate or get started.

**1. Client impact: Learn about the ONNX Runtime Plugin EP API →** [Plugin EP Usage](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/usage.html)

**2. Plugin QNN EP specific examples:**

- [C++ example →](docs/execution_providers/QNN-ExecutionProvider.md#c)

- Python example:

```python
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

# ORT QNN EP Version
print(qnn_ep.__version__)

# Register QNN EP library
ep_lib_path = qnn_ep.get_library_path()
lib_registration_name = "QNNExecutionProvider"
ort.register_execution_provider_library(lib_registration_name, ep_lib_path)

# Select QNN EP device
all_ep_devices = ort.get_ep_devices()
selected_ep_devices = [ep_device for ep_device in all_ep_devices if ep_device.ep_name == lib_registration_name]

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
ort.unregister_execution_provider_library(lib_registration_name)
```

- [More examples →](docs/execution_providers/QNN-ExecutionProvider.md#qnn-execution-provider)

---

## Install

```bash
pip install onnxruntime==1.24.4
pip install onnxruntime-qnn==2.4.0
```

**Requirements:**
- Windows ARM64 (for on-device inference with Qualcomm NPU)
- Windows X64 (for model quantization and AOT compilation)
- Python 3.11 – 3.14
- Numpy 1.25.2 or >= 1.26.4

### NuGet

- **2.5.0+**: [`Qualcomm.ML.OnnxRuntime.QNN`](https://www.nuget.org/packages/Qualcomm.ML.OnnxRuntime.QNN) — a single package covering Windows ARM64 (ARM64X), ARM64EC, and x64 (`win-arm64`, `win-arm64ec`, `win-arm64x`, `win-x64` RuntimeIdentifiers) with appropriate fallbacks.
- **2.3.0 and 2.4.0**: [`Qualcomm.ML.OnnxRuntime.QNN`](https://www.nuget.org/packages/Qualcomm.ML.OnnxRuntime.QNN) - Windows ARM64 replaced with Windows ARM64 (ARM64x).
- **2.0.0 to 2.2.0**: [`Qualcomm.ML.OnnxRuntime.QNN`](https://www.nuget.org/packages/Qualcomm.ML.OnnxRuntime.QNN) - Windows ARM64 only.

### Linux Wheels and .tgz Files

- **2.4.0+**: Linux x86_64 Wheels and .tgz files available
- **2.3.0+**: Linux x86_64 Wheels available (Preview)
- **2.1.1+**: Linux ARM64 Wheels and .tgz files available
- **2.1.0**: Linux ARM64 preview wheels available
- **2.0.0**: No Linux ARM64 Wheels or .tgz files

---

## Resources

| Topic | Link |
|---|---|
| Full documentation | [QNN Execution Provider](docs/execution_providers/QNN-ExecutionProvider.md) |
| Build from source | [Build Guide](docs/execution_providers/build.md) |
| Development guide | [Development Guide](docs/execution_providers/development.md) |

---

## Releases

The current release and past releases can be found here: https://github.com/onnxruntime/onnxruntime-qnn/releases.

For details on the general ONNX Runtime roadmap, please visit: https://onnxruntime.ai/roadmap.

---

## Contributions and Feedback

We welcome contributions! See the [contribution guidelines](CONTRIBUTING.md).

- Bug reports / feature requests: [GitHub Issues](https://github.com/onnxruntime/onnxruntime-qnn/issues)
- Questions / discussion: [GitHub Discussions](https://github.com/onnxruntime/onnxruntime-qnn/discussions)

## Data/Telemetry

Windows distributions of this project may collect usage data and send it to Microsoft to help improve our products and services. See the [privacy statement](docs/Privacy.md) for more details.

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## License

This project is licensed under the [MIT License](LICENSE).
