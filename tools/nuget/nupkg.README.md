## About

![ONNX Runtime Plugin QNN EP Logo](https://raw.githubusercontent.com/onnxruntime/onnxruntime-qnn/main/docs/images/header.png)

**ONNX Runtime QNN** is a plugin execution provider that brings Qualcomm hardware acceleration to ONNX Runtime — enabling high-performance AI inference on Qualcomm Snapdragon SoCs via the [Qualcomm AI Runtime SDK (QAIRT)](https://qpm.qualcomm.com/#/main/tools/details/Qualcomm_AI_Runtime_SDK).

This repository is maintained by Qualcomm. For the general ONNX Runtime project, visit [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime).

---

## NuGet Packages

### Qualcomm.ML.OnnxRuntime.QNN
  - 64-bit Windows
  - QNN Execution Provider
    - https://github.com/onnxruntime/onnxruntime-qnn/blob/main/docs/execution_providers/QNN-ExecutionProvider.md

---

## What is a Plugin Execution Provider?

ONNX Runtime supports hardware acceleration through **Execution Providers (EPs)**. The QNN EP is a *plugin* EP — a separately distributed shared library that plugs into a standard ONNX Runtime installation at runtime, without requiring a custom ORT build.

> **QNN EP 2.0.0 is the new Plugin QNN EP.** Starting with version 2.0.0, the QNN EP ships as a standalone plugin package that works with any standard ORT installation — no custom build required. [Learn more about Plugin EPs →](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/)

<br/>
<p align="center"><img width="80%" src="https://raw.githubusercontent.com/onnxruntime/onnxruntime-qnn/main/docs/images/PluginEP-final.png" /></p>
<br/>

| | Provider Bridge EP (QNN) | Plugin QNN EP |
|---|---|---|
| Distribution | Bundled with ORT | Separate package |
| ORT build required | Yes | No |
| Install | `dotnet add package Microsoft.ML.OnnxRuntime.QNN` | `dotnet add package Qualcomm.ML.OnnxRuntime.QNN` |

---

## Getting Started with the Plugin QNN EP

The Plugin QNN EP workflow is different from the classic built-in EP. Follow these steps to migrate or get started.

**1. Client impact: Learn about the ONNX Runtime Plugin EP API →** [Plugin EP Usage](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/usage.html)

**2. Plugin QNN EP specific examples:**

- [Nuget example](https://github.com/onnxruntime/onnxruntime-qnn/blob/main/qcom/samples/test_qnnep.cs)

---

## Install

```bash
dotnet add package Qualcomm.ML.OnnxRuntime.QNN
```

**Requirements:**
- Windows ARM64 (for on-device inference with Qualcomm NPU)

---

## Releases

The current release and past releases can be found here: https://github.com/onnxruntime/onnxruntime-qnn/releases.

For details on the general ONNX Runtime roadmap, please visit: https://onnxruntime.ai/roadmap.

---

## Data/Telemetry

Windows distributions of this project may collect usage data and send it to Microsoft to help improve our products and services. See the [privacy statement](Privacy.md) for more details.

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## License

This project is licensed under the [MIT License](LICENSE).
