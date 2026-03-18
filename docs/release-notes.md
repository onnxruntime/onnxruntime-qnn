# ONNX Runtime QNN Execution Provider v2.0.0 (Preview)

**v2.0.0 is the first Plugin QNN EP (Preview) release** — a standalone package that brings Qualcomm hardware acceleration to any standard ONNX Runtime installation, with no custom ORT build required.

```
pip install onnxruntime
pip install onnxruntime-qnn==2.0.0
```

---

## What is the Plugin QNN EP?

Starting with v2.0.0, the QNN Execution Provider ships as a **standalone shared library** (`onnxruntime_providers_qnn.dll`) built on the [Execution Provider ABI](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/) introduced in ONNX Runtime 1.24.1. The plugin is registered at runtime and only links against the standard ONNX Runtime shared library — no internal ORT dependencies, no custom builds.

This replaces the previous built-in EP model (distributed as `onnxruntime-qnn` without a version pin) with a decoupled plugin that can be versioned and released independently.

---

## Highlights

### Feature Parity with ORT 1.24 Built-in QNN EP

v2.0.0 delivers the same capabilities as the built-in QNN EP in ONNX Runtime 1.24, re-implemented on the new plugin architecture:

- **97 ONNX operators supported** — [full operator list](docs/execution_providers/QNN-ExecutionProvider.md#supported-onnx-operators)
- **HTP, CPU, and GPU backend support** via [Qualcomm AI Runtime SDK (QAIRT)](https://qpm.qualcomm.com/#/main/tools/details/Qualcomm_AI_Runtime_SDK)
- **Context binary caching** with cross-version compatibility verification
- **Mixed precision inference** — automatic FP32-to-FP16 conversion on HTP
- **QDQ quantization** — per-tensor, per-channel, and low power block quantization
- **Graph optimizations** — operator fusions (Gelu, LPBQ MatMul/Gemm, HardSigmoid, channel shuffle, and more)
- **Weight sharing** across inference sessions
- **HTP performance tuning** — burst/balanced/power-saver modes, RPC latency control
- **Profiling support** — ETW, QNN profiling, and Perfetto trace integration

As a standalone plugin, the QNN EP is no longer tied to ONNX Runtime core release timelines. This enables faster iteration on Qualcomm-specific features and optimizations.

---

## Migration

Migrating from the built-in QNN EP (`onnxruntime-qnn`) to the Plugin QNN EP (`onnxruntime-qnn==2.0.0`) requires changes to both installation and session setup.

| | Built-in QNN EP | Plugin QNN EP (v2.0.0) |
|---|---|---|
| Install | `pip install onnxruntime-qnn` | `pip install onnxruntime` + `pip install onnxruntime-qnn==2.0.0` |
| ORT build | Custom build with QNN | Standard ORT release |
| EP registration | Automatic | Explicit plugin registration via `register_execution_provider_library` |

**Client impact and migration guide:**
- [Plugin EP Usage Guide](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/usage.html) — covers the new registration API and session setup
- [C++ example](docs/execution_providers/QNN-ExecutionProvider.md#c)
- [Python example and more](docs/execution_providers/QNN-ExecutionProvider.md#qnn-execution-provider)

---

## Dependencies

| Component | Version | Notes |
|---|---|---|
| **QAIRT SDK** | 2.42.0 | Other QAIRT SDK versions may also be used |
| **ONNX Runtime** | 1.24.1+ | Compatible with any ORT version supporting the EP ABI |
| **OGA** | 0.13 | [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai) |

### Platform Support

| Package | Windows ARM64 | Windows x64 |
|---|---|---|
| Python Wheel | Inference | AOT compilation |
| NuGet | Inference | — |
| ZIP | Inference | — |

---

## Resources

| Topic | Link |
|---|---|
| QNN EP documentation | [QNN-ExecutionProvider.md](docs/execution_providers/QNN-ExecutionProvider.md) |
| Build from source | [Build Guide](docs/execution_providers/build.md) |
| Development guide | [Development Guide](docs/execution_providers/development.md) |
| Plugin EP overview | [Plugin EP Libraries](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/) |
| Plugin EP usage | [Plugin EP Usage](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/usage.html) |

---

## Contributors

This release includes contributions from the Qualcomm engineering teams.
