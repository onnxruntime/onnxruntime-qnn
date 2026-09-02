# QA End-to-End Test Plan: QNN EP UDO (User-Defined Operation)

**Scope**: Validate the ORT QNN Execution Provider's custom-op (UDO) support end-to-end using the `MyAdd` reference sample (`output = input + constant`).

**Coverage**: QNN CPU (x86), QNN HTP x86 simulator, QNN HTP on-device (arm64), EPContext binary path.

---

## Test Matrix

| ID | Backend | Model | Platform | Key assertion |
|----|---------|-------|----------|---------------|
| [QA-UDO-1](#qa-udo-1-qnn-cpu-x86) | QNN CPU | fp32 | Linux x86_64 | Node assigned to QNN EP; output ≈ input + 2.0 (err < 1e-4) |
| [QA-UDO-2](#qa-udo-2-qnn-htp-x86-simulator) | QNN HTP | QDQ uint8 | Linux x86_64 | DQ→MyAdd→Q fused on HTP; accuracy within QDQ tolerance |
| [QA-UDO-3](#qa-udo-3-qnn-htp-on-device-arm64) | QNN HTP | QDQ uint8 | arm64 device | Same as QA-UDO-2 but on real Hexagon hardware |
| [QA-UDO-4](#qa-udo-4-epc-context-binary-on-device) | QNN HTP + EPContext | QDQ uint8 | arm64 device | Pre-compiled context binary runs with `op_packages` only; no `CustomOpDomain` |

---

## Prerequisites

### Environment (all tests)

| Requirement | Version | How to obtain |
|-------------|---------|---------------|
| QAIRT SDK | any version with `qnn-op-package-generator` | Internal download; set `QNN_SDK_ROOT=<sdk>/qairt/<version>` |
| LLVM | 21.1.8 | `wget https://github.com/llvm/llvm-project/releases/download/llvmorg-21.1.8/LLVM-21.1.8-Linux-X64.tar.xz`; set `LLVM_TOOL_DIR=<path>/LLVM-21.1.8-Linux-X64` |
| Python | **3.10 exactly** | Required by `qnn-op-package-generator` |
| onnx, numpy | recent | `pip install onnx numpy` |
| ORT build | this repo | `onnxruntime_provider_test` and `libonnxruntime.so` must be available |

### Additional for HTP x86 (QA-UDO-2)
- `libQnnHtp.so` on `LD_LIBRARY_PATH` (from QAIRT SDK `lib/x86_64-linux-clang/`)

### Additional for on-device (QA-UDO-3, QA-UDO-4)
- arm64 Linux/Android device accessible via SSH
- Hexagon SDK 6.5.0.0 for arm64 HTP op-package build
- HTP skel signing if device requires it (see [Skel signing](#skel-signing-arm64-htp))
- ORT `onnx_test_runner` binary and `libonnxruntime.so` compiled for arm64

---

## Setup (all tests)

### S-1: Build op packages and models (Linux x86_64)

```bash
cd onnxruntime/test/providers/qnn/udo/sample/

# Generate ONNX models
python3 gen_myadd_model.py --constant 2.0 --outdir .

# Build CPU op package
./build_op_package.sh cpu

# Build HTP op package
./build_op_package.sh htp

# Expected outputs (in udo/ directory):
ls ../libMyAddOpPackage_cpu.so   # must exist
ls ../libMyAddOpPackage_htp.so   # must exist
```

**Pass criterion**: both `.so` files created without errors.

### S-2: (Python only) Build schema companion library

`libMyAddSchema.so` must be compiled against headers matching the installed
`onnxruntime` Python package (same `ORT_API_VERSION`). Use `build_op_package.sh schema`
after setting `ORT_INCLUDE` to the matching headers:

```bash
export ORT_INCLUDE=/path/to/ort-headers-matching-installed-onnxruntime
export ORT_LIB=<ort_build>/linux-x86_64/Release
./build_op_package.sh schema
# Produces: ./libMyAddSchema.so
```

See [sample/README.md](../../onnxruntime/test/providers/qnn/udo/sample/README.md) §"Python sample — version note" for how to determine the correct header version.

---

## QA-UDO-1: QNN CPU x86

**Goal**: Verify MyAdd executes on QNN CPU EP, not the CPU EP fallback.

### C++ path

```bash
ORT_BUILD=<ort_build>/linux-x86_64/Release

g++ -std=c++17 run_udo_sample.cc \
    -I<ort_repo>/include/onnxruntime/core/session \
    -I<ort_repo>/include \
    -L${ORT_BUILD} -lonnxruntime \
    -Wl,-rpath,${ORT_BUILD} \
    -o run_udo_sample

LD_LIBRARY_PATH=${QNN_SDK_ROOT}/lib/x86_64-linux-clang:${ORT_BUILD} \
    ./run_udo_sample cpu myadd_fp32.onnx ../libMyAddOpPackage_cpu.so
```

**Expected stdout**:
```
=== QNN CPU backend ===
Max absolute error vs (input + 2.0): ...
PASS
```

### Python path

```bash
QNN_EP_LIB=<path/to/libonnxruntime_providers_qnn.so>

LD_LIBRARY_PATH=${QNN_SDK_ROOT}/lib/x86_64-linux-clang:${LD_LIBRARY_PATH} \
python3 run_udo_sample.py cpu myadd_fp32.onnx \
    --schema-lib ./libMyAddSchema.so \
    --op-package ../libMyAddOpPackage_cpu.so \
    --qnn-ep-lib ${QNN_EP_LIB}
```

**Expected stdout**:
```
=== QNN CPU backend ===
PASS
```

### EP-assignment verification

Re-run with verbose logging and confirm the MyAdd node is NOT assigned to CPUExecutionProvider:

```bash
LD_LIBRARY_PATH=${QNN_SDK_ROOT}/lib/x86_64-linux-clang:${ORT_BUILD} \
ORT_LOG_LEVEL=1 ./run_udo_sample cpu myadd_fp32.onnx ../libMyAddOpPackage_cpu.so 2>&1 \
    | grep -i "node.*assign\|partition\|MyAdd"
```

Look for log lines indicating `MyAdd` assigned to `QNNExecutionProvider`.

### Pass criteria

- [ ] Exit code 0, "PASS" printed
- [ ] Max absolute error < 1e-4
- [ ] MyAdd node not running on CPUExecutionProvider (check logs)
- [ ] No `ORT error:` or crash

---

## QA-UDO-2: QNN HTP x86 Simulator

**Goal**: Verify the `DQ → MyAdd → Q` fusion executes on HTP simulator with acceptable QDQ accuracy.

### Prerequisite check

```bash
ls ${QNN_SDK_ROOT}/lib/x86_64-linux-clang/libQnnHtp.so   # must exist
export LD_LIBRARY_PATH=${QNN_SDK_ROOT}/lib/x86_64-linux-clang:${LD_LIBRARY_PATH}
```

### C++ path

```bash
LD_LIBRARY_PATH=${QNN_SDK_ROOT}/lib/x86_64-linux-clang:${ORT_BUILD} \
    ./run_udo_sample htp myadd_qdq.onnx ../libMyAddOpPackage_htp.so
```

### Python path

```bash
QNN_EP_LIB=<path/to/libonnxruntime_providers_qnn.so>

LD_LIBRARY_PATH=${QNN_SDK_ROOT}/lib/x86_64-linux-clang:${LD_LIBRARY_PATH} \
python3 run_udo_sample.py htp myadd_qdq.onnx \
    --schema-lib ./libMyAddSchema.so \
    --op-package ../libMyAddOpPackage_htp.so \
    --qnn-ep-lib ${QNN_EP_LIB}
```

**Expected stdout**:
```
=== QNN HTP backend ===
Max absolute error vs (input + 2.0): ...  (QDQ tol: 0.0157)
PASS
```

### Gtest equivalent (cross-check)

The existing gtest covers the same path:
```bash
${ORT_BUILD}/onnxruntime_provider_test \
    --gtest_filter=QnnHTPBackendTests.UDO_Op_MyAdd
```

### Pass criteria

- [ ] Exit code 0, "PASS" printed
- [ ] Max absolute error ≤ `2 × (2.0/255.0)` ≈ 0.0157
- [ ] Gtest `QnnHTPBackendTests.UDO_Op_MyAdd` passes
- [ ] No crash or QNN backend error

---

## QA-UDO-3: QNN HTP On-Device (arm64)

**Goal**: Verify QDQ UDO executes on real Hexagon hardware.

### A: Build arm64 op package (on Linux host)

```bash
# Use Hexagon SDK arm64 toolchain — consult QAIRT SDK docs for exact make target.
# The aarch64 variant follows the same generate → implement → make flow as x86,
# using the android/linux-aarch64 target instead of x86_64-linux-clang.
export QNN_SDK_ROOT=...
export HEXAGON_SDK_ROOT=...

PYTHONPATH=${QNN_SDK_ROOT}/lib/python \
python3 ${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-op-package-generator \
    -p ../MyAddOpPackageHtp.xml -o /tmp/udo_arm64/

cp ../MyAddHTP.cpp /tmp/udo_arm64/MyAddOpPackage/src/ops/MyAdd.cpp
cp ../HTP_Makefile /tmp/udo_arm64/MyAddOpPackage/Makefile

# Build for arm64 (aarch64-android or aarch64-linux per your device target)
make -C /tmp/udo_arm64/MyAddOpPackage android_aarch64
# Output: /tmp/udo_arm64/MyAddOpPackage/libs/aarch64-android/libMyAddOpPackage.so
```

### B: Skel signing (arm64 HTP)

If the device requires skel signing, sign the HTP skel file before deployment.
Refer to the `qairt-skel-signing` skill or QAIRT SDK signing documentation.

### C: Deploy artifacts to device

```bash
DEVICE=<device-ip-or-hostname>
DEVICE_DIR=/data/local/tmp/udo_test

ssh ${DEVICE} "mkdir -p ${DEVICE_DIR}"
scp myadd_qdq.onnx                                              ${DEVICE}:${DEVICE_DIR}/
scp /tmp/udo_arm64/MyAddOpPackage/libs/aarch64-android/libMyAddOpPackage.so \
                                                                ${DEVICE}:${DEVICE_DIR}/
# Also deploy the signed skel if applicable.

# Deploy ORT arm64 binaries (onnx_test_runner, libonnxruntime.so, QNN backend libs):
scp ${ORT_ARM64_BUILD}/onnx_test_runner  ${DEVICE}:${DEVICE_DIR}/
scp ${ORT_ARM64_BUILD}/libonnxruntime.so ${DEVICE}:${DEVICE_DIR}/
scp ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp.so ${DEVICE}:${DEVICE_DIR}/
```

### D: Run on device

```bash
ssh ${DEVICE} "cd ${DEVICE_DIR} && \
  LD_LIBRARY_PATH=${DEVICE_DIR}:${LD_LIBRARY_PATH} \
  ./onnx_test_runner -v -e qnn -j 1 \
    -i 'backend_type|htp op_packages|MyAdd:${DEVICE_DIR}/libMyAddOpPackage.so:MyAddOpPackageInterfaceProvider:CPU offload_graph_io_quantization|0' \
    myadd_qdq.onnx"
```

**Expected**: test passes; output values match reference within QDQ tolerance (`~0.016`).

### Pass criteria

- [ ] `onnx_test_runner` exits 0
- [ ] Inference output matches float reference ± `2 × (2.0/255.0)`
- [ ] No `QNN_BACKEND_ERROR_*` or skel loading errors in log
- [ ] No device crash / SSR

---

## QA-UDO-4: EPContext Binary On-Device

**Goal**: Verify a pre-compiled context binary runs on-device using only `op_packages` (no `OrtCustomOpDomain` needed).

### A: Generate EPContext binary (on-device or cross-compile host)

The context binary can be generated on x86 (if the QNN simulator produces a device-compatible binary) or directly on device.

```bash
# On device or simulator — generate context binary from QDQ model
ssh ${DEVICE} "cd ${DEVICE_DIR} && \
  LD_LIBRARY_PATH=${DEVICE_DIR} \
  ./onnx_test_runner -e qnn -j 1 \
    -i 'backend_type|htp op_packages|MyAdd:${DEVICE_DIR}/libMyAddOpPackage.so:MyAddOpPackageInterfaceProvider:CPU \
        offload_graph_io_quantization|0 \
        qnn_context_cache_enable|1 qnn_context_cache_path|${DEVICE_DIR}/myadd.bin' \
    myadd_qdq.onnx"
# Expected: myadd.bin created on device
```

### B: Confirm context binary is valid

```bash
ssh ${DEVICE} "ls -lh ${DEVICE_DIR}/myadd.bin"
# Must be non-zero size
```

### C: Run from context binary — with op_packages, WITHOUT CustomOpDomain

Create a minimal EPContext ONNX model wrapping the `.bin` (or use
`gen_qnn_ctx_onnx_model.py` from the ORT tools directory) and run it:

```bash
ssh ${DEVICE} "cd ${DEVICE_DIR} && \
  LD_LIBRARY_PATH=${DEVICE_DIR} \
  ./onnx_test_runner -e qnn -j 1 \
    -i 'backend_type|htp op_packages|MyAdd:${DEVICE_DIR}/libMyAddOpPackage.so:MyAddOpPackageInterfaceProvider:CPU \
        offload_graph_io_quantization|0 \
        qnn_context_cache_enable|1 qnn_context_cache_path|${DEVICE_DIR}/myadd.bin' \
    myadd_qdq_ctx.onnx"
```

Key verification: **do NOT pass any CustomOpDomain / schema library**. If the test passes, it confirms the op package alone is sufficient at inference time.

### Pass criteria

- [ ] Context binary (`.bin`) generated successfully
- [ ] Inference from context binary exits 0
- [ ] No `OrtCustomOpDomain` or schema lib required at inference time
- [ ] Output matches reference ± QDQ tolerance
- [ ] No `QNN_BACKEND_ERROR_OP_PACKAGE_*` errors

---

## Troubleshooting

| Error | Likely cause | Fix |
|-------|-------------|-----|
| `QNN_BACKEND_ERROR_OP_PACKAGE_NOT_FOUND` | `.so` path wrong or not accessible | Check `op_packages` path; ensure the file exists and is readable |
| `QNN_BACKEND_ERROR_OP_PACKAGE_IF_PROVIDER_NOT_FOUND` | Wrong interface symbol name | Verify the symbol matches (e.g. `MyAddOpPackageInterfaceProvider`); use `nm -D libMyAddOpPackage.so | grep InterfaceProvider` |
| `QNN_BACKEND_ERROR_OP_PACKAGE_UNSUPPORTED_VERSION` | Package compiled against a different QNN SDK version | Rebuild the op package against the same QAIRT SDK version used at runtime |
| `QNN_BACKEND_ERROR_OP_PACKAGE_DUPLICATE` | Package registered more than once | Ensure the package path/name appears only once in the `op_packages` string |
| `QNN_BACKEND_ERROR_OP_PACKAGE_REGISTRATION_FAILED` | Generic registration failure | Check QNN backend log for details; confirm target string matches backend (e.g. `"CPU"` for HTP op package) |
| Node falls back to CPUExecutionProvider | `op_packages` option not reaching QNN EP, or `backendRegisterOpPackage` failed | Enable verbose logging; confirm `LoadOpPackage succeed` appears before `GetCapability` |
| `example::MyAdd schema not found` (C++) | `Ort::CustomOpDomain` not added to `SessionOptions` | Add domain registration before creating the `Ort::Session` |
| `example::MyAdd schema not found` (Python) | `register_custom_ops_library` not called | Call `so.register_custom_ops_library("libMyAddSchema.so")` before `InferenceSession(...)` |
| HTP skel loading error 14001 | Unsigned skel on signed-process-domain device | Sign the skel; see QAIRT skel-signing documentation |
| HTP on Windows: `qnn-op-package-generator` fails | HTP generator not supported on Windows | Build the HTP op package on Linux; only CPU UDO is buildable on Windows |
| Package name is `qti.aisw` | Conflicts with built-in QNN namespace | Rename the package in the XML `PackageName` attribute (must not be `qti.aisw`) |

---

## Cross-Reference

| Resource | Location |
|---------|---------|
| Reference sample scripts | `onnxruntime/test/providers/qnn/udo/sample/` |
| C++ gtest (CI ground truth) | `onnxruntime/test/providers/qnn/udo_op_test.cc` |
| CMake build pipeline | `cmake/onnxruntime_unittests_udo.cmake` |
| Op package source assets | `onnxruntime/test/providers/qnn/udo/` |
| ORT QNN EP UDO documentation | `docs/execution_providers/QNN-ExecutionProvider.md` §"QNN User-Defined Operation" |
| `op_packages` parser unit tests | `onnxruntime/test/providers/qnn/qnn_basic_test.cc:229` |
