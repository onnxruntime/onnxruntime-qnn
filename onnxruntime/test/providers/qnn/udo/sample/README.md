# QNN EP UDO Sample: MyAdd

End-to-end reference showing how to run an ONNX model with a QNN
User-Defined Operation (UDO) through the ORT QNN Execution Provider.

**Op**: `MyAdd` (`domain="example"`) — computes `output = input + constant`.
**Backends demonstrated**: QNN CPU (float32), QNN HTP x86 simulator (QDQ uint8), and on-device HTP (arm64).

---

## Directory contents

| File | Purpose |
|------|---------|
| `gen_myadd_model.py` | Generate `myadd_fp32.onnx` (CPU) and `myadd_qdq.onnx` (HTP) |
| `gen_myadd_test_data.py` | Generate `onnx_test_runner` protobuf inputs and references for on-device HTP |
| `run_udo_sample.cc` | C++ standalone sample — CPU and HTP modes |
| `run_udo_sample.py` | Python sample — CPU and HTP modes |
| `build_op_package.sh` | Build `libMyAddOpPackage_cpu.so` / `libMyAddOpPackage_htp.so` |

Op package source assets live one level up in `../`:
`MyAddOpPackageCpu.xml`, `MyAddOpPackageHtp.xml`, `MyAddCPU.cpp`, `MyAddHTP.cpp`, `HTP_Makefile`.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| QAIRT SDK | any version with `qnn-op-package-generator` | Set `QNN_SDK_ROOT=<sdk>/qairt/<version>` |
| LLVM | 21.1.8 | Set `LLVM_TOOL_DIR=<llvm>/LLVM-21.1.8-Linux-X64` |
| Hexagon SDK | 6.5.0.0 | HTP only; set `HEXAGON_SDK_ROOT=<hexagon>/6.5.0.0` |
| Python | **3.12** | Use 3.12 throughout. The op-package generator supports 3.10/3.12, but the Python sample needs a QNN-EP-compatible host ORT (≥ 1.24), and public PyPI `onnxruntime` has no cp310 wheels past 1.23.2 — too old for the plugin. On 3.12, `pip install onnxruntime` gets a compatible release. |
| onnx, numpy | any recent | `pip install onnx numpy` |
| onnxruntime | built from this repo | needed for C++ headers and `libonnxruntime.so`; set `ORT_LIB` |
| onnxruntime (Python) | ≥ 1.24 | In a Python 3.12 venv: `pip install onnxruntime`. QNN EP libraries come from the built `onnxruntime_qnn` wheel (`build/linux-x86_64/Release/dist/`). |

---

## Step 1 — Generate ONNX models

```bash
cd sample/
python3 gen_myadd_model.py --constant 2.0 --outdir .
# Produces: myadd_fp32.onnx, myadd_qdq.onnx
```

---

## Step 2 — Build the QNN op packages

```bash
export QNN_SDK_ROOT=/path/to/qairt/<version>
export LLVM_TOOL_DIR=/path/to/LLVM-21.1.8-Linux-X64
export HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK/6.5.0.0   # HTP only

./build_op_package.sh all
# Produces:
#   ../libMyAddOpPackage_cpu.so   (QNN CPU op package)
#   ../libMyAddOpPackage_htp.so   (QNN HTP op package)
```

Individual targets: `./build_op_package.sh cpu` or `htp`.

---

## Step 3a — Run C++ sample

```bash
# Build (the in-tree build stages public headers under _deps/ort_core-src)
ORT_BUILD=/path/to/ort/build/linux-x86_64/Release
ORT_HEADERS=${ORT_BUILD}/_deps/ort_core-src/include
g++ -std=c++17 run_udo_sample.cc \
    -I${ORT_HEADERS}/onnxruntime/core/session \
    -I${ORT_HEADERS} \
    -L${ORT_BUILD} -lonnxruntime \
    -Wl,-rpath,${ORT_BUILD} \
    -o run_udo_sample

# Set env var so the QNN EP factory registers the custom-op domain automatically
export ORT_QNN_CUSTOM_OP_DOMAINS="example:MyAdd"

# CPU backend (libQnnCpu.so must be on LD_LIBRARY_PATH)
LD_LIBRARY_PATH=${QNN_SDK_ROOT}/lib/x86_64-linux-clang:${ORT_BUILD} \
    ./run_udo_sample cpu myadd_fp32.onnx ../libMyAddOpPackage_cpu.so

# HTP backend (libQnnHtp.so must be on LD_LIBRARY_PATH)
LD_LIBRARY_PATH=${QNN_SDK_ROOT}/lib/x86_64-linux-clang:${ORT_BUILD} \
    ./run_udo_sample htp myadd_qdq.onnx ../libMyAddOpPackage_htp.so
```

Expected output (CPU):
```
=== QNN CPU backend ===
Max absolute error vs (input + 2.0): 0.00e+00
PASS
```

Expected output (HTP):
```
=== QNN HTP backend ===
Max absolute error vs (input + 2.0): ...  (QDQ tol: 0.0314)
PASS
```

The HTP sample permits two output-quantization steps over `[0, 4]`, so its
tolerance is `2 × (4/255) ≈ 0.0314`.

### Host verification cross-checks

Confirm that the CPU model is assigned to QNN rather than falling back to the
CPU EP:

```bash
LD_LIBRARY_PATH=${QNN_SDK_ROOT}/lib/x86_64-linux-clang:${ORT_BUILD} \
ORT_LOG_LEVEL=1 ./run_udo_sample cpu myadd_fp32.onnx ../libMyAddOpPackage_cpu.so 2>&1 \
    | grep -i "node.*assign\|partition\|MyAdd"
```

The corresponding gtests validate the same CPU and HTP paths:

```bash
${ORT_BUILD}/onnxruntime_provider_test \
    --gtest_filter="QnnCPUBackendTests.UDO_Op_MyAdd_AutoDomainOnly"
${ORT_BUILD}/onnxruntime_provider_test \
    --gtest_filter="QnnHTPBackendTests.UDO_Op_MyAdd"
```

---

## Step 3b — Run Python sample

```bash
# Path to the onnxruntime-qnn package directory (ships libonnxruntime_providers_qnn.so
# and all QNN backend libs: libQnnCpu.so, libQnnHtp.so, etc.)
QNN_PKG=$(python3 -c "import onnxruntime_qnn, os; print(os.path.dirname(onnxruntime_qnn.__file__))")
ORT_LIB=$(python3 -c "import onnxruntime, os; print(os.path.join(os.path.dirname(onnxruntime.__file__), 'capi'))")

# Set env var so the QNN EP factory registers the custom-op domain automatically
export ORT_QNN_CUSTOM_OP_DOMAINS="example:MyAdd"

# CPU backend
LD_LIBRARY_PATH=${QNN_PKG}:${ORT_LIB}:${LD_LIBRARY_PATH} \
python3 run_udo_sample.py cpu myadd_fp32.onnx \
    --op-package ../libMyAddOpPackage_cpu.so \
    --qnn-ep-lib ${QNN_PKG}/libonnxruntime_providers_qnn.so

# HTP backend (libQnnHtp.so is bundled in QNN_PKG)
LD_LIBRARY_PATH=${QNN_PKG}:${ORT_LIB}:${LD_LIBRARY_PATH} \
python3 run_udo_sample.py htp myadd_qdq.onnx \
    --op-package ../libMyAddOpPackage_htp.so \
    --qnn-ep-lib ${QNN_PKG}/libonnxruntime_providers_qnn.so
```

---

## Step 4 — On-device HTP (arm64)

`build_op_package.sh` targets the x86 simulator only.  On-device HTP requires
**two** separately-built op-package halves (the aarch64-android registration lib
and the hexagon-v`NN` DSP skel), plus the correct test runner.

### 4a — Build both op-package halves for the device arch

Set `HEXAGON_VER` to an HTP target compatible with the device and QAIRT runtime
(obtain a supported target from the verbose-log line `Setting libnative architecture
to vNN`, or from the SoC datasheet). The QAIRT SDK must ship
`lib/hexagon-v${HEXAGON_VER}`. A runtime may select a higher native architecture
while executing a compatible lower-target package, so the two values need not be
identical.

```bash
cd onnxruntime/test/providers/qnn/udo/
export DEVICE_SERIAL=<adb-serial>          # from `adb devices`
export HEXAGON_VER=75                      # e.g. 75, 79, 81 — must be device/runtime-compatible
export QNN_SDK_ROOT=<qairt-sdk>
export HEXAGON_SDK_ROOT=<hexagon-sdk>/6.5.0.0
export LLVM_TOOL_DIR=<llvm>/LLVM-21.1.8-Linux-X64

BUILD=/tmp/udo_arm64
rm -rf "${BUILD}"
PYTHONPATH=${QNN_SDK_ROOT}/lib/python \
python3 ${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-op-package-generator \
    -p MyAddOpPackageHtp.xml -o "${BUILD}"
cp MyAddHTP.cpp "${BUILD}/MyAddOpPackage/src/ops/MyAdd.cpp"
cp HTP_Makefile "${BUILD}/MyAddOpPackage/Makefile"

env QNN_SDK_ROOT="${QNN_SDK_ROOT}" HEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT}" \
    PATH="${LLVM_TOOL_DIR}/bin:${PATH}" \
    make -C "${BUILD}/MyAddOpPackage" \
    "X86_CXX=${LLVM_TOOL_DIR}/bin/clang++ -stdlib=libc++" \
    htp_aarch64 htp_v${HEXAGON_VER}
# Outputs:
#   ARM lib : ${BUILD}/MyAddOpPackage/libs/aarch64-android/libQnnMyAddOpPackage.so
#   DSP skel: ${BUILD}/MyAddOpPackage/build/hexagon-v${HEXAGON_VER}/libQnnMyAddOpPackage.so
```

See QA-UDO-3 §A in `docs/execution_providers/qa_udo_e2e_test_plan.md` for the
full arch-detection and build details.

### 4b — Sign the DSP skel (if required)

If the device's process domain requires skel signing, sign the DSP skel before
deployment.  Refer to the `qairt-skel-signing` skill or QAIRT SDK signing docs.

### 4c — Deploy artifacts

Deploy the ARM lib at the top level of `${DEVICE_DIR}` and the DSP skel into a
separate `${DSP_DIR}` under the **exact same filename**. The test runner consumes
an onnx_test_runner-style directory (`model.onnx` + `test_data_set_0/`) and treats
subdirectories under that root as test cases, so `${DSP_DIR}` must be separate.

```bash
DEVICE_DIR=/data/local/tmp/udo_test       # model/test-data root consumed by the runner
DSP_DIR=/data/local/tmp/udo_test_dsp      # keep outside DEVICE_DIR
ORT_ARM64_BUILD=/path/to/build/android-aarch64/Release

adb -s ${DEVICE_SERIAL} shell "rm -rf ${DEVICE_DIR} ${DSP_DIR}; mkdir -p ${DEVICE_DIR}/test_data_set_0 ${DSP_DIR}"

# ORT arm64 binaries + libs
adb -s ${DEVICE_SERIAL} push ${ORT_ARM64_BUILD}/onnxruntime_plugin_ep_onnx_test ${DEVICE_DIR}/
adb -s ${DEVICE_SERIAL} push ${ORT_ARM64_BUILD}/libonnxruntime.so               ${DEVICE_DIR}/
adb -s ${DEVICE_SERIAL} push ${ORT_ARM64_BUILD}/libonnxruntime_providers_qnn.so ${DEVICE_DIR}/

# QNN aarch64-android backend libs + device-arch skel/stub
adb -s ${DEVICE_SERIAL} push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp.so              ${DEVICE_DIR}/
adb -s ${DEVICE_SERIAL} push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpPrepare.so       ${DEVICE_DIR}/
adb -s ${DEVICE_SERIAL} push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnSystem.so           ${DEVICE_DIR}/
adb -s ${DEVICE_SERIAL} push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV${HEXAGON_VER}Stub.so  ${DEVICE_DIR}/
adb -s ${DEVICE_SERIAL} push ${QNN_SDK_ROOT}/lib/hexagon-v${HEXAGON_VER}/unsigned/libQnnHtpV${HEXAGON_VER}Skel.so ${DEVICE_DIR}/

# MyAdd op package: ARM lib at top level, DSP skel outside the runner root.
# The runner treats subdirectories under DEVICE_DIR as test cases, so do not use DEVICE_DIR/dsp.
adb -s ${DEVICE_SERIAL} push ${BUILD}/MyAddOpPackage/libs/aarch64-android/libQnnMyAddOpPackage.so          ${DEVICE_DIR}/
adb -s ${DEVICE_SERIAL} push ${BUILD}/MyAddOpPackage/build/hexagon-v${HEXAGON_VER}/libQnnMyAddOpPackage.so ${DSP_DIR}/

# Model + deterministic test data (float reference = input + 2.0)
TC=/tmp/udo_qdq_testcase
rm -rf "${TC}"
mkdir -p "${TC}"
cp myadd_qdq.onnx "${TC}/model.onnx"
python3 gen_myadd_test_data.py --constant 2.0 --outdir "${TC}"
adb -s ${DEVICE_SERIAL} push "${TC}/model.onnx"               ${DEVICE_DIR}/
adb -s ${DEVICE_SERIAL} push "${TC}/test_data_set_0/."        ${DEVICE_DIR}/test_data_set_0/
```

### 4d — Run on device (dual CPU+HTP op-package registration)

On-device HTP requires the op package registered for **both** processors in one
comma-separated `op_packages` string:

- **CPU entry** — absolute path to the aarch64-android lib, target `:CPU`
  (ARM-side graph prepare)
- **HTP entry** — **bare filename** of the DSP skel, target `:HTP`
  (NSP kernel execution; resolved via `ADSP_LIBRARY_PATH`)

> Supplying only one half fails: `:CPU` alone → `INVALID_HANDLE (6001)` at
> execute (no DSP kernel registered); `:HTP` alone → `Could not find an
> implementation for MyAdd` at model load. The skel target must be compatible
> with the device/runtime; an incompatible skel can finalize at the host API but
> fail on the DSP.

```bash
DEVICE_DIR=/data/local/tmp/udo_test
DSP_DIR=/data/local/tmp/udo_test_dsp
adb -s ${DEVICE_SERIAL} shell "cd ${DEVICE_DIR} && \
  LD_LIBRARY_PATH=${DEVICE_DIR} \
  ADSP_LIBRARY_PATH='${DSP_DIR};${DEVICE_DIR};/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp' \
  ORT_QNN_CUSTOM_OP_DOMAINS=example:MyAdd \
  ./onnxruntime_plugin_ep_onnx_test \
    --plugin_ep_libs 'QNNExecutionProvider|${DEVICE_DIR}/libonnxruntime_providers_qnn.so' \
    --plugin_eps 'QNNExecutionProvider' \
    --plugin_ep_options 'backend_type|htp offload_graph_io_quantization|0 op_packages|MyAdd:${DEVICE_DIR}/libQnnMyAddOpPackage.so:MyAddOpPackageInterfaceProvider:CPU,MyAdd:libQnnMyAddOpPackage.so:MyAddOpPackageInterfaceProvider:HTP' \
    -a 0.04 -t 0.02 -j 1 \
    ${DEVICE_DIR}"
```

`ORT_QNN_CUSTOM_OP_DOMAINS=example:MyAdd` registers the ONNX custom-op schema at
factory-load time (same mechanism as Steps 3a/3b).

Expected output: `Succeeded: 1`. The QDQ output is checked against the float
reference with `-a 0.04 -t 0.02`; two quantization steps over `[0, 4]` are
about `0.0314`.

---

## Step 5 — EPContext binary on-device (arm64)

This is QA-UDO-4. It reuses the deployment from Step 4, including both
op-package halves. Context generation requires the custom-op schema; inference
from the generated context model intentionally does not. Keep context artifacts
outside `${DEVICE_DIR}`: the runner discovers every `.onnx` and subdirectory in
that directory as a model/test case.

```bash
# Generate an external context ONNX + .bin. Keep ORT_QNN_CUSTOM_OP_DOMAINS for this source-model run.
CONTEXT_DIR=/data/local/tmp/udo_context
DSP_DIR=/data/local/tmp/udo_test_dsp
adb -s ${DEVICE_SERIAL} shell "rm -rf ${CONTEXT_DIR}; mkdir -p ${CONTEXT_DIR}"
adb -s ${DEVICE_SERIAL} shell "cd ${DEVICE_DIR} && \
  LD_LIBRARY_PATH=${DEVICE_DIR} \
  ADSP_LIBRARY_PATH='${DSP_DIR};${DEVICE_DIR};/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp' \
  ORT_QNN_CUSTOM_OP_DOMAINS=example:MyAdd \
  ./onnxruntime_plugin_ep_onnx_test -b \
    -C 'ep.context_enable|1 ep.context_file_path|${CONTEXT_DIR}/myadd_ctx.onnx' \
    --plugin_ep_libs 'QNNExecutionProvider|${DEVICE_DIR}/libonnxruntime_providers_qnn.so' \
    --plugin_eps 'QNNExecutionProvider' \
    --plugin_ep_options 'backend_type|htp offload_graph_io_quantization|0 op_packages|MyAdd:${DEVICE_DIR}/libQnnMyAddOpPackage.so:MyAddOpPackageInterfaceProvider:CPU,MyAdd:libQnnMyAddOpPackage.so:MyAddOpPackageInterfaceProvider:HTP' \
    -a 0.04 -t 0.02 -j 1 ${DEVICE_DIR}"

adb -s ${DEVICE_SERIAL} shell "ls -lh ${CONTEXT_DIR}/myadd_ctx.onnx ${CONTEXT_DIR}/*.bin"

# Run the context model. Copy both artifacts so its relative .bin reference resolves.
# Do NOT set ORT_QNN_CUSTOM_OP_DOMAINS here.
adb -s ${DEVICE_SERIAL} shell "cp ${CONTEXT_DIR}/myadd_ctx.onnx ${DEVICE_DIR}/model.onnx && cp ${CONTEXT_DIR}/*.bin ${DEVICE_DIR}/"
adb -s ${DEVICE_SERIAL} shell "cd ${DEVICE_DIR} && \
  LD_LIBRARY_PATH=${DEVICE_DIR} \
  ADSP_LIBRARY_PATH='${DSP_DIR};${DEVICE_DIR};/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp' \
  ./onnxruntime_plugin_ep_onnx_test \
    --plugin_ep_libs 'QNNExecutionProvider|${DEVICE_DIR}/libonnxruntime_providers_qnn.so' \
    --plugin_eps 'QNNExecutionProvider' \
    --plugin_ep_options 'backend_type|htp offload_graph_io_quantization|0 op_packages|MyAdd:${DEVICE_DIR}/libQnnMyAddOpPackage.so:MyAddOpPackageInterfaceProvider:CPU,MyAdd:libQnnMyAddOpPackage.so:MyAddOpPackageInterfaceProvider:HTP' \
    -a 0.04 -t 0.02 -j 1 ${DEVICE_DIR}"
```

Expected output: `Succeeded: 1` for both runs. The generated `.onnx` and `.bin`
must be non-empty. The context run still needs both `op_packages` entries, but
must succeed without `ORT_QNN_CUSTOM_OP_DOMAINS`.

---

## Cross-reference

- Unit test (C++ gtest): `onnxruntime/test/providers/qnn/udo_op_test.cc`
- Build automation: `cmake/onnxruntime_unittests_udo.cmake`
- ORT QNN EP documentation: `docs/execution_providers/QNN-ExecutionProvider.md` §"QNN User-Defined Operation"
- QA test plan: `docs/execution_providers/qa_udo_e2e_test_plan.md`
