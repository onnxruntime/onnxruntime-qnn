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
| `run_udo_sample.cc` | C++ standalone sample — CPU and HTP modes |
| `register_myadd_schema.cc` | Companion lib for Python: provides the `example::MyAdd` schema |
| `run_udo_sample.py` | Python sample — CPU and HTP modes |
| `build_op_package.sh` | Build `libMyAddOpPackage_cpu.so` / `libMyAddOpPackage_htp.so` / `libMyAddSchema.so` |

Op package source assets live one level up in `../`:
`MyAddOpPackageCpu.xml`, `MyAddOpPackageHtp.xml`, `MyAddCPU.cpp`, `MyAddHTP.cpp`, `HTP_Makefile`.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| QAIRT SDK | any version with `qnn-op-package-generator` | Set `QNN_SDK_ROOT=<sdk>/qairt/<version>` |
| LLVM | 21.1.8 | Set `LLVM_TOOL_DIR=<llvm>/LLVM-21.1.8-Linux-X64` |
| Hexagon SDK | 6.5.0.0 | HTP only; set `HEXAGON_SDK_ROOT=<hexagon>/6.5.0.0` |
| Python | 3.10 **exactly** | `qnn-op-package-generator` requires 3.10 |
| onnx, numpy | any recent | `pip install onnx numpy` |
| onnxruntime | built from this repo | needed for C++ headers and `libonnxruntime.so`; set `ORT_INCLUDE` and `ORT_LIB` |
| onnxruntime (Python) | matching API version | See [Python sample — version note](#python-sample--version-note) |

---

## Step 1 — Generate ONNX models

```bash
cd sample/
python3 gen_myadd_model.py --constant 2.0 --outdir .
# Produces: myadd_fp32.onnx, myadd_qdq.onnx
```

---

## Step 2 — Build the QNN op packages and schema lib

```bash
export QNN_SDK_ROOT=/path/to/qairt/<version>
export LLVM_TOOL_DIR=/path/to/LLVM-21.1.8-Linux-X64
export HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK/6.5.0.0   # HTP only
export ORT_INCLUDE=<ort_repo>/include/onnxruntime/core/session
export ORT_LIB=<ort_build>/linux-x86_64/Release

./build_op_package.sh all
# Produces:
#   libMyAddSchema.so             (Python schema companion lib, in sample/)
#   ../libMyAddOpPackage_cpu.so   (QNN CPU op package)
#   ../libMyAddOpPackage_htp.so   (QNN HTP op package)
```

Individual targets: `./build_op_package.sh cpu`, `htp`, or `schema`.

---

## Step 3a — Run C++ sample

```bash
# Build (adjust paths to your ORT build)
ORT_BUILD=/path/to/ort/build/linux-x86_64/Release
g++ -std=c++17 run_udo_sample.cc \
    -I${ORT_BUILD}/../../../include/onnxruntime/core/session \
    -I${ORT_BUILD}/../../../include \
    -L${ORT_BUILD} -lonnxruntime \
    -Wl,-rpath,${ORT_BUILD} \
    -o run_udo_sample

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
Max absolute error vs (input + 2.0): 0.0157  (QDQ tol: 0.0157)
PASS
```

---

## Step 3b — Run Python sample

### Python sample — version note

`libMyAddSchema.so` must be compiled against headers matching the installed
`onnxruntime` Python package. The ORT API version embedded in the headers
must match the runtime library — a mismatch causes a segfault.

The API version of the installed `onnxruntime-qnn` wheel is printed by:

```bash
python3 -c "import onnxruntime_qnn; print(onnxruntime_qnn.qnn_version)"
# Or check its build_and_package_info.py
```

Then find the matching PyPI `onnxruntime` release — `onnxruntime==1.X.Y` corresponds
to `ORT_API_VERSION=X*10+Y` (roughly). Verify with:

```python
import ctypes, struct
lib = ctypes.CDLL("/path/to/libonnxruntime.so.X.Y.Z")
# API v27 → onnxruntime 1.27.x, API v29 → onnxruntime 1.29.x
```

Install the matching wheel:

```bash
# Example: onnxruntime-qnn ships ORT_API_VERSION=29 → use onnxruntime==1.29.0
pip install onnxruntime==1.29.0
```

Then build `libMyAddSchema.so` against headers from that exact version (download
from `https://github.com/microsoft/onnxruntime/tree/vX.Y.Z/include/onnxruntime/core/session`).

The `onnxruntime-qnn` wheel ships `libonnxruntime_providers_qnn.so` and all QNN
backend libs (`libQnnHtp.so`, etc.) in its package directory.

### Build the schema companion library

If you ran `./build_op_package.sh all` in Step 2, `libMyAddSchema.so` is already
built. To build it individually (use headers matching your installed onnxruntime):

```bash
export ORT_INCLUDE=/path/to/ort-headers  # matching installed onnxruntime version
export ORT_LIB=<ort_build>/linux-x86_64/Release
./build_op_package.sh schema
# Produces: ./libMyAddSchema.so
```

### Run

```bash
# Path to the onnxruntime-qnn package directory (ships libonnxruntime_providers_qnn.so
# and all QNN backend libs: libQnnCpu.so, libQnnHtp.so, etc.)
QNN_PKG=$(python3 -c "import onnxruntime_qnn, os; print(os.path.dirname(onnxruntime_qnn.__file__))")
ORT_LIB=$(python3 -c "import onnxruntime, os; print(os.path.join(os.path.dirname(onnxruntime.__file__), 'capi'))")

# CPU backend
LD_LIBRARY_PATH=${QNN_PKG}:${ORT_LIB}:${LD_LIBRARY_PATH} \
python3 run_udo_sample.py cpu myadd_fp32.onnx \
    --schema-lib ./libMyAddSchema.so \
    --op-package ../libMyAddOpPackage_cpu.so \
    --qnn-ep-lib ${QNN_PKG}/libonnxruntime_providers_qnn.so

# HTP backend (libQnnHtp.so is bundled in QNN_PKG)
LD_LIBRARY_PATH=${QNN_PKG}:${ORT_LIB}:${LD_LIBRARY_PATH} \
python3 run_udo_sample.py htp myadd_qdq.onnx \
    --schema-lib ./libMyAddSchema.so \
    --op-package ../libMyAddOpPackage_htp.so \
    --qnn-ep-lib ${QNN_PKG}/libonnxruntime_providers_qnn.so
---

## Step 4 — On-device HTP (arm64)

Building the arm64 op package requires Hexagon SDK's arm64 toolchain (not covered by
`build_op_package.sh`, which targets the x86 simulator only).

1. Cross-compile the HTP op package for `aarch64-android` or `aarch64-linux` using the
   Hexagon SDK arm64 target. See QAIRT SDK docs.
2. Sign the HTP skel if required by the device's process domain.
3. Copy artifacts to device:
   ```bash
   scp libMyAddOpPackage_htp_arm64.so  device:/data/local/tmp/
   scp myadd_qdq.onnx                  device:/data/local/tmp/
   ```
4. Run on device via SSH (adjust paths to match the device's ORT / QNN runtime location):
   ```bash
   ssh device "cd /data/local/tmp && \
     LD_LIBRARY_PATH=/data/local/tmp:$LD_LIBRARY_PATH \
     onnx_test_runner -v -e qnn -j 1 \
       -i 'backend_type|htp op_packages|MyAdd:libMyAddOpPackage_htp_arm64.so:MyAddOpPackageInterfaceProvider:CPU' \
       myadd_qdq.onnx"
   ```

---

## Python schema-lib: why is it needed?

The ONNX model's node lives in the custom domain `"example"`.  ORT needs a registered
schema for that domain **just to load the model** — before any EP selection.

- **C++**: `Ort::CustomOpDomain` registered directly in `SessionOptions`.
- **Python**: no inline C++ kernel registration, so `register_myadd_schema.cc` provides
  a minimal `RegisterCustomOps` entry point loaded via
  `SessionOptions.register_custom_ops_library()`. Its `Compute` is the CPU fallback;
  QNN execution uses the `op_packages` library.

---

## Cross-reference

- Unit test (C++ gtest): `onnxruntime/test/providers/qnn/udo_op_test.cc`
- Build automation: `cmake/onnxruntime_unittests_udo.cmake`
- ORT QNN EP documentation: `docs/execution_providers/QNN-ExecutionProvider.md` §"QNN User-Defined Operation"
- QA test plan: `docs/execution_providers/qa_udo_e2e_test_plan.md`
