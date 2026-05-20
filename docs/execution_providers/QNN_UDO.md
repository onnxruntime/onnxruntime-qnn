
# QNN User‑Defined Operation (UDO)

A **User‑Defined Operation (UDO)** allows developers to extend the Qualcomm® Neural Network (QNN) runtimes with custom operators. UDO enables execution of operations that are not natively supported in the default QNN op set, while maintaining compatibility with model conversion, compilation, and runtime execution.

---

## 1. Overview

A UDO lets you define and register custom operations—describing their inputs, outputs, parameters, data types, and backend behavior—so they can run on:

- CPU
- HTP

Once registered, UDOs integrate transparently into model conversion and runtime execution.

### Limitations

- **HTP UDO is not supported on Windows.** `qnn-op-package-generator` and the Hexagon toolchain only target HTP on Linux; the `x86_64-windows-msvc` host platform supports the CPU backend only.
- On Windows, CPU UDO must be generated with the `--gen_cmakelists` flag (see Step 2), which produces a `CMakeLists.txt`-based build instead of the Linux Makefile flow.

| Backend | Linux x86_64 | Windows x86_64 |
|---------|--------------|----------------|
| CPU     | Supported (Makefile) | Supported (CMake via `--gen_cmakelists`) |
| HTP     | Supported (Makefile + Hexagon SDK) | Not supported |

---

## 2. UDO Workflow

### **Step 1: Create a UDO Configuration File**
The configuration defines:
- Operation name
- Inputs / outputs
- Parameter definitions
- Supported data types
- Backend information

Schema references are available in the QNN SDK. You can also
see [MyAddOpPackageCpu.xml](../../onnxruntime/test/providers/qnn/udo/MyAddOpPackageCpu.xml) for CPU backend
and [MyAddOpPackageHtp.xml](../../onnxruntime/test/providers/qnn/udo/MyAddOpPackageHtp.xml) for HTP backend.

---

### **Step 2: Generate the UDO Package**

Use the QNN Op Package Generator:

```bash
qnn-op-package-generator -p <path/to/op.xml> -o <output_dir>
```

The generator creates:
- Package scaffolding
- Interface provider
- Backend‑specific template code

**Windows (CPU only):** the generator must be invoked from a `Developer PowerShell for VS 2022` shell with `python` and the `--gen_cmakelists` flag, which emits a `CMakeLists.txt` instead of a `Makefile`:

```powershell
python qnn-op-package-generator -p <path\to\op.xml> -o <output_dir> --gen_cmakelists
```

HTP UDO generation is not available on Windows (see Limitations).

---

### **Step 3: Implement Custom Operation**

Fill in the generated skeleton:

Your custom logic goes into `src/ops/*.cpp`.
see [MyAddCPU.cpp](../../onnxruntime/test/providers/qnn/udo/MyAddCPU.cpp) for CPU backend
and [MyAddHTP.cpp](../../onnxruntime/test/providers/qnn/udo/MyAddHTP.cpp) for HTP backend.

---

### **Step 4: Compile the UDO Package**

Compile against the QNN SDK and backend toolchains to produce:

- `lib<OpPackage>.so` — Implementation library
- Registration library used by QNN runtime

Note: To enable UDO compilation on CPU, you must prepare clang++:
```
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-18.1.8/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04.tar.xz
tar -xvf clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04.tar.xz
export PATH=$(realpath ./clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04/bin/):$PATH
```
Note: To enable UDO compilation on HTP, you must prepare the Hexagon SDK:
```
wget https://softwarecenter.qualcomm.com/api/download/software/sdks/Hexagon_SDK/Linux/Debian/6.5.0.0/Hexagon_SDK_Linux.zip
unzip Hexagon_SDK_Linux.zip
export HEXAGON_SDK_ROOT=$(realpath Hexagon_SDK/6.5.0.0)
```

After Step 2 generated `<output_dir>/MyAddOpPackage/`, drop the implementation from Step 3 into `src/ops/MyAdd.cpp` and run `make` from inside the package directory.

**CPU backend (`all_x86` target):**

```bash
export QNN_SDK_ROOT=<path-to-qnn-sdk>
make -C <output_dir>/MyAddOpPackage all_x86
# Output: <output_dir>/MyAddOpPackage/libs/x86_64-linux-clang/libMyAddOpPackage.so
```

**HTP backend (`htp_x86` target):**

The HTP `htp_x86` target needs a Makefile that links the LLVM C++ runtime statically; use the
[HTP_Makefile](../../onnxruntime/test/providers/qnn/udo/HTP_Makefile) shipped with this repo
in place of the one produced by the generator.

```bash
export QNN_SDK_ROOT=<path-to-qnn-sdk>
export HEXAGON_SDK_ROOT=<path-to-hexagon-sdk>/6.5.0.0
cp onnxruntime/test/providers/qnn/udo/HTP_Makefile <output_dir>/MyAddOpPackage/Makefile
make -C <output_dir>/MyAddOpPackage htp_x86
# Output: <output_dir>/MyAddOpPackage/build/x86_64-linux-clang/libQnnMyAddOpPackage.so
```

---


### **Step 5: Execute the Model with UDO**

```
./onnx_test_runner -v -e qnn -j 1 -i "backend_path|./libQnnCpu.so op_packages|<op_type>:<op_package_path>:<interface_symbol_name>[:<target>],<op_type2>:<op_package_path2>:<interface_symbol_nam2e>[:<target2>]" <models>
```

For the whole pipeline, refer [udo unit test](../../cmake/onnxruntime_unittests_udo.cmake)

---


## 3. References

- https://docs.qualcomm.com/doc/80-63442-10/topic/tutorial1.html
- https://docs.qualcomm.com/doc/80-63442-10/topic/op_package_gen_example.html

---
