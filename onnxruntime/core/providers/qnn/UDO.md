
# QNN User‑Defined Operation (UDO)

A **User‑Defined Operation (UDO)** allows developers to extend the Qualcomm® Neural Network (QNN) runtimes with custom operators. UDO enables execution of operations that are **not natively supported** in the default QNN op set, while maintaining compatibility with model conversion, compilation, and runtime execution.

---

## 1. Overview

A UDO lets you define and register custom operations—describing their inputs, outputs, parameters, data types, and backend behavior—so they can run on:

- CPU
- HTP
- GPU

Once registered, UDOs integrate transparently into model conversion and runtime execution.

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
see [IncrementOpPackageCpu.xml](../../../../onnxruntime/test/providers/qnn/udo/IncrementOpPackageCpu.xml) for CPU backend
and [IncrementOpPackageHtp.xml](../../../../onnxruntime/test/providers/qnn/udo/IncrementOpPackageHtp.xml) for HTP backend.

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

---

### **Step 3: Implement Custom Operation**

Fill in the generated skeleton:

Your custom logic goes into `src/ops/*.cpp`.
see [IncrementCPU.cpp](../../../../onnxruntime/test/providers/qnn/udo/IncrementCPU.cpp) for CPU backend
and [IncrementHTP.cpp](../../../../onnxruntime/test/providers/qnn/udo/IncrementHTP.cpp) for HTP backend.

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
export HEXAGON_SDK_ROOT=$(realpath Hexagon_SDK)
```

---


### **Step 5: Execute the Model with UDO**

```
./onnx_test_runner -v -e qnn -j 1 -i "backend_path|./libQnnCpu.so op_packages|<op_type>:<op_package_path>:<interface_symbol_name>[:<target>],<op_type2>:<op_package_path2>:<interface_symbol_nam2e>[:<target2>]" <models>
```

For the whole pipeline, refer [udo unit test](../../../../cmake/qnn_udo_unittest.cmake)

---


## 3. References

- https://docs.qualcomm.com/doc/80-63442-10/topic/tutorial1.html

---
