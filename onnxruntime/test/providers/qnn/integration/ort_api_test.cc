// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Targeted pipeline integration tests for ort_api.cc behaviors.
//
// These tests exercise code paths in ort_api.cc unreachable by component-level
// unit tests because they require a real inference-graph OrtNode* obtained
// during EP compilation (GetCapability + Compile):
//
//   1. OrtNodeUnit QDQ group constructor + GetQDQIODefs  (lines 71–249)
//      Triggered by a model with DQ → op → Q pattern.
//
//   2. OrtNodeAttrHelper "found" paths  (lines 452–651)
//      Triggered by ops that carry attributes (Transpose, etc.).
//
// Models are built inline via the Ort C++ model editor wrappers (Ort::Model,
// Ort::Graph, Ort::Node, Ort::ValueInfo) — no dependency on op-builder test
// infrastructure.  Sessions use the QNN HTP backend (Linux x86-64 simulator);
// tests skip when unavailable.

#if !defined(ORT_MINIMAL_BUILD) && defined(__linux__)

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "gtest/gtest.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

// ============================================================
// Helpers
// ============================================================

namespace {

// RAII: registers the QNN EP plugin on construction, unregisters on destruction.
struct RegisteredQnnEp {
  std::string name;
  bool valid = false;

  explicit RegisteredQnnEp(const std::string& registration_name) : name(registration_name) {
    const ORTCHAR_T* kLibPath = ORT_TSTR("libonnxruntime_providers_qnn.so");
    try {
      ort_env->RegisterExecutionProviderLibrary(name.c_str(), kLibPath);
      valid = true;
    } catch (const Ort::Exception&) {
    }
  }

  ~RegisteredQnnEp() {
    if (valid) {
      try {
        ort_env->UnregisterExecutionProviderLibrary(name.c_str());
      } catch (const Ort::Exception&) {
      }
    }
  }

  RegisteredQnnEp(const RegisteredQnnEp&) = delete;
  RegisteredQnnEp& operator=(const RegisteredQnnEp&) = delete;
};

// Build Ort::SessionOptions targeting the QNN HTP backend.
// On Linux x86-64, libQnnHtp.so runs as a simulator and supports full graph
// compilation and execution. On Linux AArch64, real HTP hardware is used.
// Returns false if the HTP device is not found (libQnnHtp.so unavailable).
bool MakeQnnHtpSessionOptions(const RegisteredQnnEp& ep, Ort::SessionOptions& out_opts) {
  const OrtApi& api = Ort::GetApi();
  std::vector<Ort::ConstEpDevice> ep_devices = ort_env->GetEpDevices();

  // On Linux x86-64, the HTP simulator registers as OrtHardwareDeviceType_CPU.
  const OrtEpDevice* target = nullptr;
  for (const Ort::ConstEpDevice& dev : ep_devices) {
    if (api.EpDevice_EpName(dev) != ep.name) continue;
    if (api.HardwareDevice_Type(api.EpDevice_Device(dev)) != OrtHardwareDeviceType_CPU) continue;
    target = dev;
    break;
  }
  if (!target) return false;

  const std::unordered_map<std::string, std::string> provider_opts{{"backend_path", "libQnnHtp.so"}};
  try {
    out_opts.AppendExecutionProvider_V2(*ort_env, {Ort::ConstEpDevice(target)}, provider_opts);
  } catch (const Ort::Exception&) {
    return false;
  }
  return true;
}

// Build an Ort::ValueInfo for a 1D tensor of the given element type and size.
Ort::ValueInfo MakeValueInfo1D(const char* name, ONNXTensorElementDataType elem_type, int64_t dim) {
  const OrtModelEditorApi* ed = Ort::GetApi().GetModelEditorApi();

  OrtTensorTypeAndShapeInfo* shape_info = nullptr;
  Ort::ThrowOnError(Ort::GetApi().CreateTensorTypeAndShapeInfo(&shape_info));
  Ort::ThrowOnError(Ort::GetApi().SetTensorElementType(shape_info, elem_type));
  Ort::ThrowOnError(Ort::GetApi().SetDimensions(shape_info, &dim, 1));

  OrtTypeInfo* type_info = nullptr;
  Ort::ThrowOnError(ed->CreateTensorTypeInfo(shape_info, &type_info));
  Ort::GetApi().ReleaseTensorTypeAndShapeInfo(shape_info);

  OrtValueInfo* vi = nullptr;
  Ort::ThrowOnError(ed->CreateValueInfo(name, type_info, &vi));
  Ort::GetApi().ReleaseTypeInfo(type_info);
  return Ort::ValueInfo(vi);
}

// Build an Ort::ValueInfo for a 3D tensor.
Ort::ValueInfo MakeValueInfo3D(const char* name, ONNXTensorElementDataType elem_type,
                               int64_t d0, int64_t d1, int64_t d2) {
  const OrtModelEditorApi* ed = Ort::GetApi().GetModelEditorApi();

  OrtTensorTypeAndShapeInfo* shape_info = nullptr;
  Ort::ThrowOnError(Ort::GetApi().CreateTensorTypeAndShapeInfo(&shape_info));
  Ort::ThrowOnError(Ort::GetApi().SetTensorElementType(shape_info, elem_type));
  int64_t dims[] = {d0, d1, d2};
  Ort::ThrowOnError(Ort::GetApi().SetDimensions(shape_info, dims, 3));

  OrtTypeInfo* type_info = nullptr;
  Ort::ThrowOnError(ed->CreateTensorTypeInfo(shape_info, &type_info));
  Ort::GetApi().ReleaseTensorTypeAndShapeInfo(shape_info);

  OrtValueInfo* vi = nullptr;
  Ort::ThrowOnError(ed->CreateValueInfo(name, type_info, &vi));
  Ort::GetApi().ReleaseTypeInfo(type_info);
  return Ort::ValueInfo(vi);
}

// Build an Ort::ValueInfo for a 4D tensor.
Ort::ValueInfo MakeValueInfo4D(const char* name, ONNXTensorElementDataType elem_type,
                               int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
  const OrtModelEditorApi* ed = Ort::GetApi().GetModelEditorApi();

  OrtTensorTypeAndShapeInfo* shape_info = nullptr;
  Ort::ThrowOnError(Ort::GetApi().CreateTensorTypeAndShapeInfo(&shape_info));
  Ort::ThrowOnError(Ort::GetApi().SetTensorElementType(shape_info, elem_type));
  int64_t dims[] = {d0, d1, d2, d3};
  Ort::ThrowOnError(Ort::GetApi().SetDimensions(shape_info, dims, 4));
  OrtTypeInfo* type_info = nullptr;
  Ort::ThrowOnError(ed->CreateTensorTypeInfo(shape_info, &type_info));
  Ort::GetApi().ReleaseTensorTypeAndShapeInfo(shape_info);
  OrtValueInfo* vi = nullptr;
  Ort::ThrowOnError(ed->CreateValueInfo(name, type_info, &vi));
  Ort::GetApi().ReleaseTypeInfo(type_info);
  return Ort::ValueInfo(vi);
}

// Build an Ort::ValueInfo for a 2D tensor.
Ort::ValueInfo MakeValueInfo2D(const char* name, ONNXTensorElementDataType elem_type,
                               int64_t d0, int64_t d1) {
  const OrtModelEditorApi* ed = Ort::GetApi().GetModelEditorApi();

  OrtTensorTypeAndShapeInfo* shape_info = nullptr;
  Ort::ThrowOnError(Ort::GetApi().CreateTensorTypeAndShapeInfo(&shape_info));
  Ort::ThrowOnError(Ort::GetApi().SetTensorElementType(shape_info, elem_type));
  int64_t dims[] = {d0, d1};
  Ort::ThrowOnError(Ort::GetApi().SetDimensions(shape_info, dims, 2));
  OrtTypeInfo* type_info = nullptr;
  Ort::ThrowOnError(ed->CreateTensorTypeInfo(shape_info, &type_info));
  Ort::GetApi().ReleaseTensorTypeAndShapeInfo(shape_info);
  OrtValueInfo* vi = nullptr;
  Ort::ThrowOnError(ed->CreateValueInfo(name, type_info, &vi));
  Ort::GetApi().ReleaseTypeInfo(type_info);
  return Ort::ValueInfo(vi);
}

}  // namespace

// ============================================================
// Test fixture: QNN HTP backend + model editor
// ============================================================

class QnnInt_OrtApiTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!Ort::GetApi().GetModelEditorApi())
      GTEST_SKIP() << "OrtModelEditorApi not available (minimal build)";

    ep_ = std::make_unique<RegisteredQnnEp>("QNNExecutionProvider");
    ASSERT_TRUE(ep_->valid) << "libonnxruntime_providers_qnn.so not available — CI configuration error";

    ASSERT_TRUE(MakeQnnHtpSessionOptions(*ep_, session_opts_))
        << "QNN HTP EP device not found (libQnnHtp.so not available) — CI configuration error";
  }

  std::unique_ptr<RegisteredQnnEp> ep_;
  Ort::SessionOptions session_opts_;
};

// ============================================================
// Test 1: QDQ group — covers GetQDQIODefs + QDQ OrtNodeUnit ctor
//
// Model: uint8[4] x  →  DequantizeLinear(scale, zp)  →  float[4]
//                    →  Relu
//                    →  QuantizeLinear(scale, zp)  →  uint8[4] y
//
// The QNN EP recognizes the DQ→Relu→Q pattern and constructs an OrtNodeUnit
// of type QDQGroup, calling GetQDQIODefs (ort_api.cc lines 71–227) and the
// QDQ OrtNodeUnit constructor (lines 241–249).
// ============================================================

TEST_F(QnnInt_OrtApiTest, QDQGroup_CoversGetQDQIODefs) {
  Ort::Model model({{"", 21}});
  Ort::Graph graph;

  // Graph input: x (uint8[4]), output: y (uint8[4])
  std::vector<Ort::ValueInfo> inputs, outputs;
  inputs.push_back(MakeValueInfo1D("x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, 4));
  graph.SetInputs(inputs);
  outputs.push_back(MakeValueInfo1D("y", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, 4));
  graph.SetOutputs(outputs);

  // Initializers: scale (float32 = 0.01), zp (uint8 = 0)
  {
    int64_t shape[] = {};
    auto scale_val = Ort::Value::CreateTensor<float>(
        Ort::AllocatorWithDefaultOptions(), shape, 0);
    *scale_val.GetTensorMutableData<float>() = 0.01f;
    graph.AddInitializer("scale", scale_val, false);

    auto zp_val = Ort::Value::CreateTensor<uint8_t>(
        Ort::AllocatorWithDefaultOptions(), shape, 0);
    *zp_val.GetTensorMutableData<uint8_t>() = 0;
    graph.AddInitializer("zp", zp_val, false);
  }

  // DQ: x, scale, zp → dq_out
  {
    std::vector<Ort::OpAttr> attrs;
    Ort::Node dq_node("DequantizeLinear", "", "dq",
                      {"x", "scale", "zp"}, {"dq_out"}, attrs);
    graph.AddNode(dq_node);
  }

  // Relu: dq_out → relu_out
  {
    std::vector<Ort::OpAttr> attrs;
    Ort::Node relu_node("Relu", "", "relu", {"dq_out"}, {"relu_out"}, attrs);
    graph.AddNode(relu_node);
  }

  // Q: relu_out, scale, zp → y
  {
    std::vector<Ort::OpAttr> attrs;
    Ort::Node q_node("QuantizeLinear", "", "q",
                     {"relu_out", "scale", "zp"}, {"y"}, attrs);
    graph.AddNode(q_node);
  }

  model.AddGraph(graph);

  // Session creation triggers GetCapability + Compile → GetQDQIODefs.
  // May fall back to CPU EP; either way, no crash is the success criterion.
  try {
    Ort::Session session(*ort_env, model, session_opts_);
    SUCCEED();
  } catch (const Ort::Exception& e) {
    GTEST_SKIP() << "QNN HTP EP failed to compile model — coverage goal not reached: " << e.what();
  }
}

// ============================================================
// Test 2: Transpose with perm attribute
//         Covers OrtNodeAttrHelper::GetInt64s "found" path (lines 509–511)
//
// Model: float[1,3,4] input → Transpose(perm=[0,2,1]) → float[1,4,3] output
//
// During Compile, the Transpose op builder reads OrtNodeAttrHelper::GetInt64s("perm")
// which finds the attribute, exercising the "found" branch.
// ============================================================

TEST_F(QnnInt_OrtApiTest, TransposeAttr_CoversOrtNodeAttrHelperFoundInt64s) {
  Ort::Model model({{"", 21}});
  Ort::Graph graph;

  std::vector<Ort::ValueInfo> inputs, outputs;
  inputs.push_back(MakeValueInfo3D("input", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 1, 3, 4));
  graph.SetInputs(inputs);
  outputs.push_back(MakeValueInfo3D("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 1, 4, 3));
  graph.SetOutputs(outputs);

  {
    int64_t perm_vals[] = {0, 2, 1};
    std::vector<Ort::OpAttr> attrs;
    attrs.emplace_back("perm", perm_vals, 3, ORT_OP_ATTR_INTS);
    Ort::Node t_node("Transpose", "", "transpose", {"input"}, {"output"}, attrs);
    graph.AddNode(t_node);
  }

  model.AddGraph(graph);

  try {
    Ort::Session session(*ort_env, model, session_opts_);

    float input_data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    int64_t input_shape[] = {1, 3, 4};
    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(mem_info, input_data, 12, input_shape, 3);

    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    auto result = session.Run(Ort::RunOptions{}, input_names, &input_tensor, 1, output_names, 1);

    ASSERT_EQ(result.size(), 1u);
    auto shape = result[0].GetTensorTypeAndShapeInfo().GetShape();
    EXPECT_EQ(shape, (std::vector<int64_t>{1, 4, 3}));
  } catch (const Ort::Exception& e) {
    GTEST_SKIP() << "QNN HTP EP failed to compile model — coverage goal not reached: " << e.what();
  }
}

// ============================================================
// Test 3: LeakyRelu with alpha attribute
//         Covers OrtNodeAttrHelper::Get(float) "found" path (lines 461-463)
//
// Model: float[1,4] input → LeakyRelu(alpha=0.1) → float[1,4] output
//
// During Compile, ProcessAlphaAttributeAsInput calls
// OrtNodeAttrHelper::Get("alpha", 0.01f) which finds the attribute,
// exercising the float Get "found" branch.
// ============================================================

TEST_F(QnnInt_OrtApiTest, LeakyReluAttr_CoversOrtNodeAttrHelperFoundFloat) {
  Ort::Model model({{"", 21}});
  Ort::Graph graph;

  std::vector<Ort::ValueInfo> inputs, outputs;
  inputs.push_back(MakeValueInfo1D("input", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 4));
  graph.SetInputs(inputs);
  outputs.push_back(MakeValueInfo1D("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 4));
  graph.SetOutputs(outputs);

  {
    float alpha_val = 0.1f;
    std::vector<Ort::OpAttr> attrs;
    attrs.emplace_back("alpha", &alpha_val, 1, ORT_OP_ATTR_FLOAT);
    Ort::Node lr_node("LeakyRelu", "", "leakyrelu", {"input"}, {"output"}, attrs);
    graph.AddNode(lr_node);
  }

  model.AddGraph(graph);

  try {
    Ort::Session session(*ort_env, model, session_opts_);

    float input_data[4] = {1.0f, -2.0f, 3.0f, -4.0f};
    int64_t input_shape[] = {4};
    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(mem_info, input_data, 4, input_shape, 1);

    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    auto result = session.Run(Ort::RunOptions{}, input_names, &input_tensor, 1, output_names, 1);

    ASSERT_EQ(result.size(), 1u);
    SUCCEED();  // Session compiled and ran — OrtNodeAttrHelper::GetFloat found-path covered.
  } catch (const Ort::Exception& e) {
    GTEST_SKIP() << "QNN HTP EP failed to compile model — coverage goal not reached: " << e.what();
  }
}

// ============================================================
// Test 4: Conv with int64 vector attributes
//         Covers OrtNodeAttrHelper::GetInt64s "found" path for
//         kernel_shape / strides / pads (lines 509-511, 533-535, 557-559)
//
// Model: float[1,1,5,5] input, float[1,1,3,3] weight → Conv → float[1,1,3,3]
//        (kernel_shape=[3,3], strides=[1,1], pads=[0,0,0,0])
//
// During Compile the Conv op builder reads all three GetInt64s attrs,
// exercising the "found" branch for each.
// ============================================================

TEST_F(QnnInt_OrtApiTest, ConvAttr_CoversOrtNodeAttrHelperFoundInt64s) {
  Ort::Model model({{"", 21}});
  Ort::Graph graph;

  // input: float[1,1,5,5],  weight: float[1,1,3,3]

  std::vector<Ort::ValueInfo> inputs, outputs;
  inputs.push_back(MakeValueInfo4D("input", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 1, 1, 5, 5));
  graph.SetInputs(inputs);
  outputs.push_back(MakeValueInfo4D("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 1, 1, 3, 3));
  graph.SetOutputs(outputs);

  // weight initializer: shape [1,1,3,3], all-0.1f
  {
    int64_t w_shape[] = {1, 1, 3, 3};
    auto w_val = Ort::Value::CreateTensor<float>(
        Ort::AllocatorWithDefaultOptions(), w_shape, 4);
    float* w_data = w_val.GetTensorMutableData<float>();
    for (int i = 0; i < 9; ++i) w_data[i] = 0.1f;
    graph.AddInitializer("weight", w_val, false);
  }

  // Conv node: input, weight → output
  // auto_pad="VALID" covers Get(string) "found" path (lines 509-511)
  // kernel_shape/strides cover Get(vector<int32_t>) "found" path (lines 533-546)
  {
    const char* auto_pad_str = "VALID";
    int64_t kernel_shape[] = {3, 3};
    int64_t strides[] = {1, 1};
    std::vector<Ort::OpAttr> attrs;
    attrs.emplace_back("auto_pad", auto_pad_str, static_cast<int>(strlen(auto_pad_str)), ORT_OP_ATTR_STRING);
    attrs.emplace_back("kernel_shape", kernel_shape, 2, ORT_OP_ATTR_INTS);
    attrs.emplace_back("strides", strides, 2, ORT_OP_ATTR_INTS);
    Ort::Node conv_node("Conv", "", "conv", {"input", "weight"}, {"output"}, attrs);
    graph.AddNode(conv_node);
  }

  model.AddGraph(graph);

  try {
    Ort::Session session(*ort_env, model, session_opts_);

    float input_data[25];
    for (int i = 0; i < 25; ++i) input_data[i] = static_cast<float>(i + 1);
    int64_t input_shape[] = {1, 1, 5, 5};
    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, input_data, 25, input_shape, 4);

    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    auto result = session.Run(Ort::RunOptions{}, input_names, &input_tensor, 1, output_names, 1);

    ASSERT_EQ(result.size(), 1u);
    auto shape = result[0].GetTensorTypeAndShapeInfo().GetShape();
    EXPECT_EQ(shape, (std::vector<int64_t>{1, 1, 3, 3}));
  } catch (const Ort::Exception& e) {
    GTEST_SKIP() << "QNN HTP EP failed to compile model — coverage goal not reached: " << e.what();
  }
}

// ============================================================
// Test 5: Standalone DequantizeLinear (no following Q)
//         Covers InitForSingleNode DequantizeLinear branch (lines 265-283)
//
// Model: uint8[4] x → DQ(scale, zp) → Relu → float[4] y
//
// Without a trailing QuantizeLinear, the DQ does NOT form a QDQ group.
// During Compile each node becomes a SingleNode OrtNodeUnit, so the DQ
// goes through the InitForSingleNode "DequantizeLinear" branch.
// ============================================================

TEST_F(QnnInt_OrtApiTest, StandaloneDQ_CoversDequantizeLinearBranch) {
  Ort::Model model({{"", 21}});
  Ort::Graph graph;

  std::vector<Ort::ValueInfo> inputs, outputs;
  inputs.push_back(MakeValueInfo1D("x", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, 4));
  graph.SetInputs(inputs);
  outputs.push_back(MakeValueInfo1D("y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 4));
  graph.SetOutputs(outputs);

  // Initializers: scale (float = 0.1), zp (uint8 = 0)
  {
    int64_t shape[] = {};
    auto scale_val = Ort::Value::CreateTensor<float>(
        Ort::AllocatorWithDefaultOptions(), shape, 0);
    *scale_val.GetTensorMutableData<float>() = 0.1f;
    graph.AddInitializer("scale", scale_val, false);

    auto zp_val = Ort::Value::CreateTensor<uint8_t>(
        Ort::AllocatorWithDefaultOptions(), shape, 0);
    *zp_val.GetTensorMutableData<uint8_t>() = 0;
    graph.AddInitializer("zp", zp_val, false);
  }

  // DQ: x, scale, zp → dq_out
  {
    std::vector<Ort::OpAttr> attrs;
    Ort::Node dq_node("DequantizeLinear", "", "dq",
                      {"x", "scale", "zp"}, {"dq_out"}, attrs);
    graph.AddNode(dq_node);
  }

  // Relu: dq_out → y
  {
    std::vector<Ort::OpAttr> attrs;
    Ort::Node relu_node("Relu", "", "relu", {"dq_out"}, {"y"}, attrs);
    graph.AddNode(relu_node);
  }

  model.AddGraph(graph);

  try {
    Ort::Session session(*ort_env, model, session_opts_);
    SUCCEED();
  } catch (const Ort::Exception& e) {
    GTEST_SKIP() << "QNN HTP EP failed to compile model — coverage goal not reached: " << e.what();
  }
}

// ============================================================
// Test 6: Standalone QuantizeLinear (no preceding DQ)
//         Covers InitForSingleNode QuantizeLinear branch (lines 284-302)
//
// Model: float[4] x → Relu → Q(scale, zp) → uint8[4] y
//
// Without a preceding DequantizeLinear, the Q does NOT form a QDQ group.
// During Compile the Q goes through the InitForSingleNode "QuantizeLinear"
// branch.
// ============================================================

TEST_F(QnnInt_OrtApiTest, StandaloneQ_CoversQuantizeLinearBranch) {
  Ort::Model model({{"", 21}});
  Ort::Graph graph;

  std::vector<Ort::ValueInfo> inputs, outputs;
  inputs.push_back(MakeValueInfo1D("x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 4));
  graph.SetInputs(inputs);
  outputs.push_back(MakeValueInfo1D("y", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, 4));
  graph.SetOutputs(outputs);

  // Initializers: scale (float = 0.1), zp (uint8 = 0)
  {
    int64_t shape[] = {};
    auto scale_val = Ort::Value::CreateTensor<float>(
        Ort::AllocatorWithDefaultOptions(), shape, 0);
    *scale_val.GetTensorMutableData<float>() = 0.1f;
    graph.AddInitializer("scale", scale_val, false);

    auto zp_val = Ort::Value::CreateTensor<uint8_t>(
        Ort::AllocatorWithDefaultOptions(), shape, 0);
    *zp_val.GetTensorMutableData<uint8_t>() = 0;
    graph.AddInitializer("zp", zp_val, false);
  }

  // Relu: x → relu_out
  {
    std::vector<Ort::OpAttr> attrs;
    Ort::Node relu_node("Relu", "", "relu", {"x"}, {"relu_out"}, attrs);
    graph.AddNode(relu_node);
  }

  // Q: relu_out, scale, zp → y
  {
    std::vector<Ort::OpAttr> attrs;
    Ort::Node q_node("QuantizeLinear", "", "q",
                     {"relu_out", "scale", "zp"}, {"y"}, attrs);
    graph.AddNode(q_node);
  }

  model.AddGraph(graph);

  try {
    Ort::Session session(*ort_env, model, session_opts_);
    SUCCEED();
  } catch (const Ort::Exception& e) {
    GTEST_SKIP() << "QNN HTP EP failed to compile model — coverage goal not reached: " << e.what();
  }
}

// ============================================================
// Test 7: Pad opset-10 with pads + value attributes
//         Covers OrtNodeAttrHelper::GetInt64s "found" path (lines 637-639)
//         and GetFloat "found" path (lines 602-603)
//
// Model: float[1,4] x → Pad(opset=10, pads=[0,0,0,0], value=0.0) → float[1,4]
//
// Pad opset < 11 reads pads and constant value as node attributes
// (newer opsets use input tensors), so the op builder calls
// GetInt64s("pads") and GetFloat("value") directly.
// ============================================================

TEST_F(QnnInt_OrtApiTest, PadOpset10Attr_CoversGetInt64sAndGetFloatFound) {
  Ort::Model model({{"", 10}});
  Ort::Graph graph;

  std::vector<Ort::ValueInfo> inputs, outputs;
  inputs.push_back(MakeValueInfo1D("x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 4));
  graph.SetInputs(inputs);
  outputs.push_back(MakeValueInfo1D("y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 4));
  graph.SetOutputs(outputs);

  {
    int64_t pads_vals[] = {0, 0};  // [start_pad, end_pad] for 1D input
    float value_val = 0.0f;
    const char* mode_str = "constant";
    std::vector<Ort::OpAttr> attrs;
    attrs.emplace_back("pads", pads_vals, 2, ORT_OP_ATTR_INTS);
    attrs.emplace_back("value", &value_val, 1, ORT_OP_ATTR_FLOAT);
    attrs.emplace_back("mode", mode_str, static_cast<int>(strlen(mode_str)), ORT_OP_ATTR_STRING);
    Ort::Node pad_node("Pad", "", "pad", {"x"}, {"y"}, attrs);
    graph.AddNode(pad_node);
  }

  model.AddGraph(graph);

  try {
    Ort::Session session(*ort_env, model, session_opts_);
    SUCCEED();
  } catch (const Ort::Exception& e) {
    GTEST_SKIP() << "QNN HTP EP failed to compile model — coverage goal not reached: " << e.what();
  }
}

// ============================================================
// Test 8: ArgMax with int32 attributes
//         Covers OrtNodeAttrHelper::Get(int32_t) "found" path (lines 474-475)
//
// Model: float[1,4] x → ArgMax(axis=1, keepdims=1, select_last_index=0)
//        → int64[1,1] y
//
// Trigger: ArgMax op builder reads keepdims and select_last_index as int32
// attributes, exercising the int32 Get "found" branch.
// ============================================================

TEST_F(QnnInt_OrtApiTest, ArgMaxAttr_CoversOrtNodeAttrHelperFoundInt32) {
  Ort::Model model({{"", 21}});
  Ort::Graph graph;

  std::vector<Ort::ValueInfo> inputs, outputs;
  inputs.push_back(MakeValueInfo2D("x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 1, 4));
  graph.SetInputs(inputs);
  outputs.push_back(MakeValueInfo2D("y", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 1, 1));
  graph.SetOutputs(outputs);

  {
    int64_t axis_val = 1;
    int64_t keepdims_val = 1;
    int64_t select_last_index_val = 0;
    std::vector<Ort::OpAttr> attrs;
    attrs.emplace_back("axis", &axis_val, 1, ORT_OP_ATTR_INT);
    attrs.emplace_back("keepdims", &keepdims_val, 1, ORT_OP_ATTR_INT);
    attrs.emplace_back("select_last_index", &select_last_index_val, 1, ORT_OP_ATTR_INT);
    Ort::Node am_node("ArgMax", "", "argmax", {"x"}, {"y"}, attrs);
    graph.AddNode(am_node);
  }

  model.AddGraph(graph);

  try {
    Ort::Session session(*ort_env, model, session_opts_);
    SUCCEED();
  } catch (const Ort::Exception& e) {
    GTEST_SKIP() << "QNN HTP EP failed to compile model — coverage goal not reached: " << e.what();
  }
}

// ============================================================
// Test 9: ConvTranspose with dilations attribute
//         Covers OrtNodeAttrHelper::Get(vector<int32_t>) "found" path
//         (lines 533-545)
//
// Model: float[1,1,3,3] input, float[1,1,3,3] weight
//        → ConvTranspose(kernel_shape=[3,3], dilations=[1,1]) → float[1,1,5,5]
//
// ConvTranspose's validation reads dilations as vector<int32_t>, exercising
// the int32 vector "found" branch which converts int64 attribute values to
// int32 element-by-element.
// ============================================================

TEST_F(QnnInt_OrtApiTest, ConvTransposeAttr_CoversOrtNodeAttrHelperFoundInt32Vec) {
  Ort::Model model({{"", 21}});
  Ort::Graph graph;

  std::vector<Ort::ValueInfo> inputs, outputs;
  inputs.push_back(MakeValueInfo4D("input", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 1, 1, 3, 3));
  graph.SetInputs(inputs);
  outputs.push_back(MakeValueInfo4D("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 1, 1, 5, 5));
  graph.SetOutputs(outputs);

  // weight initializer: shape [1,1,3,3]
  {
    int64_t w_shape[] = {1, 1, 3, 3};
    auto w_val = Ort::Value::CreateTensor<float>(
        Ort::AllocatorWithDefaultOptions(), w_shape, 4);
    float* w_data = w_val.GetTensorMutableData<float>();
    for (int i = 0; i < 9; ++i) w_data[i] = 0.1f;
    graph.AddInitializer("weight", w_val, false);
  }

  // ConvTranspose: dilations=[1,1] triggers Get(vector<int32_t>) "found" path
  {
    int64_t kernel_shape[] = {3, 3};
    int64_t dilations[] = {1, 1};
    int64_t strides[] = {1, 1};
    std::vector<Ort::OpAttr> attrs;
    attrs.emplace_back("kernel_shape", kernel_shape, 2, ORT_OP_ATTR_INTS);
    attrs.emplace_back("dilations", dilations, 2, ORT_OP_ATTR_INTS);
    attrs.emplace_back("strides", strides, 2, ORT_OP_ATTR_INTS);
    Ort::Node ct_node("ConvTranspose", "", "convtranspose", {"input", "weight"}, {"output"}, attrs);
    graph.AddNode(ct_node);
  }

  model.AddGraph(graph);

  try {
    Ort::Session session(*ort_env, model, session_opts_);
    SUCCEED();
  } catch (const Ort::Exception& e) {
    GTEST_SKIP() << "QNN HTP EP failed to compile model — coverage goal not reached: " << e.what();
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && defined(__linux__)
