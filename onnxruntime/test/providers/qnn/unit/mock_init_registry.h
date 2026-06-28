// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// MockInitializerRegistry — registry-based OrtApi stubs for QDQ unit tests.
//
// Enables tests to register named mock OrtValueInfo* objects with typed scalar
// or tensor data. Stubs read from this registry to satisfy the OrtApi calls
// made by QnnQuantParamsWrapper::Init (UnpackScales / UnpackZeroPoints) and
// QnnModelWrapper::UnpackInitializerData.
//
// Usage:
//   QnnModelWrapperTestContext ctx;
//   auto& reg = g_mock_init_reg;
//   reg.clear();
//   auto scale_vi = reg.AddScalarFloat("scale", 0.1f);
//   auto zp_vi    = reg.AddScalarUint8("zp", 0);
//   SetupMockInitRegistryStubs(ctx);
//   // Pass scale_vi / zp_vi as OrtNodeUnitIODef::QuantParam fields.
//
// The registry and stubs live in an anonymous namespace, so each TU including
// this header gets its own private copy of `g_mock_init_reg`. Tests should
// call `g_mock_init_reg.clear()` at the top of each test to reset state.
//
// Chain-state assumption (re: g_mock_chain_vi)
// --------------------------------------------
// `GetValueInfoTypeInfo` and `ValueInfo_GetInitializerValue` write the most
// recently looked-up VI into `g_mock_chain_vi`; the downstream stubs
// (`GetTensorElementType`, `GetDimensionsCount`, `GetDimensions`,
// `GetTensorData`) read from it. This works because the EP code calls these
// stubs in a strict per-VI sequence — `GetTypeInfo(A) → CastToTensorInfo →
// {GetElementType,GetDimensions,...}` for VI A, then `GetTypeInfo(B) → ...`
// for VI B — without interleaving lookups across multiple VIs between the
// chain head and a chain tail. Verified for SRC3 against `UnpackScales`,
// `UnpackZeroPoints`, `utils::UnpackInitializerData`, and the `CASE_UNPACK`
// macro. If the EP ever introduces interleaved chains, replace this with a
// per-call argument or a per-thread pointer; the silent-error fallback below
// will surface the mismatch loudly when chain state is broken.

#pragma once

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstring>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn/ort_api.h"

#include "test/providers/qnn/unit/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

struct MockInitSpec {
  ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  std::vector<int64_t> dims;  // {} = scalar
  std::vector<uint8_t> raw_bytes;
};

// Global registry.  Tests call clear() at setup.
struct MockInitRegistry {
  // Map from OrtValueInfo* sentinel → spec.
  std::unordered_map<const OrtValueInfo*, MockInitSpec> vi_to_spec;
  // Map from name → OrtValueInfo* (for Graph_GetInitializers / FindInitializer).
  std::unordered_map<std::string, const OrtValueInfo*> name_to_vi;
  // Name reverse-lookup (for GetValueInfoName stub).
  std::unordered_map<const OrtValueInfo*, std::string> vi_to_name;
  // Flat array for Graph_GetInitializers.
  std::vector<const OrtValueInfo*> vi_array;
  // Storage for sentinel ints (one per registered OrtValueInfo).
  std::deque<int> sentinel_storage;

  void clear() {
    vi_to_spec.clear();
    name_to_vi.clear();
    vi_to_name.clear();
    vi_array.clear();
    sentinel_storage.clear();
  }

  const OrtValueInfo* Add(const std::string& name, MockInitSpec spec) {
    sentinel_storage.push_back(0);
    const OrtValueInfo* vi = reinterpret_cast<const OrtValueInfo*>(&sentinel_storage.back());
    vi_to_spec[vi] = std::move(spec);
    name_to_vi[name] = vi;
    vi_to_name[vi] = name;
    vi_array.push_back(vi);
    return vi;
  }

  const OrtValueInfo* AddScalarFloat(const std::string& name, float value) {
    MockInitSpec s;
    s.elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    s.raw_bytes.resize(sizeof(float));
    std::memcpy(s.raw_bytes.data(), &value, sizeof(float));
    return Add(name, std::move(s));
  }
  const OrtValueInfo* AddScalarUint8(const std::string& name, uint8_t value) {
    MockInitSpec s;
    s.elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    s.raw_bytes = {value};
    return Add(name, std::move(s));
  }
  const OrtValueInfo* AddScalarUint16(const std::string& name, uint16_t value) {
    MockInitSpec s;
    s.elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
    s.raw_bytes.resize(sizeof(uint16_t));
    std::memcpy(s.raw_bytes.data(), &value, sizeof(uint16_t));
    return Add(name, std::move(s));
  }
  const OrtValueInfo* AddScalarInt32(const std::string& name, int32_t value) {
    MockInitSpec s;
    s.elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    s.raw_bytes.resize(sizeof(int32_t));
    std::memcpy(s.raw_bytes.data(), &value, sizeof(int32_t));
    return Add(name, std::move(s));
  }

  // Tensor (multi-element) helpers: caller supplies dims and an aligned data span.
  const OrtValueInfo* AddTensorFloat(const std::string& name,
                                     std::vector<int64_t> dims,
                                     const std::vector<float>& data) {
    MockInitSpec s;
    s.elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    s.dims = std::move(dims);
    s.raw_bytes.resize(data.size() * sizeof(float));
    if (!data.empty()) {
      std::memcpy(s.raw_bytes.data(), data.data(), s.raw_bytes.size());
    }
    return Add(name, std::move(s));
  }
  const OrtValueInfo* AddTensorUint8(const std::string& name,
                                     std::vector<int64_t> dims,
                                     const std::vector<uint8_t>& data) {
    MockInitSpec s;
    s.elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    s.dims = std::move(dims);
    s.raw_bytes = data;
    return Add(name, std::move(s));
  }
  const OrtValueInfo* AddTensorInt32(const std::string& name,
                                     std::vector<int64_t> dims,
                                     const std::vector<int32_t>& data) {
    MockInitSpec s;
    s.elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    s.dims = std::move(dims);
    s.raw_bytes.resize(data.size() * sizeof(int32_t));
    if (!data.empty()) {
      std::memcpy(s.raw_bytes.data(), data.data(), s.raw_bytes.size());
    }
    return Add(name, std::move(s));
  }
  // INT4 / UINT4: caller supplies per-element values; helper packs two values
  // per byte (low nibble = even index, high = odd). The EP's CASE_UNPACK_INT4
  // expects packed Int4x2 layout — feeding it 1 byte per element would over-
  // read the buffer when num_dims indicates more than ~ceil(N/2) bytes.
  const OrtValueInfo* AddTensorInt4As8bit(const std::string& name,
                                          std::vector<int64_t> dims,
                                          const std::vector<int8_t>& values) {
    MockInitSpec s;
    s.elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4;
    s.dims = std::move(dims);
    const size_t n = values.size();
    s.raw_bytes.assign((n + 1) / 2, 0);
    for (size_t i = 0; i < n; ++i) {
      const uint8_t nibble = static_cast<uint8_t>(values[i]) & 0x0F;
      s.raw_bytes[i / 2] |= static_cast<uint8_t>(nibble << ((i & 1u) * 4u));
    }
    return Add(name, std::move(s));
  }
  const OrtValueInfo* AddTensorUint4As8bit(const std::string& name,
                                           std::vector<int64_t> dims,
                                           const std::vector<uint8_t>& values) {
    MockInitSpec s;
    s.elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4;
    s.dims = std::move(dims);
    const size_t n = values.size();
    s.raw_bytes.assign((n + 1) / 2, 0);
    for (size_t i = 0; i < n; ++i) {
      const uint8_t nibble = values[i] & 0x0F;
      s.raw_bytes[i / 2] |= static_cast<uint8_t>(nibble << ((i & 1u) * 4u));
    }
    return Add(name, std::move(s));
  }
};

static MockInitRegistry g_mock_init_reg;
// OrtTypeInfo / OrtTensorTypeAndShapeInfo sentinels (single pair, reused).
static int g_mock_type_info_sentinel = 0;
static int g_mock_shape_info_sentinel = 0;
// OrtValue sentinel used by ValueInfo_GetInitializerValue.
static int g_mock_ort_value_sentinel = 0;
// The OrtValueInfo* currently being processed through the stub chain.
// See "Chain-state assumption" in the file header comment.
static const OrtValueInfo* g_mock_chain_vi = nullptr;

// Build a fresh "lookup miss" OrtStatus each call. Returning a non-null
// OrtStatus* makes the EP code path fail loudly instead of silently coercing
// the missing VI's element type / dims to defaults — SRC2 falls back silently,
// which can hide chain-state breakage. We must hand back a freshly allocated
// OrtStatus on every call: the caller (ORT_CXX_RETURN_ON_API_FAIL) wraps it in
// Ort::Status which will Release it — a static singleton would UAF on the
// second miss.
inline OrtStatus* MakeMockChainMissStatus() {
  Ort::Status err(
      "mock_init_registry: g_mock_chain_vi not registered "
      "(chain-state broken or VI never added)",
      ORT_FAIL);
  return err.release();
}

inline OrtStatus* StubMockGraphGetModelPath(const OrtGraph*,
                                            const ORTCHAR_T** out) noexcept {
  static const ORTCHAR_T empty[] = "";
  *out = empty;
  return nullptr;
}
inline OrtStatus* StubMockGetNumInitializers(const OrtGraph*,
                                             size_t* out) noexcept {
  *out = g_mock_init_reg.vi_array.size();
  return nullptr;
}
inline OrtStatus* StubMockGetInitializers(const OrtGraph*,
                                          const OrtValueInfo** out,
                                          size_t count) noexcept {
  for (size_t i = 0; i < count && i < g_mock_init_reg.vi_array.size(); ++i)
    out[i] = g_mock_init_reg.vi_array[i];
  return nullptr;
}
inline OrtStatus* StubMockGetValueInfoName(const OrtValueInfo* vi,
                                           const char** out) noexcept {
  auto it = g_mock_init_reg.vi_to_name.find(vi);
  static const char empty_name[] = "";
  *out = (it != g_mock_init_reg.vi_to_name.end()) ? it->second.c_str() : empty_name;
  return nullptr;
}
inline OrtStatus* StubMockValueInfoIsConstant(const OrtValueInfo*,
                                              bool* out) noexcept {
  *out = true;
  return nullptr;
}
inline OrtStatus* StubMockValueInfoGetExternal(const OrtValueInfo*,
                                               OrtExternalInitializerInfo** out) noexcept {
  *out = nullptr;
  return nullptr;
}
inline OrtStatus* StubMockGetValueInfoTypeInfo(const OrtValueInfo* vi,
                                               const OrtTypeInfo** out) noexcept {
  g_mock_chain_vi = vi;  // record for downstream stubs
  *out = reinterpret_cast<const OrtTypeInfo*>(&g_mock_type_info_sentinel);
  return nullptr;
}
inline OrtStatus* StubMockCastTypeInfoToTensorInfo(const OrtTypeInfo*,
                                                   const OrtTensorTypeAndShapeInfo** out) noexcept {
  *out = reinterpret_cast<const OrtTensorTypeAndShapeInfo*>(&g_mock_shape_info_sentinel);
  return nullptr;
}
inline OrtStatus* StubMockGetTensorElementType(const OrtTensorTypeAndShapeInfo*,
                                               ONNXTensorElementDataType* out) noexcept {
  auto it = g_mock_init_reg.vi_to_spec.find(g_mock_chain_vi);
  if (it == g_mock_init_reg.vi_to_spec.end()) {
    return MakeMockChainMissStatus();
  }
  *out = it->second.elem_type;
  return nullptr;
}
inline OrtStatus* StubMockGetDimensionsCount(const OrtTensorTypeAndShapeInfo*,
                                             size_t* out) noexcept {
  auto it = g_mock_init_reg.vi_to_spec.find(g_mock_chain_vi);
  if (it == g_mock_init_reg.vi_to_spec.end()) {
    return MakeMockChainMissStatus();
  }
  *out = it->second.dims.size();
  return nullptr;
}
inline OrtStatus* StubMockGetDimensions(const OrtTensorTypeAndShapeInfo*,
                                        int64_t* d, size_t count) noexcept {
  auto it = g_mock_init_reg.vi_to_spec.find(g_mock_chain_vi);
  if (it == g_mock_init_reg.vi_to_spec.end()) {
    return MakeMockChainMissStatus();
  }
  for (size_t i = 0; i < count && i < it->second.dims.size(); ++i)
    d[i] = it->second.dims[i];
  return nullptr;
}
inline OrtStatus* StubMockValueInfoGetInitializerValue(const OrtValueInfo* vi,
                                                       const OrtValue** out) noexcept {
  g_mock_chain_vi = vi;  // refresh in case chain was broken
  *out = reinterpret_cast<const OrtValue*>(&g_mock_ort_value_sentinel);
  return nullptr;
}
inline OrtStatus* StubMockGetTensorData(const OrtValue*,
                                        const void** out) noexcept {
  auto it = g_mock_init_reg.vi_to_spec.find(g_mock_chain_vi);
  if (it != g_mock_init_reg.vi_to_spec.end() && !it->second.raw_bytes.empty()) {
    *out = it->second.raw_bytes.data();
    return nullptr;
  }
  *out = nullptr;
  return MakeMockChainMissStatus();
}

// SetupMockInitRegistryStubs — install all stubs for the mock initializer registry.
// Call AFTER populating g_mock_init_reg with AddScalarFloat / etc.
// Accepts any context type that exposes a stub_ort_api member (OrtApiStubContext
// or any struct that inherits from it).
template <typename T>
inline void SetupMockInitRegistryStubs(T& ctx) {
  g_mock_chain_vi = nullptr;
  ctx.stub_ort_api.Graph_GetModelPath = StubMockGraphGetModelPath;
  ctx.stub_ort_api.Graph_GetNumInitializers = StubMockGetNumInitializers;
  ctx.stub_ort_api.Graph_GetInitializers = StubMockGetInitializers;
  ctx.stub_ort_api.GetValueInfoName = StubMockGetValueInfoName;
  ctx.stub_ort_api.ValueInfo_IsConstantInitializer = StubMockValueInfoIsConstant;
  ctx.stub_ort_api.ValueInfo_GetExternalInitializerInfo = StubMockValueInfoGetExternal;
  ctx.stub_ort_api.GetValueInfoTypeInfo = StubMockGetValueInfoTypeInfo;
  ctx.stub_ort_api.CastTypeInfoToTensorInfo = StubMockCastTypeInfoToTensorInfo;
  ctx.stub_ort_api.GetTensorElementType = StubMockGetTensorElementType;
  ctx.stub_ort_api.GetDimensionsCount = StubMockGetDimensionsCount;
  ctx.stub_ort_api.GetDimensions = StubMockGetDimensions;
  ctx.stub_ort_api.ValueInfo_GetInitializerValue = StubMockValueInfoGetInitializerValue;
  ctx.stub_ort_api.GetTensorData = StubMockGetTensorData;
}

}  // namespace

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
