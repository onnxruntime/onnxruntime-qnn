// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Component-level unit tests for ort_api.cc.
//
// Covers the pure-utility functions that do not require a real OrtNode/OrtGraph:
//   GetProviderOptionPrefix, ReadFileIntoBuffer, OrtLoadDynamicLibrary,
//   OrtUnloadDynamicLibrary, OrtGetSymbolFromLibrary, GetSessionConfigEntryOrDefault,
//   OrtGetRuntimePath, GetDynamicLibraryLocationByAddress.
//
// NOTE: OrtNodeAttrHelper, OrtNodeUnit, ParseOrtValueInfo, and GetQDQIODefs are
// integration test territory — they require a real OrtNode* and would crash or
// produce UB with fake/null pointers. Those functions are NOT covered here.
// Component-level ceiling for ort_api.cc is ~15% for this reason.

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

// Platform coverage:
//   Linux x86-64 (coverage build): all sections.
//   Windows: GetSessionConfigEntryOrDefault, OrtNodeAttrHelper (not-found),
//            and OrtNodeUnit stub series only.

#ifndef _WIN32
#include <unistd.h>  // mkstemp, unlink
#endif

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace test {

// ============================================================
// GetProviderOptionPrefix
// ============================================================

TEST(QnnUnit_OrtApiTest, GetProviderOptionPrefix_UppercaseName_Lowercased) {
  EXPECT_EQ(GetProviderOptionPrefix("QNNExecutionProvider"), "ep.qnnexecutionprovider.");
}

TEST(QnnUnit_OrtApiTest, GetProviderOptionPrefix_AlreadyLowercase) {
  EXPECT_EQ(GetProviderOptionPrefix("cpu"), "ep.cpu.");
}

TEST(QnnUnit_OrtApiTest, GetProviderOptionPrefix_EmptyName) {
  EXPECT_EQ(GetProviderOptionPrefix(""), "ep..");
}

TEST(QnnUnit_OrtApiTest, GetProviderOptionPrefix_MixedCase) {
  EXPECT_EQ(GetProviderOptionPrefix("MyProvider"), "ep.myprovider.");
}

// ============================================================
// OrtGetRuntimePath  (Linux: always returns "")
// ============================================================

#ifndef _WIN32
TEST(QnnUnit_OrtApiTest, OrtGetRuntimePath_ReturnsEmptyStringOnLinux) {
  EXPECT_TRUE(OrtGetRuntimePath().empty());
}
#endif

// ============================================================
// GetDynamicLibraryLocationByAddress  (Linux: always returns {})
// ============================================================

#ifndef _WIN32
TEST(QnnUnit_OrtApiTest, GetDynamicLibraryLocationByAddress_ReturnsEmptyOnLinux) {
  EXPECT_TRUE(GetDynamicLibraryLocationByAddress(nullptr).empty());
  EXPECT_TRUE(GetDynamicLibraryLocationByAddress(reinterpret_cast<const void*>(uintptr_t{1})).empty());
}
#endif

// ============================================================
// ReadFileIntoBuffer  (Linux POSIX path)
// ============================================================

#ifndef _WIN32

namespace {
// Write content to a new temp file. Returns the path, or "" on failure.
std::string MakeTempFile(const std::string& content) {
  char tmpl[] = "/tmp/ort_api_ut_XXXXXX";
  int fd = ::mkstemp(tmpl);
  if (fd == -1) return "";
  ::close(fd);
  std::string path(tmpl);
  std::ofstream f(path, std::ios::binary);
  f.write(content.data(), static_cast<std::streamsize>(content.size()));
  return path;
}
}  // namespace

TEST(QnnUnit_OrtApiTest, ReadFileIntoBuffer_NullPath_Fails) {
  std::vector<char> buf(10);
  auto status = ReadFileIntoBuffer(nullptr, 0, 5, {buf.data(), buf.size()});
  EXPECT_FALSE(status.IsOK());
}

TEST(QnnUnit_OrtApiTest, ReadFileIntoBuffer_NegativeOffset_Fails) {
  std::vector<char> buf(10);
  auto status = ReadFileIntoBuffer("/tmp", -1, 5, {buf.data(), buf.size()});
  EXPECT_FALSE(status.IsOK());
}

TEST(QnnUnit_OrtApiTest, ReadFileIntoBuffer_LengthExceedsBuffer_Fails) {
  std::vector<char> buf(5);
  auto status = ReadFileIntoBuffer("/tmp/whatever", 0, 10, {buf.data(), buf.size()});
  EXPECT_FALSE(status.IsOK());
}

TEST(QnnUnit_OrtApiTest, ReadFileIntoBuffer_FileNotFound_Fails) {
  std::vector<char> buf(10);
  auto status = ReadFileIntoBuffer(
      "/tmp/ort_api_nonexistent_file_xyz_123456789", 0, 5, {buf.data(), buf.size()});
  EXPECT_FALSE(status.IsOK());
}

TEST(QnnUnit_OrtApiTest, ReadFileIntoBuffer_ZeroLength_Succeeds) {
  std::string path = MakeTempFile("hello");
  ASSERT_FALSE(path.empty());

  std::vector<char> buf(10, 'x');
  auto status = ReadFileIntoBuffer(path.c_str(), 0, 0, {buf.data(), buf.size()});
  EXPECT_TRUE(status.IsOK());
  // Buffer should be unchanged since 0 bytes were requested.
  EXPECT_EQ(buf[0], 'x');

  ::unlink(path.c_str());
}

TEST(QnnUnit_OrtApiTest, ReadFileIntoBuffer_ReadFullFile_Succeeds) {
  const std::string content = "hello world";
  std::string path = MakeTempFile(content);
  ASSERT_FALSE(path.empty());

  std::vector<char> buf(content.size());
  auto status = ReadFileIntoBuffer(path.c_str(), 0, content.size(), {buf.data(), buf.size()});
  EXPECT_TRUE(status.IsOK());
  EXPECT_EQ(std::string(buf.begin(), buf.end()), content);

  ::unlink(path.c_str());
}

TEST(QnnUnit_OrtApiTest, ReadFileIntoBuffer_ReadWithOffset_Succeeds) {
  const std::string content = "hello world";
  std::string path = MakeTempFile(content);
  ASSERT_FALSE(path.empty());

  constexpr int64_t kOffset = 6;  // skip "hello "
  const std::string expected = content.substr(static_cast<size_t>(kOffset));
  std::vector<char> buf(expected.size());
  auto status = ReadFileIntoBuffer(path.c_str(), kOffset, expected.size(), {buf.data(), buf.size()});
  EXPECT_TRUE(status.IsOK());
  EXPECT_EQ(std::string(buf.begin(), buf.end()), expected);

  ::unlink(path.c_str());
}

// ============================================================
// OrtLoadDynamicLibrary / OrtUnloadDynamicLibrary / OrtGetSymbolFromLibrary
// ============================================================

TEST(QnnUnit_OrtApiTest, OrtLoadDynamicLibrary_NonExistentLib_Fails) {
  void* handle = nullptr;
  auto status = OrtLoadDynamicLibrary("/this_library_does_not_exist_xyz.so", false, &handle);
  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ(handle, nullptr);
}

TEST(QnnUnit_OrtApiTest, OrtLoadUnloadDynamicLibrary_ValidLib_Succeeds) {
  void* handle = nullptr;
  auto load_status = OrtLoadDynamicLibrary("libm.so.6", false, &handle);
  if (!load_status.IsOK()) GTEST_SKIP() << "libm.so.6 not available";
  EXPECT_NE(handle, nullptr);

  auto unload_status = OrtUnloadDynamicLibrary(handle);
  EXPECT_TRUE(unload_status.IsOK());
}

TEST(QnnUnit_OrtApiTest, OrtUnloadDynamicLibrary_NullHandle_Fails) {
  // ort_api.cc explicitly null-checks the handle before calling dlclose()
  // (see OrtUnloadDynamicLibrary), so this assertion is intentional and
  // platform-safe regardless of dlclose(NULL) behavior on the current libc.
  auto status = OrtUnloadDynamicLibrary(nullptr);
  EXPECT_FALSE(status.IsOK());
}

TEST(QnnUnit_OrtApiTest, OrtGetSymbolFromLibrary_NullHandle_KnownSymbol_Succeeds) {
  // RTLD_DEFAULT search — "malloc" is always present in the process.
  void* symbol = nullptr;
  auto status = OrtGetSymbolFromLibrary(nullptr, "malloc", &symbol);
  EXPECT_TRUE(status.IsOK());
  EXPECT_NE(symbol, nullptr);
}

TEST(QnnUnit_OrtApiTest, OrtGetSymbolFromLibrary_NullHandle_UnknownSymbol_Fails) {
  void* symbol = nullptr;
  auto status = OrtGetSymbolFromLibrary(
      nullptr, "__ort_api_test_nonexistent_symbol_xyz_abc_qnn", &symbol);
  EXPECT_FALSE(status.IsOK());
}

TEST(QnnUnit_OrtApiTest, OrtGetSymbolFromLibrary_ValidHandle_FindsSymbol) {
  void* handle = nullptr;
  auto load_status = OrtLoadDynamicLibrary("libm.so.6", false, &handle);
  if (!load_status.IsOK() || !handle) GTEST_SKIP() << "libm.so.6 not available";

  void* symbol = nullptr;
  auto status = OrtGetSymbolFromLibrary(handle, "sin", &symbol);
  EXPECT_TRUE(status.IsOK());
  EXPECT_NE(symbol, nullptr);

  (void)OrtUnloadDynamicLibrary(handle);
}

#endif  // !_WIN32

// ============================================================
// GetSessionConfigEntryOrDefault
// ============================================================

TEST(QnnUnit_OrtApiTest, GetSessionConfigEntryOrDefault_KeyNotFound_ReturnsDefault) {
  OrtSessionOptions* raw_opts = nullptr;
  ASSERT_EQ(Ort::GetApi().CreateSessionOptions(&raw_opts), nullptr);
  ASSERT_NE(raw_opts, nullptr);

  std::string result;
  OrtStatus* s = GetSessionConfigEntryOrDefault(
      Ort::GetApi(), *raw_opts, "nonexistent.config.key", "expected_default", result);
  EXPECT_EQ(s, nullptr);
  EXPECT_EQ(result, "expected_default");

  Ort::GetApi().ReleaseSessionOptions(raw_opts);
}

TEST(QnnUnit_OrtApiTest, GetSessionConfigEntryOrDefault_KeyFound_ReturnsValue) {
  OrtSessionOptions* raw_opts = nullptr;
  ASSERT_EQ(Ort::GetApi().CreateSessionOptions(&raw_opts), nullptr);
  ASSERT_NE(raw_opts, nullptr);

  ASSERT_EQ(Ort::GetApi().AddSessionConfigEntry(raw_opts, "ep.qnn.backend_path", "libQnnCpu.so"), nullptr);

  std::string result;
  OrtStatus* s = GetSessionConfigEntryOrDefault(
      Ort::GetApi(), *raw_opts, "ep.qnn.backend_path", "default_value", result);
  EXPECT_EQ(s, nullptr);
  EXPECT_EQ(result, "libQnnCpu.so");

  Ort::GetApi().ReleaseSessionOptions(raw_opts);
}

// ============================================================
// OrtNodeAttrHelper — uses OrtModelEditorApi::CreateNode to obtain
// a real OrtNode* with known attributes, avoiding a full ORT session.
// ============================================================

namespace {

// Returns nullptr if OrtModelEditorApi is unavailable (minimal build).
const OrtModelEditorApi* GetEditorApi() {
  return Ort::GetApi().GetModelEditorApi();
}

// Creates a standalone OrtNode via model editor. Caller owns the node;
// release with Ort::GetApi().ReleaseNode(node).
// Attributes are copied into the node; callers should release their OrtOpAttr*
// objects after this call.
OrtNode* MakeAttrTestNode(const OrtModelEditorApi* ed,
                          OrtOpAttr** attrs,
                          size_t attr_count) {
  OrtNode* node = nullptr;
  OrtStatus* s = ed->CreateNode("Relu", "", "attr_test",
                                nullptr, 0, nullptr, 0,
                                attrs, attr_count, &node);
  if (s) {
    Ort::GetApi().ReleaseStatus(s);
    return nullptr;
  }
  return node;
}

}  // namespace

// All Get-with-default overloads: key absent → return supplied default.
// Also covers GetFloat/GetInt64/GetFloats/GetInt64s/GetString → nullopt,
// and HasAttr → false.
TEST(QnnUnit_OrtApiTest, OrtNodeAttrHelper_Get_KeyNotFound_ReturnsDefault) {
  const OrtModelEditorApi* ed = GetEditorApi();
  if (!ed) GTEST_SKIP() << "OrtModelEditorApi not available";

  OrtNode* node = MakeAttrTestNode(ed, nullptr, 0);
  ASSERT_NE(node, nullptr);

  OrtNodeAttrHelper h(*node);

  EXPECT_NEAR(h.Get("f", 1.5f), 1.5f, 1e-7f);
  EXPECT_EQ(h.Get("i32", int32_t{7}), 7);
  EXPECT_EQ(h.Get("u32", uint32_t{8}), 8u);
  EXPECT_EQ(h.Get("i64", int64_t{99}), 99LL);
  EXPECT_EQ(h.Get("s", std::string{"def"}), "def");
  EXPECT_EQ(h.Get("vs", std::vector<std::string>{"x"}), (std::vector<std::string>{"x"}));
  EXPECT_EQ(h.Get("vi32", std::vector<int32_t>{1, 2}), (std::vector<int32_t>{1, 2}));
  EXPECT_EQ(h.Get("vu32", std::vector<uint32_t>{3u}), (std::vector<uint32_t>{3u}));
  EXPECT_EQ(h.Get("vi64", std::vector<int64_t>{5, 6}), (std::vector<int64_t>{5, 6}));
  EXPECT_EQ(h.Get("vf", std::vector<float>{0.1f}), (std::vector<float>{0.1f}));

  EXPECT_EQ(h.GetFloat("f"), std::nullopt);
  EXPECT_EQ(h.GetInt64("i64"), std::nullopt);
  EXPECT_EQ(h.GetFloats("vf"), std::nullopt);
  EXPECT_EQ(h.GetInt64s("vi64"), std::nullopt);
  EXPECT_EQ(h.GetString("s"), std::nullopt);

  EXPECT_FALSE(h.HasAttr("f"));

  Ort::GetApi().ReleaseNode(node);
}

// NOTE: OrtNodeAttrHelper "found" (key present → return value) paths require a real
// inference-graph OrtNode* from the OrtEpApi compilation path.
// OrtModelEditorApi::CreateNode produces an editor node whose attribute store is
// incompatible with OrtApi::Node_GetAttributeByName. Those paths are integration
// test territory and are not covered here.

// ============================================================
// OrtNodeUnit + ParseOrtValueInfo — via stub OrtApi + fake opaque pointers
//
// ParseOrtValueInfo is file-local in ort_api.cc; it is exercised through the
// OrtNodeUnit(const OrtNode*, const OrtApi&) constructor → InitForSingleNode.
//
// All OrtApi calls in ParseOrtValueInfo and InitForSingleNode go through the
// *parameter* api (not Ort::GetApi()), so every branch is stubable.
//
// Exception: DequantizeLinear / QuantizeLinear paths in InitForSingleNode call
// OrtNodeAttrHelper which uses the global OrtApi — those paths are integration
// test territory and are not covered here.
// ============================================================

namespace {

// Non-null sentinel addresses — stubs never dereference these.
const OrtValueInfo* const kFakeIO =
    reinterpret_cast<const OrtValueInfo*>(uintptr_t{0x1000});
const OrtTypeInfo* const kFakeTypeInfo =
    reinterpret_cast<const OrtTypeInfo*>(uintptr_t{0x2000});
const OrtTensorTypeAndShapeInfo* const kFakeTensorShape =
    reinterpret_cast<const OrtTensorTypeAndShapeInfo*>(uintptr_t{0x3000});
const OrtNode* const kFakeNode =
    reinterpret_cast<const OrtNode*>(uintptr_t{0x100});
const OrtNode* const kFakeConsumer =
    reinterpret_cast<const OrtNode*>(uintptr_t{0x300});
const OrtGraph* const kFakeGraph =
    reinterpret_cast<const OrtGraph*>(uintptr_t{0x400});

// ParseOrtValueInfo stubs
OrtStatus* StubGetValueInfoName(const OrtValueInfo*, const char** name) noexcept {
  static const char n[] = "x";
  *name = n;
  return nullptr;
}
OrtStatus* StubGetValueInfoTypeInfo(const OrtValueInfo*, const OrtTypeInfo** ti) noexcept {
  *ti = kFakeTypeInfo;
  return nullptr;
}
OrtStatus* StubCastTypeInfoToTensorInfo(
    const OrtTypeInfo*, const OrtTensorTypeAndShapeInfo** ts) noexcept {
  *ts = kFakeTensorShape;
  return nullptr;
}
OrtStatus* StubCastTypeInfoToTensorInfo_Null(
    const OrtTypeInfo*, const OrtTensorTypeAndShapeInfo** ts) noexcept {
  *ts = nullptr;
  return nullptr;
}
OrtStatus* StubGetTensorElementType(
    const OrtTensorTypeAndShapeInfo*, ONNXTensorElementDataType* t) noexcept {
  *t = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  return nullptr;
}
bool StubHasShape_True(const OrtTensorTypeAndShapeInfo*) noexcept { return true; }
bool StubHasShape_False(const OrtTensorTypeAndShapeInfo*) noexcept { return false; }
OrtStatus* StubGetDimsCount_Two(const OrtTensorTypeAndShapeInfo*, size_t* n) noexcept {
  *n = 2;
  return nullptr;
}
OrtStatus* StubGetDimsCount_Zero(const OrtTensorTypeAndShapeInfo*, size_t* n) noexcept {
  *n = 0;
  return nullptr;
}
OrtStatus* StubGetDimensions(
    const OrtTensorTypeAndShapeInfo*, int64_t* dims, size_t) noexcept {
  dims[0] = 3;
  dims[1] = 4;
  return nullptr;
}

// InitForSingleNode stubs
OrtStatus* StubNodeGetNumInputs_One(const OrtNode*, size_t* n) noexcept {
  *n = 1;
  return nullptr;
}
OrtStatus* StubNodeGetNumOutputs_One(const OrtNode*, size_t* n) noexcept {
  *n = 1;
  return nullptr;
}
OrtStatus* StubNodeGetInputs_One(
    const OrtNode*, const OrtValueInfo** io, size_t) noexcept {
  io[0] = kFakeIO;
  return nullptr;
}
OrtStatus* StubNodeGetInputs_NullFirst(
    const OrtNode*, const OrtValueInfo** io, size_t) noexcept {
  io[0] = nullptr;
  return nullptr;
}
OrtStatus* StubNodeGetOutputs_One(
    const OrtNode*, const OrtValueInfo** io, size_t) noexcept {
  io[0] = kFakeIO;
  return nullptr;
}
OrtStatus* StubNodeGetOperatorType_Relu(const OrtNode*, const char** op) noexcept {
  static const char t[] = "Relu";
  *op = t;
  return nullptr;
}
OrtStatus* StubCreateStatus_Sentinel(OrtErrorCode, const char*) noexcept {
  return reinterpret_cast<OrtStatus*>(uintptr_t{1});
}
void StubReleaseStatus_Noop(OrtStatus*) noexcept {}

// GetInputEdgesCount / GetOutputNodes stubs
OrtStatus* StubValueInfoGetProducer_NonNull(
    const OrtValueInfo*, const OrtNode** p, size_t*) noexcept {
  *p = reinterpret_cast<const OrtNode*>(uintptr_t{0x200});
  return nullptr;
}
OrtStatus* StubValueInfoGetProducer_Null(
    const OrtValueInfo*, const OrtNode** p, size_t*) noexcept {
  *p = nullptr;
  return nullptr;
}
OrtStatus* StubValueInfoGetNumConsumers_One(const OrtValueInfo*, size_t* n) noexcept {
  *n = 1;
  return nullptr;
}
OrtStatus* StubValueInfoGetNumConsumers_Zero(const OrtValueInfo*, size_t* n) noexcept {
  *n = 0;
  return nullptr;
}
OrtStatus* StubValueInfoGetConsumers_One(
    const OrtValueInfo*, const OrtNode** c, int64_t* idx, size_t) noexcept {
  c[0] = kFakeConsumer;
  idx[0] = 0;
  return nullptr;
}
OrtStatus* StubValueInfoGetConsumers_Zero(
    const OrtValueInfo*, const OrtNode**, int64_t*, size_t) noexcept {
  return nullptr;
}

// GetModelPathString stubs
OrtStatus* StubGraphGetModelPath_Ok(const OrtGraph*, const ORTCHAR_T** path) noexcept {
  static const ORTCHAR_T p[] = ORT_TSTR("/model/path.onnx");
  *path = p;
  return nullptr;
}

// Builds a fully-stubbed OrtApi for OrtNodeUnit(const OrtNode*, const OrtApi&).
// Defaults: single Relu node, one float input/output, shape [3, 4].
OrtApi MakeNodeUnitApi() {
  OrtApi api{};
  api.GetValueInfoName = StubGetValueInfoName;
  api.GetValueInfoTypeInfo = StubGetValueInfoTypeInfo;
  api.CastTypeInfoToTensorInfo = StubCastTypeInfoToTensorInfo;
  api.GetTensorElementType = StubGetTensorElementType;
  api.TensorTypeAndShape_HasShape = StubHasShape_True;
  api.GetDimensionsCount = StubGetDimsCount_Two;
  api.GetDimensions = StubGetDimensions;
  api.Node_GetNumInputs = StubNodeGetNumInputs_One;
  api.Node_GetNumOutputs = StubNodeGetNumOutputs_One;
  api.Node_GetInputs = StubNodeGetInputs_One;
  api.Node_GetOutputs = StubNodeGetOutputs_One;
  api.Node_GetOperatorType = StubNodeGetOperatorType_Relu;
  api.CreateStatus = StubCreateStatus_Sentinel;
  api.ReleaseStatus = StubReleaseStatus_Noop;
  return api;
}

}  // namespace

// ============================================================
// OrtNodeUnit — SingleNode constructor (covers InitForSingleNode + ParseOrtValueInfo)
// ============================================================

TEST(QnnUnit_OrtApiTest, OrtNodeUnit_SingleNode_HasShape_CorrectIODefs) {
  OrtApi api = MakeNodeUnitApi();
  OrtNodeUnit unit(kFakeNode, api);

  EXPECT_EQ(unit.UnitType(), OrtNodeUnit::Type::SingleNode);
  ASSERT_EQ(unit.Inputs().size(), 1u);
  ASSERT_EQ(unit.Outputs().size(), 1u);

  const auto& in = unit.Inputs()[0];
  EXPECT_EQ(in.name, "x");
  EXPECT_EQ(in.type, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  ASSERT_TRUE(in.shape.has_value());
  EXPECT_EQ(*in.shape, (std::vector<int64_t>{3, 4}));
  EXPECT_FALSE(in.quant_param.has_value());

  const auto& out = unit.Outputs()[0];
  EXPECT_EQ(out.name, "x");
  EXPECT_EQ(out.type, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  ASSERT_TRUE(out.shape.has_value());
}

TEST(QnnUnit_OrtApiTest, OrtNodeUnit_SingleNode_NoShape_ShapeIsNullopt) {
  OrtApi api = MakeNodeUnitApi();
  api.TensorTypeAndShape_HasShape = StubHasShape_False;
  OrtNodeUnit unit(kFakeNode, api);

  ASSERT_EQ(unit.Inputs().size(), 1u);
  EXPECT_EQ(unit.Inputs()[0].name, "x");
  EXPECT_EQ(unit.Inputs()[0].type, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  EXPECT_FALSE(unit.Inputs()[0].shape.has_value());
}

TEST(QnnUnit_OrtApiTest, OrtNodeUnit_SingleNode_ZeroDims_ShapeIsEmptyVector) {
  OrtApi api = MakeNodeUnitApi();
  api.GetDimensionsCount = StubGetDimsCount_Zero;
  // GetDimensions is not called when num_dims == 0; leave stub null to confirm.
  api.GetDimensions = nullptr;

  OrtNodeUnit unit(kFakeNode, api);

  ASSERT_EQ(unit.Inputs().size(), 1u);
  ASSERT_TRUE(unit.Inputs()[0].shape.has_value());
  EXPECT_TRUE(unit.Inputs()[0].shape->empty());
}

TEST(QnnUnit_OrtApiTest, OrtNodeUnit_SingleNode_NullInput_OptionalFallback) {
  OrtApi api = MakeNodeUnitApi();
  api.Node_GetInputs = StubNodeGetInputs_NullFirst;

  OrtNodeUnit unit(kFakeNode, api);

  ASSERT_EQ(unit.Inputs().size(), 1u);
  EXPECT_EQ(unit.Inputs()[0].name, "");
  EXPECT_EQ(unit.Inputs()[0].type, ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);
  EXPECT_FALSE(unit.Inputs()[0].shape.has_value());
}

TEST(QnnUnit_OrtApiTest, OrtNodeUnit_SingleNode_NonTensorType_FallsBackToUndefined) {
  OrtApi api = MakeNodeUnitApi();
  // Non-tensor: ParseOrtValueInfo returns an error; InitForSingleNode uses undefined fallback.
  api.CastTypeInfoToTensorInfo = StubCastTypeInfoToTensorInfo_Null;

  OrtNodeUnit unit(kFakeNode, api);

  ASSERT_EQ(unit.Inputs().size(), 1u);
  EXPECT_EQ(unit.Inputs()[0].name, "");
  EXPECT_EQ(unit.Inputs()[0].type, ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);
}

// ============================================================
// OrtNodeUnit::GetInputEdgesCount
// ============================================================

TEST(QnnUnit_OrtApiTest, OrtNodeUnit_GetInputEdgesCount_WithProducer) {
  OrtApi api = MakeNodeUnitApi();
  OrtNodeUnit unit(kFakeNode, api);

  OrtApi edge_api = MakeNodeUnitApi();
  edge_api.ValueInfo_GetValueProducer = StubValueInfoGetProducer_NonNull;

  EXPECT_EQ(unit.GetInputEdgesCount(edge_api), 1u);
}

TEST(QnnUnit_OrtApiTest, OrtNodeUnit_GetInputEdgesCount_NoProducer) {
  OrtApi api = MakeNodeUnitApi();
  OrtNodeUnit unit(kFakeNode, api);

  OrtApi edge_api = MakeNodeUnitApi();
  edge_api.ValueInfo_GetValueProducer = StubValueInfoGetProducer_Null;

  EXPECT_EQ(unit.GetInputEdgesCount(edge_api), 0u);
}

// ============================================================
// OrtNodeUnit::GetOutputNodes
// ============================================================

TEST(QnnUnit_OrtApiTest, OrtNodeUnit_GetOutputNodes_WithConsumer) {
  OrtApi api = MakeNodeUnitApi();
  OrtNodeUnit unit(kFakeNode, api);

  OrtApi out_api = MakeNodeUnitApi();
  out_api.ValueInfo_GetValueNumConsumers = StubValueInfoGetNumConsumers_One;
  out_api.ValueInfo_GetValueConsumers = StubValueInfoGetConsumers_One;

  auto consumers = unit.GetOutputNodes(out_api);
  ASSERT_EQ(consumers.size(), 1u);
  EXPECT_EQ(consumers[0], kFakeConsumer);
}

TEST(QnnUnit_OrtApiTest, OrtNodeUnit_GetOutputNodes_NoConsumers) {
  OrtApi api = MakeNodeUnitApi();
  OrtNodeUnit unit(kFakeNode, api);

  OrtApi out_api = MakeNodeUnitApi();
  out_api.ValueInfo_GetValueNumConsumers = StubValueInfoGetNumConsumers_Zero;
  out_api.ValueInfo_GetValueConsumers = StubValueInfoGetConsumers_Zero;

  EXPECT_TRUE(unit.GetOutputNodes(out_api).empty());
}

// ============================================================
// GetModelPathString
// ============================================================

TEST(QnnUnit_OrtApiTest, GetModelPathString_Success_ReturnsPath) {
  OrtApi api{};
  api.Graph_GetModelPath = StubGraphGetModelPath_Ok;
  api.ReleaseStatus = StubReleaseStatus_Noop;

  auto path = GetModelPathString(kFakeGraph, api);
  EXPECT_FALSE(path.empty());
#ifndef _WIN32
  EXPECT_NE(path.find(ORT_TSTR("path.onnx")), std::basic_string<ORTCHAR_T>::npos);
#endif
}

TEST(QnnUnit_OrtApiTest, GetModelPathString_Error_ReturnsEmptyString) {
  OrtApi api{};
  api.Graph_GetModelPath = [](const OrtGraph*, const ORTCHAR_T**) noexcept -> OrtStatus* {
    return reinterpret_cast<OrtStatus*>(uintptr_t{1});
  };
  api.ReleaseStatus = StubReleaseStatus_Noop;

  EXPECT_TRUE(GetModelPathString(kFakeGraph, api).empty());
}

// ============================================================
// Additional stubs for error-path and null-IO coverage
// ============================================================

namespace {

OrtStatus* StubNodeGetOutputs_NullFirst(
    const OrtNode*, const OrtValueInfo** io, size_t) noexcept {
  io[0] = nullptr;
  return nullptr;
}

OrtStatus* StubNodeGetNumInputs_Error(const OrtNode*, size_t*) noexcept {
  return reinterpret_cast<OrtStatus*>(uintptr_t{1});
}
OrtStatus* StubNodeGetNumOutputs_Error(const OrtNode*, size_t*) noexcept {
  return reinterpret_cast<OrtStatus*>(uintptr_t{1});
}
OrtStatus* StubNodeGetInputs_Error(
    const OrtNode*, const OrtValueInfo**, size_t) noexcept {
  return reinterpret_cast<OrtStatus*>(uintptr_t{1});
}
OrtStatus* StubNodeGetOutputs_Error(
    const OrtNode*, const OrtValueInfo**, size_t) noexcept {
  return reinterpret_cast<OrtStatus*>(uintptr_t{1});
}

OrtStatus* StubValueInfoGetProducer_Error(
    const OrtValueInfo*, const OrtNode**, size_t*) noexcept {
  return reinterpret_cast<OrtStatus*>(uintptr_t{1});
}
OrtStatus* StubValueInfoGetNumConsumers_Error(
    const OrtValueInfo*, size_t*) noexcept {
  return reinterpret_cast<OrtStatus*>(uintptr_t{1});
}
OrtStatus* StubValueInfoGetConsumers_Error(
    const OrtValueInfo*, const OrtNode**, int64_t*, size_t) noexcept {
  return reinterpret_cast<OrtStatus*>(uintptr_t{1});
}

}  // namespace

// ============================================================
// OrtNodeUnit — null-output path in InitForSingleNode (line 327)
// ============================================================

TEST(QnnUnit_OrtApiTest, OrtNodeUnit_SingleNode_NullOutput_OptionalFallback) {
  OrtApi api = MakeNodeUnitApi();
  api.Node_GetOutputs = StubNodeGetOutputs_NullFirst;

  OrtNodeUnit unit(kFakeNode, api);

  ASSERT_EQ(unit.Outputs().size(), 1u);
  EXPECT_EQ(unit.Outputs()[0].name, "");
  EXPECT_EQ(unit.Outputs()[0].type, ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);
}

// ============================================================
// OrtNodeUnit::GetInputEdgesCount — error + null-input paths
// ============================================================

TEST(QnnUnit_OrtApiTest, OrtNodeUnit_GetInputEdgesCount_NullInput_SkipsIt) {
  OrtApi api = MakeNodeUnitApi();
  OrtNodeUnit unit(kFakeNode, api);

  OrtApi edge_api = MakeNodeUnitApi();
  edge_api.Node_GetInputs = StubNodeGetInputs_NullFirst;

  EXPECT_EQ(unit.GetInputEdgesCount(edge_api), 0u);
}

TEST(QnnUnit_OrtApiTest, OrtNodeUnit_GetInputEdgesCount_NumInputsError_ReturnsZero) {
  OrtApi api = MakeNodeUnitApi();
  OrtNodeUnit unit(kFakeNode, api);

  OrtApi edge_api = MakeNodeUnitApi();
  edge_api.Node_GetNumInputs = StubNodeGetNumInputs_Error;

  EXPECT_EQ(unit.GetInputEdgesCount(edge_api), 0u);
}

TEST(QnnUnit_OrtApiTest, OrtNodeUnit_GetInputEdgesCount_GetInputsError_ReturnsZero) {
  OrtApi api = MakeNodeUnitApi();
  OrtNodeUnit unit(kFakeNode, api);

  OrtApi edge_api = MakeNodeUnitApi();
  edge_api.Node_GetInputs = StubNodeGetInputs_Error;

  EXPECT_EQ(unit.GetInputEdgesCount(edge_api), 0u);
}

TEST(QnnUnit_OrtApiTest, OrtNodeUnit_GetInputEdgesCount_ProducerError_SkipsInput) {
  OrtApi api = MakeNodeUnitApi();
  OrtNodeUnit unit(kFakeNode, api);

  OrtApi edge_api = MakeNodeUnitApi();
  edge_api.ValueInfo_GetValueProducer = StubValueInfoGetProducer_Error;

  EXPECT_EQ(unit.GetInputEdgesCount(edge_api), 0u);
}

// ============================================================
// OrtNodeUnit::GetOutputNodes — error + null-output paths
// ============================================================

TEST(QnnUnit_OrtApiTest, OrtNodeUnit_GetOutputNodes_NullOutput_SkipsIt) {
  OrtApi api = MakeNodeUnitApi();
  OrtNodeUnit unit(kFakeNode, api);

  OrtApi out_api = MakeNodeUnitApi();
  out_api.Node_GetOutputs = StubNodeGetOutputs_NullFirst;

  EXPECT_TRUE(unit.GetOutputNodes(out_api).empty());
}

TEST(QnnUnit_OrtApiTest, OrtNodeUnit_GetOutputNodes_NumOutputsError_ReturnsEmpty) {
  OrtApi api = MakeNodeUnitApi();
  OrtNodeUnit unit(kFakeNode, api);

  OrtApi out_api = MakeNodeUnitApi();
  out_api.Node_GetNumOutputs = StubNodeGetNumOutputs_Error;

  EXPECT_TRUE(unit.GetOutputNodes(out_api).empty());
}

TEST(QnnUnit_OrtApiTest, OrtNodeUnit_GetOutputNodes_NumConsumersError_SkipsOutput) {
  OrtApi api = MakeNodeUnitApi();
  OrtNodeUnit unit(kFakeNode, api);

  OrtApi out_api = MakeNodeUnitApi();
  out_api.ValueInfo_GetValueNumConsumers = StubValueInfoGetNumConsumers_Error;

  EXPECT_TRUE(unit.GetOutputNodes(out_api).empty());
}

TEST(QnnUnit_OrtApiTest, OrtNodeUnit_GetOutputNodes_GetOutputsError_ReturnsEmpty) {
  OrtApi api = MakeNodeUnitApi();
  OrtNodeUnit unit(kFakeNode, api);

  OrtApi out_api = MakeNodeUnitApi();
  out_api.Node_GetOutputs = StubNodeGetOutputs_Error;

  EXPECT_TRUE(unit.GetOutputNodes(out_api).empty());
}

TEST(QnnUnit_OrtApiTest, OrtNodeUnit_GetOutputNodes_GetConsumersError_SkipsOutput) {
  OrtApi api = MakeNodeUnitApi();
  OrtNodeUnit unit(kFakeNode, api);

  OrtApi out_api = MakeNodeUnitApi();
  out_api.ValueInfo_GetValueNumConsumers = StubValueInfoGetNumConsumers_One;
  out_api.ValueInfo_GetValueConsumers = StubValueInfoGetConsumers_Error;

  EXPECT_TRUE(unit.GetOutputNodes(out_api).empty());
}

// ============================================================
// OrtNodeUnit ctor — InitForSingleNode error path
// Lines 234, 236-237: ReleaseStatus + inputs_/outputs_.clear()
// ============================================================

TEST(QnnUnit_OrtApiTest, OrtNodeUnit_InitForSingleNode_Error_ClearsIODefs) {
  OrtApi api = MakeNodeUnitApi();
  // Node_GetNumInputs returns a fake non-null OrtStatus* → InitForSingleNode
  // returns it → constructor fires the error branch (lines 234, 236-237).
  api.Node_GetNumInputs = StubNodeGetNumInputs_Error;

  OrtNodeUnit unit(kFakeNode, api);

  EXPECT_TRUE(unit.Inputs().empty());
  EXPECT_TRUE(unit.Outputs().empty());
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
