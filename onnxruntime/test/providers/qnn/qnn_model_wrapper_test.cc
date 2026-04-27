// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

// These tests require direct access to both QNN EP builder internals
// (QnnModelWrapper, QnnTensorWrapper) and the OrtGraph abstract interface.
// This is only possible when QNN EP is built as a static library, because the
// shared library build redefines ORT types as opaque wrappers in
// provider_api.h / provider_wrappedtypes.h.
#if !defined(ORT_MINIMAL_BUILD) && BUILD_QNN_EP_STATIC_LIB

#include <cassert>
#include <cstring>

#include "core/graph/ep_api_types.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_def.h"

using namespace onnxruntime;
using namespace onnxruntime::qnn;

namespace onnxruntime {
namespace test {

namespace {

// Minimal stub implementation of OrtGraph's pure virtual interface.
// AddTensorWrapper never calls any OrtGraph methods — it only consults
// graph_inputs_ / graph_outputs_ inside QnnModelWrapper — so all virtual
// methods can return trivial/empty values.
// Using OrtGraph directly (not EpGraph) avoids depending on GraphViewer,
// Graph, Model, or EpGraph::Create, all of which are ORT internals not
// exported from onnxruntime.dll.
struct FakeOrtGraph : public OrtGraph {
  FakeOrtGraph() : OrtGraph(OrtGraphIrApi::kEpApi) {}

  const std::string& GetName() const override {
    static const std::string kName = "FakeGraph";
    return kName;
  }
  std::unique_ptr<onnxruntime::ModelMetadata> GetModelMetadata() const override { return nullptr; }
  const ORTCHAR_T* GetModelPath() const override { return nullptr; }
  int64_t GetOnnxIRVersion() const override { return 0; }
  onnxruntime::Status GetNumOperatorSets(size_t& n) const override {
    n = 0;
    return {};
  }
  onnxruntime::Status GetOperatorSets(gsl::span<const char*>, gsl::span<int64_t>) const override { return {}; }
  size_t GetNumInputs() const override { return 0; }
  onnxruntime::Status GetInputs(gsl::span<const OrtValueInfo*>) const override { return {}; }
  size_t GetNumOutputs() const override { return 0; }
  onnxruntime::Status GetOutputs(gsl::span<const OrtValueInfo*>) const override { return {}; }
  size_t GetNumInitializers() const override { return 0; }
  onnxruntime::Status GetInitializers(gsl::span<const OrtValueInfo*>) const override { return {}; }
  size_t GetNumNodes() const override { return 0; }
  onnxruntime::Status GetNodes(gsl::span<const OrtNode*>) const override { return {}; }
  onnxruntime::Status GetParentNode(const OrtNode*& node) const override {
    node = nullptr;
    return {};
  }
};

// Synthesize an Ort::Logger with ORT_LOGGING_LEVEL_FATAL without instantiating
// logging::LoggingManager (whose constructor and destructor are ORT internals not
// exported from onnxruntime.dll and therefore unavailable in this test binary).
//
// Ort::Logger is trivially copyable and its header documents it as "the size of
// two pointers". Its private layout is:
//   [const OrtLogger* logger_,  OrtLoggingLevel cached_severity_level_]
// cached_severity_level_ sits immediately after logger_ (no padding: a pointer
// is followed by a 32-bit int, which requires no gap on any ABI used here).
//
// With cached_severity_level_ = ORT_LOGGING_LEVEL_FATAL all ORT_CXX_LOG severity
// guards evaluate to false, so the null logger_ pointer is never forwarded to
// OrtApi::Logger_LogMessage.
static Ort::Logger MakeFatalOrtLogger() {
  static_assert(sizeof(Ort::Logger) == 2 * sizeof(void*),
                "Ort::Logger size mismatch — update layout assumptions in MakeFatalOrtLogger");
  static_assert(std::is_trivially_copyable<Ort::Logger>::value,
                "Ort::Logger must be trivially copyable for memcpy to be valid");
  // NOTE: The static_asserts above guard against SIZE changes but cannot detect field
  // REORDERING (private members are inaccessible to offsetof). If a new field is inserted
  // before cached_severity_level_, the struct size may remain unchanged (due to padding)
  // while the offset sizeof(void*) would now point at the wrong field.
  //
  // The assert below is a runtime canary: GetLoggingSeverityLevel() reads
  // cached_severity_level_ directly. If the memcpy landed at the wrong offset, the method
  // will not return FATAL and the assert fires immediately in debug builds.
  Ort::Logger logger;
  const OrtLoggingLevel fatal = ORT_LOGGING_LEVEL_FATAL;
  std::memcpy(reinterpret_cast<char*>(&logger) + sizeof(void*), &fatal, sizeof(OrtLoggingLevel));
  assert(logger.GetLoggingSeverityLevel() == ORT_LOGGING_LEVEL_FATAL &&
         "MakeFatalOrtLogger: offset sizeof(void*) for cached_severity_level_ is wrong — "
         "Ort::Logger layout has changed, update the memcpy offset");
  return logger;
}

// Helper to create a minimal QnnModelWrapper for unit testing.
// AddTensorWrapper does not invoke any QNN SDK functions, so we can use
// null handles and a zeroed-out interface struct.
// OrtEpApi and OrtModelEditorApi are zero-initialized members: AddTensorWrapper
// never calls through api_ptrs_, so null function pointers are safe here.
struct QnnModelWrapperTestContext {
  // FakeOrtGraph: stub for the OrtGraph& required by QnnModelWrapper's constructor.
  // Stored as a member so the const-ref in QnnModelWrapper stays valid.
  FakeOrtGraph fake_graph;
  QNN_INTERFACE_VER_TYPE qnn_interface;
  Qnn_BackendHandle_t backend_handle;
  // Ort::Logger with FATAL severity — all ORT_CXX_LOG guards fail, so the null
  // internal OrtLogger* is never used. This member must outlive any QnnModelWrapper
  // created by CreateWrapper().
  Ort::Logger ort_logger;
  std::unordered_map<std::string, size_t> input_index_map;
  std::unordered_map<std::string, size_t> output_index_map;
  // Stored as members so QnnModelWrapper's const-refs remain valid for its lifetime.
  GraphInputOutputInfo graph_inputs_info;
  GraphInputOutputInfo graph_outputs_info;
  // Zero-initialized API tables stored as members (not static locals) to make
  // per-instance ownership explicit and avoid the singleton appearance.
  OrtEpApi zero_ep_api{};
  OrtModelEditorApi zero_model_editor_api{};

  QnnModelWrapperTestContext()
      : qnn_interface(QNN_INTERFACE_VER_TYPE_INIT),
        backend_handle(nullptr),
        ort_logger(MakeFatalOrtLogger()) {}

  std::unique_ptr<QnnModelWrapper> CreateWrapper(const ModelSettings& settings) {
    // Sync index maps into GraphInputOutputInfo. QnnModelWrapper stores const-refs
    // to graph_inputs_info / graph_outputs_info, which are members of this struct
    // and outlive any wrapper created here.
    graph_inputs_info.indices = input_index_map;
    graph_outputs_info.indices = output_index_map;

    const ApiPtrs api_ptrs{Ort::GetApi(), zero_ep_api, zero_model_editor_api};

    return std::make_unique<QnnModelWrapper>(
        fake_graph,
        api_ptrs,
        ort_logger,
        qnn_interface,
        backend_handle,
        graph_inputs_info,
        graph_outputs_info,
        QnnBackendType::HTP,
        settings);
  }
};

}  // namespace

// Verifies that when htp_shared_memory is disabled (default), the mem type of a
// graph input tensor remains QNN_TENSORMEMTYPE_RAW.
TEST(QnnModelWrapperTest, AddTensorWrapper_SharedMemoryDisabled_GraphInput_MemTypeIsRaw) {
  QnnModelWrapperTestContext ctx;
  ctx.input_index_map = {{"input0", 0}};

  ModelSettings settings{};
  settings.htp_shared_memory = false;
  auto wrapper = ctx.CreateWrapper(settings);

  QnnTensorWrapper tensor("input0", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                          QnnQuantParamsWrapper(), std::vector<uint32_t>{1, 3, 224, 224});

  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(tensor)));

  const auto& stored = wrapper->GetQnnTensorWrapper("input0");
  EXPECT_EQ(GetQnnTensorMemType(stored.GetQnnTensor()), QNN_TENSORMEMTYPE_RAW);
}

// Verifies that when htp_shared_memory is enabled, a graph input tensor
// gets mem type set to QNN_TENSORMEMTYPE_MEMHANDLE.
TEST(QnnModelWrapperTest, AddTensorWrapper_SharedMemoryEnabled_GraphInput_MemTypeIsMemHandle) {
  QnnModelWrapperTestContext ctx;
  ctx.input_index_map = {{"input0", 0}};

  ModelSettings settings{};
  settings.htp_shared_memory = true;
  auto wrapper = ctx.CreateWrapper(settings);

  QnnTensorWrapper tensor("input0", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                          QnnQuantParamsWrapper(), std::vector<uint32_t>{1, 3, 224, 224});

  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(tensor)));

  const auto& stored = wrapper->GetQnnTensorWrapper("input0");
  EXPECT_EQ(GetQnnTensorMemType(stored.GetQnnTensor()), QNN_TENSORMEMTYPE_MEMHANDLE);
}

// Verifies that when htp_shared_memory is enabled, a graph output tensor
// gets mem type set to QNN_TENSORMEMTYPE_MEMHANDLE.
TEST(QnnModelWrapperTest, AddTensorWrapper_SharedMemoryEnabled_GraphOutput_MemTypeIsMemHandle) {
  QnnModelWrapperTestContext ctx;
  ctx.output_index_map = {{"output0", 0}};

  ModelSettings settings{};
  settings.htp_shared_memory = true;
  auto wrapper = ctx.CreateWrapper(settings);

  QnnTensorWrapper tensor("output0", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32,
                          QnnQuantParamsWrapper(), std::vector<uint32_t>{1, 1000});

  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(tensor)));

  const auto& stored = wrapper->GetQnnTensorWrapper("output0");
  EXPECT_EQ(GetQnnTensorMemType(stored.GetQnnTensor()), QNN_TENSORMEMTYPE_MEMHANDLE);
}

// Verifies that when htp_shared_memory is enabled, an intermediate (native) tensor
// that is neither a graph input nor output retains QNN_TENSORMEMTYPE_RAW.
TEST(QnnModelWrapperTest, AddTensorWrapper_SharedMemoryEnabled_IntermediateTensor_MemTypeIsRaw) {
  QnnModelWrapperTestContext ctx;
  // "intermediate0" is NOT in input_index_map or output_index_map.

  ModelSettings settings{};
  settings.htp_shared_memory = true;
  auto wrapper = ctx.CreateWrapper(settings);

  QnnTensorWrapper tensor("intermediate0", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                          QnnQuantParamsWrapper(), std::vector<uint32_t>{1, 256});

  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(tensor)));

  const auto& stored = wrapper->GetQnnTensorWrapper("intermediate0");
  EXPECT_EQ(GetQnnTensorMemType(stored.GetQnnTensor()), QNN_TENSORMEMTYPE_RAW);
}

// Verifies that mem-type is determined by index-map membership, not tensor type.
// An APP_WRITE tensor that is absent from input_index_map must retain RAW even
// when htp_shared_memory is enabled.
TEST(QnnModelWrapperTest, AddTensorWrapper_SharedMemoryEnabled_AppWriteNotInInputMap_MemTypeIsRaw) {
  QnnModelWrapperTestContext ctx;
  // "unregistered" is intentionally absent from input_index_map and output_index_map.

  ModelSettings settings{};
  settings.htp_shared_memory = true;
  auto wrapper = ctx.CreateWrapper(settings);

  QnnTensorWrapper tensor("unregistered", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                          QnnQuantParamsWrapper(), std::vector<uint32_t>{1, 256});

  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(tensor)));

  const auto& stored = wrapper->GetQnnTensorWrapper("unregistered");
  EXPECT_EQ(GetQnnTensorMemType(stored.GetQnnTensor()), QNN_TENSORMEMTYPE_RAW);
}

// Verifies that when htp_shared_memory is disabled, a graph output tensor
// retains QNN_TENSORMEMTYPE_RAW.
TEST(QnnModelWrapperTest, AddTensorWrapper_SharedMemoryDisabled_GraphOutput_MemTypeIsRaw) {
  QnnModelWrapperTestContext ctx;
  ctx.output_index_map = {{"output0", 0}};

  ModelSettings settings{};
  settings.htp_shared_memory = false;
  auto wrapper = ctx.CreateWrapper(settings);

  QnnTensorWrapper tensor("output0", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32,
                          QnnQuantParamsWrapper(), std::vector<uint32_t>{1, 1000});

  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(tensor)));

  const auto& stored = wrapper->GetQnnTensorWrapper("output0");
  EXPECT_EQ(GetQnnTensorMemType(stored.GetQnnTensor()), QNN_TENSORMEMTYPE_RAW);
}

// Verifies that both graph input and output tensors get MEMHANDLE when
// htp_shared_memory is enabled, within the same wrapper instance.
TEST(QnnModelWrapperTest, AddTensorWrapper_SharedMemoryEnabled_BothInputAndOutput_MemTypeIsMemHandle) {
  QnnModelWrapperTestContext ctx;
  ctx.input_index_map = {{"input0", 0}};
  ctx.output_index_map = {{"output0", 0}};

  ModelSettings settings{};
  settings.htp_shared_memory = true;
  auto wrapper = ctx.CreateWrapper(settings);

  QnnTensorWrapper input_tensor("input0", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                                QnnQuantParamsWrapper(), std::vector<uint32_t>{1, 3, 224, 224});
  QnnTensorWrapper output_tensor("output0", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32,
                                 QnnQuantParamsWrapper(), std::vector<uint32_t>{1, 1000});

  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(input_tensor)));
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(output_tensor)));

  const auto& stored_input = wrapper->GetQnnTensorWrapper("input0");
  EXPECT_EQ(GetQnnTensorMemType(stored_input.GetQnnTensor()), QNN_TENSORMEMTYPE_MEMHANDLE);

  const auto& stored_output = wrapper->GetQnnTensorWrapper("output0");
  EXPECT_EQ(GetQnnTensorMemType(stored_output.GetQnnTensor()), QNN_TENSORMEMTYPE_MEMHANDLE);
}

// Verifies that adding a duplicate tensor (same name) returns true
// and does not overwrite the existing entry.
TEST(QnnModelWrapperTest, AddTensorWrapper_DuplicateTensor_ReturnsTrueWithoutOverwrite) {
  QnnModelWrapperTestContext ctx;
  ctx.input_index_map = {{"input0", 0}};

  ModelSettings settings{};
  settings.htp_shared_memory = false;
  auto wrapper = ctx.CreateWrapper(settings);

  QnnTensorWrapper tensor1("input0", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32,
                           QnnQuantParamsWrapper(), std::vector<uint32_t>{1, 3, 224, 224});
  ASSERT_TRUE(wrapper->AddTensorWrapper(std::move(tensor1)));

  // Attempt to add another tensor with the same name
  QnnTensorWrapper tensor2("input0", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16,
                           QnnQuantParamsWrapper(), std::vector<uint32_t>{1, 3, 112, 112});
  EXPECT_TRUE(wrapper->AddTensorWrapper(std::move(tensor2)));

  // Should still have the original data type
  const auto& stored = wrapper->GetQnnTensorWrapper("input0");
  EXPECT_EQ(stored.GetTensorDataType(), QNN_DATATYPE_FLOAT_32);
}

// Verifies that adding a tensor with an empty name returns false.
TEST(QnnModelWrapperTest, AddTensorWrapper_EmptyName_ReturnsFalse) {
  QnnModelWrapperTestContext ctx;

  ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  QnnTensorWrapper tensor("", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                          QnnQuantParamsWrapper(), std::vector<uint32_t>{1, 256});

  EXPECT_FALSE(wrapper->AddTensorWrapper(std::move(tensor)));
  // Verify the tensor was not written into the internal map.
  EXPECT_FALSE(wrapper->IsQnnTensorWrapperExist(""));
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && BUILD_QNN_EP_STATIC_LIB
