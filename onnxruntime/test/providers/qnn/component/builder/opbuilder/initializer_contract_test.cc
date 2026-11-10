// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {
namespace {

// External data keeps the declared tensor shape intact while varying the
// unpacked byte count. This bypasses model-loader validation and reaches the
// real op builder's consumer-side guard without a QNN backend or ORT session.
struct ExternalInitializer {
  const OrtValueInfo* value_info = nullptr;
  std::filesystem::path path;
  size_t byte_count = 0;
  size_t reads = 0;
  size_t null_queries = 0;
};

ExternalInitializer* external_initializer = nullptr;

OrtStatus* GetExternalInitializer(const OrtValueInfo* value_info,
                                  OrtExternalInitializerInfo** info) noexcept {
  if (value_info == nullptr) {
    ++external_initializer->null_queries;
    return Ort::Status("Unexpected null initializer in contract test", ORT_FAIL).release();
  }
  *info = nullptr;
  if (value_info == external_initializer->value_info) {
    ++external_initializer->reads;
    *info = reinterpret_cast<OrtExternalInitializerInfo*>(external_initializer);
  }
  return nullptr;
}

OrtStatus* GetTensorDataAllowEmpty(const OrtValue*, const void** data) noexcept {
  auto it = g_mock_init_reg.vi_to_spec.find(g_mock_chain_vi);
  if (it == g_mock_init_reg.vi_to_spec.end()) {
    return MakeMockChainMissStatus();
  }
  *data = it->second.raw_bytes.data();
  return nullptr;
}

struct InitializerCase {
  const char* op;
  const char* input;
  ONNXTensorElementDataType type;
  size_t required_bytes;
  const char* error;
  bool support_check = false;
};

std::vector<InitializerCase> InitializerCases() {
  return {
      {"Clip", "min", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 4, "initializer byte count"},
      {"Clip", "max", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 8, "initializer byte count"},
      {"Expand", "shape", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 16, "initializer byte count"},
      {"TopK", "k", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 8, "scalar int64 initializer"},
      {"Range", "start", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 4, "scalar initializers"},
      {"Range", "limit", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, 4, "scalar initializers"},
      {"Range", "delta", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 8, "scalar initializers"},
      {"NonMaxSuppression", "max_output_boxes_per_class", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 8,
       "scalar int64 initializer"},
      {"NonMaxSuppression", "max_output_boxes_per_class", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 8,
       "scalar int64 initializer", true},
      {"NonMaxSuppression", "iou_threshold", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 4, "scalar float initializer"},
      {"NonMaxSuppression", "score_threshold", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 4, "scalar float initializer"},
      {"STFT", "frame_step", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, 4, "initializer byte count"},
      {"STFT", "frame_length", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, 4, "initializer byte count"},
      {"STFT", "frame_step", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 8, "initializer byte count"},
      {"STFT", "frame_length", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 8, "initializer byte count"},
  };
}

std::string InitializerCaseName(const InitializerCase& test_case) {
  const char* type = test_case.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT   ? "Float"
                     : test_case.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ? "Int32"
                                                                             : "Int64";
  return std::string(test_case.op) + "_" + test_case.input + "_" + type +
         (test_case.support_check ? "_SupportCheck" : "_Build");
}

class InitializerContractTest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_mock_init_reg.clear();
    external_initializer = &external_;
    SetupMockInitRegistryStubs(context_);
    context_.stub_ort_api.GetTensorData = GetTensorDataAllowEmpty;
    context_.stub_ort_api.ValueInfo_GetExternalInitializerInfo = GetExternalInitializer;
    context_.stub_ort_api.ExternalInitializerInfo_GetFilePath =
        [](const OrtExternalInitializerInfo* info) noexcept -> const ORTCHAR_T* {
      return reinterpret_cast<const ExternalInitializer*>(info)->path.c_str();
    };
    context_.stub_ort_api.ExternalInitializerInfo_GetFileOffset =
        [](const OrtExternalInitializerInfo*) noexcept -> int64_t { return 0; };
    context_.stub_ort_api.ExternalInitializerInfo_GetByteSize =
        [](const OrtExternalInitializerInfo* info) noexcept -> size_t {
      return reinterpret_cast<const ExternalInitializer*>(info)->byte_count;
    };
    context_.stub_ort_api.ReleaseExternalInitializerInfo = [](OrtExternalInitializerInfo*) noexcept {};
    // Only the NMS support-check cases proceed to backend validation. Stub that
    // final step so a valid buffer can prove it passes the consumer's guard.
    context_.qnn_interface.backendValidateOpConfig =
        [](Qnn_BackendHandle_t, Qnn_OpConfig_t) -> Qnn_ErrorHandle_t { return QNN_SUCCESS; };
  }

  void TearDown() override {
    std::error_code error;
    if (!external_.path.empty()) {
      std::filesystem::remove(external_.path, error);
    }
    external_initializer = nullptr;
  }

  template <typename T>
  void AddValues(const char* name, ONNXTensorElementDataType type,
                 std::vector<int64_t> dims, std::vector<T> values) {
    MockInitSpec spec;
    spec.elem_type = type;
    spec.dims = std::move(dims);
    spec.raw_bytes.resize(values.size() * sizeof(T));
    if (!spec.raw_bytes.empty()) {
      std::memcpy(spec.raw_bytes.data(), values.data(), spec.raw_bytes.size());
    }
    g_mock_init_reg.Add(name, std::move(spec));
  }

  void AddScalar(const char* name, ONNXTensorElementDataType type, int64_t value) {
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      AddValues<float>(name, type, {}, {static_cast<float>(value)});
    } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      AddValues<int32_t>(name, type, {}, {static_cast<int32_t>(value)});
    } else {
      AddValues<int64_t>(name, type, {}, {value});
    }
  }

  void Prepare(const InitializerCase& test_case) {
    const auto make_io = [](const char* name, ONNXTensorElementDataType type, std::vector<int64_t> shape) {
      return MakeMockIODef(name, type, std::move(shape));
    };
    const auto float_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    const auto int64_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    const std::string op = test_case.op;
    if (op == "Clip") {
      inputs_ = {make_io("data", test_case.type, {1, 4}), make_io("min", test_case.type, {}),
                 make_io("max", test_case.type, {})};
      outputs_ = {make_io("output", test_case.type, {1, 4})};
      AddScalar("min", test_case.type, -1);
      AddScalar("max", test_case.type, 1);
    } else if (op == "Expand") {
      inputs_ = {make_io("data", float_type, {1, 4}), make_io("shape", int64_type, {2})};
      outputs_ = {make_io("output", float_type, {2, 4})};
      AddValues<int64_t>("shape", int64_type, {2}, {2, 4});
    } else if (op == "TopK") {
      inputs_ = {make_io("data", float_type, {1, 4}), make_io("k", int64_type, {1})};
      outputs_ = {make_io("values", float_type, {1, 2}), make_io("indices", int64_type, {1, 2})};
      AddValues<int64_t>("k", int64_type, {1}, {2});
    } else if (op == "Range") {
      inputs_ = {make_io("start", test_case.type, {}), make_io("limit", test_case.type, {}),
                 make_io("delta", test_case.type, {})};
      outputs_ = {make_io("output", test_case.type, {4})};
      AddScalar("start", test_case.type, 0);
      AddScalar("limit", test_case.type, 4);
      AddScalar("delta", test_case.type, 1);
    } else if (op == "NonMaxSuppression") {
      inputs_ = {make_io("boxes", float_type, {1, 4, 4}), make_io("scores", float_type, {1, 1, 4}),
                 make_io("max_output_boxes_per_class", int64_type, {}),
                 make_io("iou_threshold", float_type, {}), make_io("score_threshold", float_type, {})};
      outputs_ = {make_io("output", int64_type, {2, 3})};
      AddScalar("max_output_boxes_per_class", int64_type, 2);
      g_mock_init_reg.AddScalarFloat("iou_threshold", 0.5f);
      g_mock_init_reg.AddScalarFloat("score_threshold", 0.25f);
    } else {
      inputs_ = {make_io("signal", float_type, {1, 128, 1}), make_io("frame_step", test_case.type, {}),
                 make_io("window", float_type, {16}), make_io("frame_length", test_case.type, {})};
      outputs_ = {make_io("output", float_type, {1, 15, 9, 2})};
      AddScalar("frame_step", test_case.type, 8);
      AddScalar("frame_length", test_case.type, 16);
      AddValues<float>("window", float_type, {16}, std::vector<float>(16, 1.0f));
    }
  }

  void UseExternalBytes(const char* name, size_t byte_count) {
    external_.value_info = g_mock_init_reg.name_to_vi.at(name);
    external_.byte_count = byte_count;
    const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
    external_.path = std::filesystem::temp_directory_path() /
                     ("qnn_initializer_contract_" + std::to_string(timestamp) + ".bin");
    auto bytes = g_mock_init_reg.vi_to_spec.at(external_.value_info).raw_bytes;
    bytes.resize(byte_count);
    std::ofstream stream(external_.path, std::ios::binary);
    ASSERT_TRUE(stream.is_open());
    if (!bytes.empty()) {
      stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    ASSERT_TRUE(stream.good());
  }

  void MakeDynamic(const char* name) {
    const auto* value_info = g_mock_init_reg.name_to_vi.at(name);
    auto& initializers = g_mock_init_reg.vi_array;
    initializers.erase(std::remove(initializers.begin(), initializers.end(), value_info), initializers.end());
    g_mock_init_reg.name_to_vi.erase(name);
  }

  Ort::Status Run(const InitializerCase& test_case, bool support_check = false) {
    const auto* builder = qnn::GetOpBuilder(test_case.op);
    if (builder == nullptr) {
      return Ort::Status("Missing op builder in initializer contract test", ORT_FAIL);
    }
    wrapper_ = context_.CreateWrapper(qnn::ModelSettings{},
                                      std::string(test_case.op) == "NonMaxSuppression"
                                          ? qnn::QnnBackendType::CPU
                                          : qnn::QnnBackendType::HTP);
    auto node = MakeMockNodeUnit(test_case.op, inputs_, outputs_, "initializer_contract", "", 17);
    if (support_check) {
      return builder->IsOpSupported(*wrapper_, node, context_.ort_logger);
    }
    return builder->AddToModelBuilder(*wrapper_, node, context_.ort_logger, false);
  }

  OpBuilderTestContext context_;
  ExternalInitializer external_;
  std::unique_ptr<qnn::QnnModelWrapper> wrapper_;
  std::vector<OrtNodeUnitIODef> inputs_;
  std::vector<OrtNodeUnitIODef> outputs_;
};

class QnnUnit_InitializerByteCountTest : public InitializerContractTest,
                                         public ::testing::WithParamInterface<std::pair<InitializerCase, size_t>> {};

TEST_P(QnnUnit_InitializerByteCountTest, RejectsMismatchedExternalBytesBeforeRead) {
  const auto& [test_case, byte_count] = GetParam();
  Prepare(test_case);
  UseExternalBytes(test_case.input, byte_count);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  const auto status = Run(test_case, test_case.support_check);
  ASSERT_FALSE(status.IsOK());
  EXPECT_NE(std::string(status.GetErrorMessage()).find(test_case.error), std::string::npos)
      << status.GetErrorMessage();
  EXPECT_GT(external_.reads, 0u);
  EXPECT_EQ(external_.null_queries, 0u);
}

std::vector<std::pair<InitializerCase, size_t>> ByteCountCases() {
  std::vector<std::pair<InitializerCase, size_t>> cases;
  for (const auto& test_case : InitializerCases()) {
    for (size_t byte_count : {size_t{0}, test_case.required_bytes - 1, test_case.required_bytes + 1}) {
      cases.emplace_back(test_case, byte_count);
    }
  }
  return cases;
}

INSTANTIATE_TEST_SUITE_P(
    AllBuilders, QnnUnit_InitializerByteCountTest, ::testing::ValuesIn(ByteCountCases()),
    [](const ::testing::TestParamInfo<std::pair<InitializerCase, size_t>>& info) {
      return InitializerCaseName(info.param.first) + "_Bytes" + std::to_string(info.param.second);
    });

class QnnUnit_InitializerValidBytesTest : public InitializerContractTest,
                                          public ::testing::WithParamInterface<InitializerCase> {};

TEST_P(QnnUnit_InitializerValidBytesTest, AcceptsValidInitializerBytes) {
  const auto& test_case = GetParam();
  Prepare(test_case);
  UseExternalBytes(test_case.input, test_case.required_bytes);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  const auto status = Run(test_case, test_case.support_check);
  EXPECT_TRUE(status.IsOK()) << status.GetErrorMessage();
  EXPECT_GT(external_.reads, 0u);
}

INSTANTIATE_TEST_SUITE_P(
    AllBuilders, QnnUnit_InitializerValidBytesTest, ::testing::ValuesIn(InitializerCases()),
    [](const ::testing::TestParamInfo<InitializerCase>& info) { return InitializerCaseName(info.param); });

TEST_F(InitializerContractTest, ClipRejectsZeroElementEmbeddedInitializer) {
  const InitializerCase test_case{"Clip", "min", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 4, "initializer byte count"};
  Prepare(test_case);
  auto& spec = g_mock_init_reg.vi_to_spec.at(g_mock_init_reg.name_to_vi.at("min"));
  spec.dims = {0};
  spec.raw_bytes.clear();
  inputs_[1].shape = std::vector<int64_t>{0};
  const auto status = Run(test_case);
  ASSERT_FALSE(status.IsOK());
  EXPECT_NE(std::string(status.GetErrorMessage()).find(test_case.error), std::string::npos)
      << status.GetErrorMessage();
}

class QnnUnit_StftInitializerContractTest : public InitializerContractTest,
                                            public ::testing::WithParamInterface<InitializerCase> {};

TEST_P(QnnUnit_StftInitializerContractTest, RejectsDynamicParameterBeforeUnpackingNull) {
  const auto& test_case = GetParam();
  Prepare(test_case);
  MakeDynamic(test_case.input);
  const auto status = Run(test_case);
  ASSERT_FALSE(status.IsOK());
  const std::string expected = std::string("STFT ") + test_case.input + " must be a constant initializer";
  EXPECT_NE(std::string(status.GetErrorMessage()).find(expected), std::string::npos)
      << status.GetErrorMessage();
  EXPECT_EQ(external_.null_queries, 0u);
}

TEST_P(QnnUnit_StftInitializerContractTest, RejectsNonPositiveOrOverflowingScalarWithoutThrowing) {
  const auto& test_case = GetParam();
  for (int64_t value : std::vector<int64_t>{-1, 0, 4294967296LL, 4294967304LL, std::numeric_limits<int64_t>::max()}) {
    if (test_case.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 && value > std::numeric_limits<int32_t>::max()) {
      continue;
    }
    SCOPED_TRACE(value);
    g_mock_init_reg.clear();
    Prepare(test_case);
    auto& spec = g_mock_init_reg.vi_to_spec.at(g_mock_init_reg.name_to_vi.at(test_case.input));
    if (test_case.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      const auto scalar = static_cast<int32_t>(value);
      std::memcpy(spec.raw_bytes.data(), &scalar, sizeof(scalar));
    } else {
      std::memcpy(spec.raw_bytes.data(), &value, sizeof(value));
    }
    Ort::Status status;
    ASSERT_NO_THROW(status = Run(test_case));
    ASSERT_FALSE(status.IsOK());
    EXPECT_NE(std::string(status.GetErrorMessage()).find("positive and fit in uint32"), std::string::npos)
        << status.GetErrorMessage();
  }
}

INSTANTIATE_TEST_SUITE_P(
    BothScalarTypes, QnnUnit_StftInitializerContractTest,
    ::testing::Values(
        InitializerCase{"STFT", "frame_step", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, 4, "initializer byte count"},
        InitializerCase{"STFT", "frame_length", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, 4, "initializer byte count"},
        InitializerCase{"STFT", "frame_step", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 8, "initializer byte count"},
        InitializerCase{"STFT", "frame_length", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 8, "initializer byte count"}),
    [](const ::testing::TestParamInfo<InitializerCase>& info) { return InitializerCaseName(info.param); });

}  // namespace
}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
