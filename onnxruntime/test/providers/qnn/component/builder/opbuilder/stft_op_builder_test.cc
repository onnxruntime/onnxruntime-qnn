// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <algorithm>
#include <cstring>
#include <limits>

#include "gtest/gtest.h"

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {
namespace {

struct StftParameterCase {
  const char* input;
  ONNXTensorElementDataType type;
};

static size_t g_null_initializer_queries = 0;

class QnnUnit_StftInitializerContractTest : public ::testing::TestWithParam<StftParameterCase> {
 protected:
  void SetUp() override {
    g_mock_init_reg.clear();
    g_null_initializer_queries = 0;
    SetupMockInitRegistryStubs(context_);
    context_.stub_ort_api.GetTensorData = [](const OrtValue*, const void** data) noexcept -> OrtStatus* {
      const auto it = g_mock_init_reg.vi_to_spec.find(g_mock_chain_vi);
      if (it == g_mock_init_reg.vi_to_spec.end()) return MakeMockChainMissStatus();
      *data = it->second.raw_bytes.data();
      return nullptr;
    };
    context_.stub_ort_api.ValueInfo_GetExternalInitializerInfo =
        [](const OrtValueInfo* value_info, OrtExternalInitializerInfo** info) noexcept -> OrtStatus* {
      if (value_info == nullptr) {
        ++g_null_initializer_queries;
        return Ort::Status("Unexpected null STFT initializer", ORT_FAIL).release();
      }
      return StubMockValueInfoGetExternal(value_info, info);
    };
    const auto type = GetParam().type;
    inputs_ = {MakeMockIODef("signal", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{1, 128, 1}),
               MakeMockIODef("frame_step", type, std::vector<int64_t>{}),
               MakeMockIODef("window", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{16}),
               MakeMockIODef("frame_length", type, std::vector<int64_t>{})};
    outputs_ = {MakeMockIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{1, 15, 9, 2})};
    AddScalar("frame_step", 8);
    AddScalar("frame_length", 16);
    MockInitSpec window;
    window.elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    window.dims = {16};
    const std::vector<float> values(16, 1.0f);
    window.raw_bytes.resize(values.size() * sizeof(float));
    std::memcpy(window.raw_bytes.data(), values.data(), window.raw_bytes.size());
    g_mock_init_reg.Add("window", std::move(window));
    wrapper_ = context_.CreateWrapper(qnn::ModelSettings{}, qnn::QnnBackendType::HTP);
  }

  void AddScalar(const char* name, int64_t value) {
    MockInitSpec spec;
    spec.elem_type = GetParam().type;
    if (spec.elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      const auto scalar = static_cast<int32_t>(value);
      spec.raw_bytes.resize(sizeof(scalar));
      std::memcpy(spec.raw_bytes.data(), &scalar, sizeof(scalar));
    } else {
      spec.raw_bytes.resize(sizeof(value));
      std::memcpy(spec.raw_bytes.data(), &value, sizeof(value));
    }
    g_mock_init_reg.Add(name, std::move(spec));
  }

  Ort::Status Run() {
    const auto* builder = qnn::GetOpBuilder("STFT");
    if (builder == nullptr) return Ort::Status("STFT builder not registered", ORT_FAIL);
    auto node = MakeMockNodeUnit("STFT", inputs_, outputs_, "stft_initializer_contract", "", 17);
    return builder->AddToModelBuilder(*wrapper_, node, context_.ort_logger, false);
  }

  OpBuilderTestContext context_;
  std::unique_ptr<qnn::QnnModelWrapper> wrapper_;
  std::vector<OrtNodeUnitIODef> inputs_;
  std::vector<OrtNodeUnitIODef> outputs_;
};

TEST_P(QnnUnit_StftInitializerContractTest, AcceptsValidConstantParameters) {
  const auto status = Run();
  EXPECT_TRUE(status.IsOK()) << status.GetErrorMessage();
}

TEST_P(QnnUnit_StftInitializerContractTest, RejectsDynamicParameterBeforeUnpackingNull) {
  const auto& test_case = GetParam();
  const auto* value_info = g_mock_init_reg.name_to_vi.at(test_case.input);
  auto& initializers = g_mock_init_reg.vi_array;
  initializers.erase(std::remove(initializers.begin(), initializers.end(), value_info), initializers.end());
  g_mock_init_reg.name_to_vi.erase(test_case.input);
  const auto status = Run();
  ASSERT_FALSE(status.IsOK());
  const std::string expected = std::string("STFT ") + test_case.input + " must be a constant initializer";
  EXPECT_NE(std::string(status.GetErrorMessage()).find(expected), std::string::npos) << status.GetErrorMessage();
  EXPECT_EQ(g_null_initializer_queries, 0u);
}

TEST_P(QnnUnit_StftInitializerContractTest, RejectsNonScalarInitializer) {
  const auto& test_case = GetParam();
  auto& spec = g_mock_init_reg.vi_to_spec.at(g_mock_init_reg.name_to_vi.at(test_case.input));
  const size_t input_index = std::string(test_case.input) == "frame_step" ? 1 : 3;
  for (const int64_t element_count : {0, 2}) {
    SCOPED_TRACE(element_count);
    spec.dims = {element_count};
    spec.raw_bytes.resize(static_cast<size_t>(element_count) *
                          (test_case.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ? sizeof(int32_t) : sizeof(int64_t)));
    inputs_[input_index].shape = spec.dims;
    const auto status = Run();
    ASSERT_FALSE(status.IsOK());
    EXPECT_NE(std::string(status.GetErrorMessage()).find("expected size"), std::string::npos) << status.GetErrorMessage();
  }
}

TEST_P(QnnUnit_StftInitializerContractTest, RejectsNonPositiveOrOverflowingScalarWithoutThrowing) {
  const auto& test_case = GetParam();
  auto& spec = g_mock_init_reg.vi_to_spec.at(g_mock_init_reg.name_to_vi.at(test_case.input));
  for (const int64_t value : std::vector<int64_t>{-1, 0, 4294967296LL, 4294967304LL, std::numeric_limits<int64_t>::max()}) {
    if (test_case.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 && value > std::numeric_limits<int32_t>::max()) continue;
    SCOPED_TRACE(value);
    if (test_case.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      const auto scalar = static_cast<int32_t>(value);
      std::memcpy(spec.raw_bytes.data(), &scalar, sizeof(scalar));
    } else {
      std::memcpy(spec.raw_bytes.data(), &value, sizeof(value));
    }
    Ort::Status status;
    ASSERT_NO_THROW(status = Run());
    ASSERT_FALSE(status.IsOK());
    EXPECT_NE(std::string(status.GetErrorMessage()).find("positive and fit in uint32"), std::string::npos)
        << status.GetErrorMessage();
  }
}

INSTANTIATE_TEST_SUITE_P(
    BothScalarTypes, QnnUnit_StftInitializerContractTest,
    ::testing::Values(StftParameterCase{"frame_step", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32},
                      StftParameterCase{"frame_length", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32},
                      StftParameterCase{"frame_step", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64},
                      StftParameterCase{"frame_length", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64}),
    [](const ::testing::TestParamInfo<StftParameterCase>& info) {
      return std::string(info.param.input) + (info.param.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ? "_Int32" : "_Int64");
    });

}  // namespace
}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
