// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <array>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "test/providers/qnn/infra/qnn_fake_ort_graph.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {
namespace {

constexpr std::array<int64_t, 6> kTransposePerm{0, 1, 2, 4, 3, 5};

struct FusionValidationState {
  bool reject_transpose = false;
  std::vector<std::string> op_types;
  std::vector<uint32_t> transpose_perm;
};

FusionValidationState fusion_validation_state;

Qnn_ErrorHandle_t ValidateFusionOp(Qnn_BackendHandle_t, Qnn_OpConfig_t op_config) {
  auto& state = fusion_validation_state;
  state.op_types.emplace_back(op_config.v1.typeName);
  if (state.op_types.back() == QNN_OP_TRANSPOSE) {
    EXPECT_EQ(op_config.v1.numOfParams, 1u);
    if (op_config.v1.numOfParams != 1) {
      return static_cast<Qnn_ErrorHandle_t>(1);
    }
    const auto& param = op_config.v1.params[0];
    EXPECT_EQ(param.paramType, QNN_PARAMTYPE_TENSOR);
    if (param.paramType != QNN_PARAMTYPE_TENSOR) {
      return static_cast<Qnn_ErrorHandle_t>(1);
    }
    const auto& buffer = qnn::GetQnnTensorClientBuf(param.tensorParam);
    EXPECT_NE(buffer.data, nullptr);
    EXPECT_EQ(buffer.dataSize % sizeof(uint32_t), 0u);
    if (buffer.data == nullptr || buffer.dataSize % sizeof(uint32_t) != 0) {
      return static_cast<Qnn_ErrorHandle_t>(1);
    }
    state.transpose_perm.resize(buffer.dataSize / sizeof(uint32_t));
    std::memcpy(state.transpose_perm.data(), buffer.data, buffer.dataSize);
    if (state.reject_transpose) {
      return static_cast<Qnn_ErrorHandle_t>(1);
    }
  }
  return QNN_SUCCESS;
}

struct FusionValidationCase {
  bool reject_on_backend;
  bool reject_on_affinity;
  const char* name;
};

class QnnUnit_NodeGroupTest : public ::testing::TestWithParam<FusionValidationCase> {};

TEST_P(QnnUnit_NodeGroupTest, GetQnnNodeGroups_ValidationRegistrationsFollowAcceptance) {
  const auto& test_case = GetParam();
  OrtApiStubContext ctx;
  InstallFakeGraphApiStubs(ctx.stub_ort_api);
  ctx.stub_ort_api.ValueInfo_IsConstantInitializer = [](const OrtValueInfo* vi, bool* out) noexcept -> OrtStatus* {
    const auto& name = reinterpret_cast<const FakeValueInfo*>(vi)->name;
    *out = name == "shape1" || name == "shape2";
    return nullptr;
  };
  ctx.stub_ort_api.ValueInfo_IsGraphOutput = [](const OrtValueInfo* vi, bool* out) noexcept -> OrtStatus* {
    *out = reinterpret_cast<const FakeValueInfo*>(vi)->name == "y";
    return nullptr;
  };
  // This graph has one attribute: the rank-6 transpose permutation.
  ctx.stub_ort_api.ReadOpAttr = [](const OrtOpAttr*, OrtOpAttrType type, void* buffer,
                                   size_t buffer_size, size_t* out_size) noexcept -> OrtStatus* {
    if (type != OrtOpAttrType::ORT_OP_ATTR_INTS) {
      return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "Expected an integer-array attribute");
    }
    *out_size = sizeof(kTransposePerm);
    if (buffer != nullptr) {
      if (buffer_size < *out_size) {
        return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "Attribute buffer is too small");
      }
      std::memcpy(buffer, kTransposePerm.data(), *out_size);
    }
    return nullptr;
  };
  OrtGlobalApiOverride api_guard(&ctx.stub_ort_api);
  Ort::Logger logger{MakeNullLogger()};
  StubBackendManager backend(ctx.MakeApiPtrs(), logger);
  backend.BackendType() = qnn::QnnBackendType::HTP;
  backend.QnnInterface().backendValidateOpConfig = ValidateFusionOp;
  fusion_validation_state = {};
  fusion_validation_state.reject_transpose = test_case.reject_on_backend;

  // Rank6ToRank5Fusion drops the leading unit dimension from r1 and t.
  // The shape/permutation deliberately do not match SpaceToDepth or ChannelShuffle.
  FakeValueInfo x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {2, 3, 4, 5, 6}};
  FakeValueInfo r1{"r1", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 2, 3, 4, 5, 6}};
  FakeValueInfo t{"t", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 2, 3, 5, 4, 6}};
  FakeValueInfo y{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {2, 3, 5, 4, 6}};
  FakeValueInfo shape1{"shape1", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, {6}};
  FakeValueInfo shape2{"shape2", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, {5}};
  FakeOpAttr perm_attr;
  perm_attr.name = "perm";
  perm_attr.type = OrtOpAttrType::ORT_OP_ATTR_INTS;
  FakeGraph graph{{
                      {"reshape1", "Reshape", "", 13, {&x, &shape1}, {&r1}, {}, 0},
                      {"transpose", "Transpose", "", 13, {&r1}, {&t}, {{"perm", &perm_attr}}, 1},
                      {"reshape2", "Reshape", "", 13, {&t, &shape2}, {&y}, {}, 2},
                  },
                  {&x},
                  {&y},
                  {&shape1, &shape2}};
  x.consumers = {&graph.nodes[0]};
  r1.producer = &graph.nodes[0];
  r1.consumers = {&graph.nodes[1]};
  t.producer = &graph.nodes[1];
  t.consumers = {&graph.nodes[2]};
  y.producer = &graph.nodes[2];

  std::vector<std::unique_ptr<OrtNodeUnit>> node_units;
  std::unordered_map<const OrtNode*, const OrtNodeUnit*> node_unit_map;
  for (const auto& node : graph.nodes) {
    node_units.push_back(std::make_unique<OrtNodeUnit>(node.AsNode(), ctx.stub_ort_api));
    node_unit_map.emplace(node.AsNode(), node_units.back().get());
  }

  qnn::GraphInputOutputInfo inputs;
  inputs.names = {"x"};
  inputs.indices = {{"x", 0}};
  qnn::GraphInputOutputInfo outputs;
  outputs.names = {"y"};
  outputs.indices = {{"y", 0}};
  qnn::ModelSettings settings;
  if (test_case.reject_on_affinity) {
    settings.op_affinity.SeedDefaultIfAbsent("Transpose", qnn::QnnBackendType::CPU);
  }
  qnn::QnnModelWrapper wrapper(*graph.AsGraph(), ctx.MakeApiPtrs(), logger, *backend.Get(),
                               inputs, outputs, settings);
  ASSERT_TRUE(wrapper.AddTensorWrapper(qnn::QnnTensorWrapper(
      "preexisting", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
      qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{1})));

  std::vector<std::unique_ptr<qnn::IQnnNodeGroup>> groups;
  ASSERT_TRUE(qnn::GetQnnNodeGroups(groups, wrapper, node_unit_map, node_units.size(), logger).IsOK());
  const std::vector<std::string> expected_ops = test_case.reject_on_backend
                                                    ? std::vector<std::string>{QNN_OP_RESHAPE, QNN_OP_TRANSPOSE}
                                                    : std::vector<std::string>{QNN_OP_RESHAPE, QNN_OP_TRANSPOSE, QNN_OP_RESHAPE};
  EXPECT_EQ(fusion_validation_state.op_types, expected_ops);
  EXPECT_EQ(fusion_validation_state.transpose_perm, (std::vector<uint32_t>{0, 1, 3, 2, 4}));
  EXPECT_TRUE(wrapper.IsQnnTensorWrapperExist("preexisting"));

  if (!test_case.reject_on_backend && !test_case.reject_on_affinity) {
    ASSERT_EQ(groups.size(), 1u);
    EXPECT_EQ(groups[0]->Type(), "Rank6ToRank5Fusion");
    EXPECT_EQ(wrapper.GetModelTensorsMap().size(), 5u);
    EXPECT_EQ(wrapper.GetQnnTensorWrapper("r1").GetTensorDims(), (std::vector<uint32_t>{2, 3, 4, 5, 6}));
    return;
  }

  ASSERT_EQ(groups.size(), 3u);
  for (const auto& group : groups) {
    EXPECT_EQ(group->Type(), "OrtNodeUnit");
  }
  EXPECT_EQ(wrapper.GetModelTensorsMap().size(), 1u);

  // Normal Transpose construction must register the original rank-6 tensors
  // and permutation under the names used by the rejected rank-5 fusion.
  ASSERT_TRUE(groups[1]->AddToModelBuilder(wrapper, logger).IsOK());
  EXPECT_EQ(wrapper.GetQnnTensorWrapper("r1").GetTensorDims(), (std::vector<uint32_t>{1, 2, 3, 4, 5, 6}));
  EXPECT_EQ(wrapper.GetQnnTensorWrapper("t").GetTensorDims(), (std::vector<uint32_t>{1, 2, 3, 5, 4, 6}));
  qnn::QnnParamWrapper perm_key(1, "transpose", QNN_OP_TRANSPOSE_PARAM_PERM,
                                std::vector<uint32_t>{6}, std::vector<uint32_t>{0, 1, 2, 4, 3, 5});
  fusion_validation_state.reject_transpose = false;
  ASSERT_TRUE(wrapper.CreateQnnNode("check_fallback", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_TRANSPOSE,
                                    {"r1"}, {"t"}, {perm_key.GetParamTensorName()}, /*do_op_validation=*/true));
  EXPECT_EQ(fusion_validation_state.transpose_perm, (std::vector<uint32_t>{0, 1, 2, 4, 3, 5}));
}

INSTANTIATE_TEST_SUITE_P(
    QnnUnit_FusionValidation, QnnUnit_NodeGroupTest,
    ::testing::Values(FusionValidationCase{true, false, "BackendRejected"},
                      FusionValidationCase{false, false, "Accepted"},
                      FusionValidationCase{false, true, "AffinityRejected"}),
    [](const ::testing::TestParamInfo<FusionValidationCase>& info) { return info.param.name; });

}  // namespace
}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
