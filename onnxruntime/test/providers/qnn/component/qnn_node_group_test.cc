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

// Shared infrastructure for node-group tests. Each test or derived fixture
// supplies its own graph, model settings, and backend callbacks.
class QnnUnit_NodeGroupTest : public ::testing::Test {
 protected:
  void SetUp() override {
    InstallFakeGraphApiStubs(ctx_.stub_ort_api);
    api_guard_ = std::make_unique<OrtGlobalApiOverride>(&ctx_.stub_ort_api);
    backend_ = std::make_unique<StubBackendManager>(ctx_.MakeApiPtrs(), logger_);
  }

  // These dependencies outlive graph/model objects owned by derived fixtures.
  OrtApiStubContext ctx_;
  std::unique_ptr<OrtGlobalApiOverride> api_guard_;
  Ort::Logger logger_{MakeNullLogger()};
  std::unique_ptr<StubBackendManager> backend_;
};

constexpr std::array<int64_t, 6> kTransposePerm{0, 1, 2, 4, 3, 5};

struct TransposeValidationMetadata {
  std::string param_name;
  std::vector<uint32_t> param_dims;
  std::vector<uint32_t> perm;
  std::vector<uint32_t> output_dims;
};

struct FusionValidationState {
  bool reject_transpose = false;
  bool reject_first_transpose = false;
  std::vector<std::string> op_types;
  std::vector<uint32_t> transpose_perm;
  std::vector<TransposeValidationMetadata> transposes;
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
    EXPECT_EQ(op_config.v1.numOfOutputs, 1u);
    if (op_config.v1.numOfOutputs != 1) {
      return static_cast<Qnn_ErrorHandle_t>(1);
    }
    const auto& output = op_config.v1.outputTensors[0];
    const auto* param_dims = qnn::GetQnnTensorDims(param.tensorParam);
    const auto* output_dims = qnn::GetQnnTensorDims(output);
    state.transposes.push_back({qnn::GetQnnTensorName(param.tensorParam),
                                {param_dims, param_dims + qnn::GetQnnTensorRank(param.tensorParam)},
                                state.transpose_perm,
                                {output_dims, output_dims + qnn::GetQnnTensorRank(output)}});
    if (state.reject_transpose || (state.reject_first_transpose && state.transposes.size() == 1)) {
      return static_cast<Qnn_ErrorHandle_t>(1);
    }
  }
  return QNN_SUCCESS;
}

struct FusionValidationCase {
  bool reject_on_backend;
  bool reject_on_affinity;
  const char* name;
  bool accept_later_candidate = false;
};

class QnnUnit_NodeGroupFusionValidationTest : public QnnUnit_NodeGroupTest,
                                              public ::testing::WithParamInterface<FusionValidationCase> {
 protected:
  void SetUp() override {
    QnnUnit_NodeGroupTest::SetUp();
    ctx_.stub_ort_api.ValueInfo_IsConstantInitializer = [](const OrtValueInfo* vi, bool* out) noexcept -> OrtStatus* {
      const auto& name = reinterpret_cast<const FakeValueInfo*>(vi)->name;
      *out = name == "shape1" || name == "shape2";
      return nullptr;
    };
    ctx_.stub_ort_api.ValueInfo_IsGraphOutput = [](const OrtValueInfo* vi, bool* out) noexcept -> OrtStatus* {
      *out = reinterpret_cast<const FakeValueInfo*>(vi)->name == "y";
      return nullptr;
    };
    // This graph has one attribute: the rank-6 transpose permutation.
    ctx_.stub_ort_api.ReadOpAttr = [](const OrtOpAttr*, OrtOpAttrType type, void* buffer,
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
    backend_->BackendType() = qnn::QnnBackendType::HTP;
    backend_->QnnInterface().backendValidateOpConfig = ValidateFusionOp;
    fusion_validation_state = {};

    perm_attr_.name = "perm";
    perm_attr_.type = OrtOpAttrType::ORT_OP_ATTR_INTS;
    x_.consumers = {&graph_.nodes[0]};
    r1_.producer = &graph_.nodes[0];
    r1_.consumers = {&graph_.nodes[1]};
    t_.producer = &graph_.nodes[1];
    t_.consumers = {&graph_.nodes[2]};
    y_.producer = &graph_.nodes[2];

    inputs_.names = {"x"};
    inputs_.indices = {{"x", 0}};
    outputs_.names = {"y"};
    outputs_.indices = {{"y", 0}};
  }

  // Called after each test configures its graph, validation failures, and affinity.
  void InitializeFusionModel() {
    for (const auto& node : graph_.nodes) {
      node_units_.push_back(std::make_unique<OrtNodeUnit>(node.AsNode(), ctx_.stub_ort_api));
      node_unit_map_.emplace(node.AsNode(), node_units_.back().get());
    }
    wrapper_ = std::make_unique<qnn::QnnModelWrapper>(*graph_.AsGraph(), ctx_.MakeApiPtrs(), logger_,
                                                      *backend_->Get(), inputs_, outputs_, settings_);
    ASSERT_TRUE(wrapper_->AddTensorWrapper(qnn::QnnTensorWrapper(
        "preexisting", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
        qnn::QnnQuantParamsWrapper(), std::vector<uint32_t>{1})));
  }

  void CheckStandaloneTransposeRegistrations() {
    // Normal Transpose construction must register the original rank-6 tensors
    // and permutation under the names used by the rejected rank-5 fusion.
    ASSERT_TRUE(groups_[1]->AddToModelBuilder(*wrapper_, logger_).IsOK());
    EXPECT_EQ(wrapper_->GetQnnTensorWrapper("r1").GetTensorDims(), (std::vector<uint32_t>{1, 2, 3, 4, 5, 6}));
    EXPECT_EQ(wrapper_->GetQnnTensorWrapper("t").GetTensorDims(), (std::vector<uint32_t>{1, 2, 3, 5, 4, 6}));
    qnn::QnnParamWrapper perm_key(1, "transpose", QNN_OP_TRANSPOSE_PARAM_PERM,
                                  std::vector<uint32_t>{6}, std::vector<uint32_t>{0, 1, 2, 4, 3, 5});
    fusion_validation_state.reject_transpose = false;
    ASSERT_TRUE(wrapper_->CreateQnnNode("check_fallback", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_TRANSPOSE,
                                        {"r1"}, {"t"}, {perm_key.GetParamTensorName()}, /*do_op_validation=*/true));
    EXPECT_EQ(fusion_validation_state.transpose_perm, (std::vector<uint32_t>{0, 1, 2, 4, 3, 5}));
  }

  // Rank6ToRank5Fusion drops the leading unit dimension from r1 and t.
  // The shape/permutation deliberately do not match SpaceToDepth or ChannelShuffle.
  FakeValueInfo x_{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {2, 3, 4, 5, 6}};
  FakeValueInfo r1_{"r1", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 2, 3, 4, 5, 6}};
  FakeValueInfo t_{"t", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 2, 3, 5, 4, 6}};
  FakeValueInfo y_{"y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {2, 3, 5, 4, 6}};
  FakeValueInfo shape1_{"shape1", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, {6}};
  FakeValueInfo shape2_{"shape2", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, {5}};
  FakeOpAttr perm_attr_;
  FakeGraph graph_{{
                       {"reshape1", "Reshape", "", 13, {&x_, &shape1_}, {&r1_}, {}, 0},
                       {"transpose", "Transpose", "", 13, {&r1_}, {&t_}, {{"perm", &perm_attr_}}, 1},
                       {"reshape2", "Reshape", "", 13, {&t_, &shape2_}, {&y_}, {}, 2},
                   },
                   {&x_},
                   {&y_},
                   {&shape1_, &shape2_}};
  std::vector<std::unique_ptr<OrtNodeUnit>> node_units_;
  std::unordered_map<const OrtNode*, const OrtNodeUnit*> node_unit_map_;
  qnn::GraphInputOutputInfo inputs_;
  qnn::GraphInputOutputInfo outputs_;
  qnn::ModelSettings settings_;
  std::unique_ptr<qnn::QnnModelWrapper> wrapper_;
  std::vector<std::unique_ptr<qnn::IQnnNodeGroup>> groups_;
};

TEST_P(QnnUnit_NodeGroupFusionValidationTest, GetQnnNodeGroups_ValidationRegistrationsFollowAcceptance) {
  const auto& test_case = GetParam();
  fusion_validation_state.reject_transpose = test_case.reject_on_backend;
  fusion_validation_state.reject_first_transpose = test_case.accept_later_candidate;
  if (test_case.reject_on_affinity) {
    settings_.op_affinity.SeedDefaultIfAbsent("Transpose", qnn::QnnBackendType::CPU);
  }
  if (test_case.accept_later_candidate) {
    // Moving the unit dimension makes ReshapeTransposeFusion a later match.
    x_.shape = {2, 1, 3, 4, 5, 6};
    // Rank6ToRank5Fusion keys its perm by node name; ReshapeTransposeFusion
    // keys it by output name. Use "t" for both so their parameter names collide.
    graph_.nodes[1].name = "t";
  }
  ASSERT_NO_FATAL_FAILURE(InitializeFusionModel());
  ASSERT_TRUE(qnn::GetQnnNodeGroups(groups_, *wrapper_, node_unit_map_, node_units_.size(), logger_).IsOK());
  EXPECT_TRUE(wrapper_->IsQnnTensorWrapperExist("preexisting"));

  if (test_case.accept_later_candidate) {
    // Both candidates start at reshape1 in the same TryQnnFusions invocation.
    EXPECT_EQ(fusion_validation_state.op_types,
              (std::vector<std::string>{QNN_OP_RESHAPE, QNN_OP_TRANSPOSE, QNN_OP_TRANSPOSE}));
    ASSERT_EQ(groups_.size(), 2u);
    EXPECT_EQ(groups_[0]->Type(), "ReshapeTransposeFusion");
    ASSERT_EQ(groups_[0]->GetNodeUnits().size(), 2u);
    EXPECT_EQ(groups_[0]->GetNodeUnits()[0], node_units_[0].get());
    EXPECT_EQ(groups_[0]->GetNodeUnits()[1], node_units_[1].get());
    EXPECT_EQ(groups_[1]->Type(), "OrtNodeUnit");
    EXPECT_EQ(groups_[1]->GetTargetNodeUnit(), node_units_[2].get());

    qnn::QnnParamWrapper perm_key(1, "t", QNN_OP_TRANSPOSE_PARAM_PERM,
                                  std::vector<uint32_t>{6}, std::vector<uint32_t>{1, 0, 2, 4, 3, 5});
    ASSERT_EQ(fusion_validation_state.transposes.size(), 2u);
    const auto& rejected = fusion_validation_state.transposes[0];
    const auto& accepted = fusion_validation_state.transposes[1];
    EXPECT_EQ(rejected.param_name, perm_key.GetParamTensorName());
    EXPECT_EQ(accepted.param_name, rejected.param_name);
    EXPECT_EQ(rejected.param_dims, (std::vector<uint32_t>{5}));
    EXPECT_EQ(accepted.param_dims, (std::vector<uint32_t>{6}));
    EXPECT_EQ(rejected.perm, (std::vector<uint32_t>{0, 1, 3, 2, 4}));
    EXPECT_EQ(accepted.perm, (std::vector<uint32_t>{1, 0, 2, 4, 3, 5}));
    EXPECT_EQ(rejected.output_dims, (std::vector<uint32_t>{2, 3, 5, 4, 6}));
    EXPECT_EQ(accepted.output_dims, (std::vector<uint32_t>{1, 2, 3, 5, 4, 6}));

    // Only the accepted candidate's x/t tensors and the preexisting tensor remain.
    EXPECT_EQ(wrapper_->GetModelTensorsMap().size(), 3u);
    EXPECT_FALSE(wrapper_->IsQnnTensorWrapperExist("r1"));
    EXPECT_FALSE(wrapper_->IsQnnTensorWrapperExist("y"));
    ASSERT_TRUE(wrapper_->IsQnnTensorWrapperExist("x"));
    ASSERT_TRUE(wrapper_->IsQnnTensorWrapperExist("t"));
    EXPECT_EQ(wrapper_->GetQnnTensorWrapper("x").GetTensorDims(), (std::vector<uint32_t>{2, 1, 3, 4, 5, 6}));
    EXPECT_EQ(wrapper_->GetQnnTensorWrapper("t").GetTensorDims(), (std::vector<uint32_t>{1, 2, 3, 5, 4, 6}));

    // Read the committed parameter through node validation after dispatch returns.
    ASSERT_TRUE(wrapper_->CreateQnnNode("check_accepted", QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_TRANSPOSE,
                                        {"x"}, {"t"}, {perm_key.GetParamTensorName()}, /*do_op_validation=*/true));
    EXPECT_EQ(fusion_validation_state.transposes.back().param_dims, (std::vector<uint32_t>{6}));
    EXPECT_EQ(fusion_validation_state.transpose_perm, (std::vector<uint32_t>{1, 0, 2, 4, 3, 5}));
    return;
  }

  const std::vector<std::string> expected_ops = test_case.reject_on_backend
                                                    ? std::vector<std::string>{QNN_OP_RESHAPE, QNN_OP_TRANSPOSE}
                                                    : std::vector<std::string>{QNN_OP_RESHAPE, QNN_OP_TRANSPOSE, QNN_OP_RESHAPE};
  EXPECT_EQ(fusion_validation_state.op_types, expected_ops);
  EXPECT_EQ(fusion_validation_state.transpose_perm, (std::vector<uint32_t>{0, 1, 3, 2, 4}));

  if (!test_case.reject_on_backend && !test_case.reject_on_affinity) {
    ASSERT_EQ(groups_.size(), 1u);
    EXPECT_EQ(groups_[0]->Type(), "Rank6ToRank5Fusion");
    EXPECT_EQ(wrapper_->GetModelTensorsMap().size(), 5u);
    EXPECT_EQ(wrapper_->GetQnnTensorWrapper("r1").GetTensorDims(), (std::vector<uint32_t>{2, 3, 4, 5, 6}));
    return;
  }

  ASSERT_EQ(groups_.size(), 3u);
  for (const auto& group : groups_) {
    EXPECT_EQ(group->Type(), "OrtNodeUnit");
  }
  EXPECT_EQ(wrapper_->GetModelTensorsMap().size(), 1u);
  ASSERT_NO_FATAL_FAILURE(CheckStandaloneTransposeRegistrations());
}

INSTANTIATE_TEST_SUITE_P(
    QnnUnit_FusionValidation, QnnUnit_NodeGroupFusionValidationTest,
    ::testing::Values(FusionValidationCase{true, false, "BackendRejected"},
                      FusionValidationCase{false, false, "Accepted"},
                      FusionValidationCase{false, true, "AffinityRejected"},
                      FusionValidationCase{false, false, "FailedCandidateThenAccepted", true}),
    [](const ::testing::TestParamInfo<FusionValidationCase>& info) { return info.param.name; });

}  // namespace
}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
