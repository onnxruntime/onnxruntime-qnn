// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

#include "core/graph/onnx_protobuf.h"
#include "gtest/gtest.h"
#include "test/providers/qnn/qnn_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::GraphProto;
using ONNX_NAMESPACE::NodeProto;
using ONNX_NAMESPACE::TensorProto;
using ONNX_NAMESPACE::TensorProto_DataType;
using ONNX_NAMESPACE::TypeProto;
using ONNX_NAMESPACE::ValueInfoProto;

// Constructs a branch GraphProto: `input_name` * `const_value` -> `output_name`.
// `input_name` is resolved as an outer-scope tensor at runtime.
GraphProto MakeMulBranchSubgraph(const std::string& branch_name,
                                 const std::string& input_name,
                                 const std::string& output_name,
                                 const std::vector<int64_t>& shape,
                                 TensorProto_DataType dtype,
                                 float const_value,
                                 const std::string& const_initializer_name) {
  GraphProto branch;
  branch.set_name(branch_name);

  ValueInfoProto* out = branch.add_output();
  out->set_name(output_name);
  TypeProto* out_type = out->mutable_type();
  out_type->mutable_tensor_type()->set_elem_type(dtype);
  for (int64_t d : shape) {
    out_type->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(d);
  }

  TensorProto* init = branch.add_initializer();
  init->set_name(const_initializer_name);
  init->set_data_type(dtype);
  init->add_dims(1);
  init->add_float_data(const_value);

  NodeProto* mul = branch.add_node();
  mul->set_name(branch_name + "_mul");
  mul->set_op_type("Mul");
  mul->set_domain("");
  mul->add_input(input_name);
  mul->add_input(const_initializer_name);
  mul->add_output(output_name);

  return branch;
}

AttributeProto MakeBranchAttribute(const std::string& attr_name, GraphProto&& branch) {
  AttributeProto attr;
  attr.set_name(attr_name);
  attr.set_type(AttributeProto::GRAPH);
  *attr.mutable_g() = std::move(branch);
  return attr;
}

// Branch with no compute nodes; output value lives entirely in a branch initializer.
// Models the post-ORT-fold shape of a pure-constant branch.
GraphProto MakeConstantBranchSubgraph(const std::string& branch_name,
                                      const std::string& output_name,
                                      const std::vector<int64_t>& shape,
                                      TensorProto_DataType dtype,
                                      const std::vector<float>& data) {
  GraphProto branch;
  branch.set_name(branch_name);

  ValueInfoProto* out = branch.add_output();
  out->set_name(output_name);
  TypeProto* tp = out->mutable_type();
  tp->mutable_tensor_type()->set_elem_type(dtype);
  for (int64_t d : shape) {
    tp->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(d);
  }

  TensorProto* init = branch.add_initializer();
  init->set_name(output_name);
  init->set_data_type(dtype);
  for (int64_t d : shape) init->add_dims(d);
  for (float v : data) init->add_float_data(v);

  return branch;
}

GetTestModelFn BuildIfTestCase(const std::vector<int64_t>& then_shape,
                               const std::vector<int64_t>& else_shape,
                               TensorProto_DataType then_dtype,
                               TensorProto_DataType else_dtype,
                               const std::string& if_output_name,
                               const std::string& then_output_name,
                               const std::string& else_output_name,
                               const std::vector<int64_t>& input_shape = {4}) {
  return [=](ModelTestBuilder& builder) {
    int64_t num_elements = 1;
    for (auto d : input_shape) num_elements *= d;
    std::vector<float> data(static_cast<size_t>(num_elements));
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<float>(i + 1);

    builder.MakeInput<float>("x_in", input_shape, data);
    builder.AddNode("x_sigmoid", "Sigmoid", {"x_in"}, {"x"});
    builder.MakeInputBool("cond", {1});
    builder.MakeOutput<float>(if_output_name, then_shape);

    GraphProto then_g = MakeMulBranchSubgraph(
        "then_branch", "x", then_output_name, then_shape, then_dtype, 2.0f, "then_const");
    GraphProto else_g = MakeMulBranchSubgraph(
        "else_branch", "x", else_output_name, else_shape, else_dtype, -1.0f, "else_const");

    builder.AddNode(
        "if_node",
        "If",
        {"cond"},
        {if_output_name},
        "",
        {
            MakeBranchAttribute("then_branch", std::move(then_g)),
            MakeBranchAttribute("else_branch", std::move(else_g)),
        });
  };
}

// Builds an If model where each branch is independently dynamic (Mul) or pure-constant
// (initializer-only, no nodes).
GetTestModelFn BuildIfMixedTestCase(bool then_constant,
                                    bool else_constant,
                                    const std::vector<int64_t>& shape = {4}) {
  return [=](ModelTestBuilder& builder) {
    int64_t num_elements = 1;
    for (auto d : shape) num_elements *= d;
    std::vector<float> input_data(static_cast<size_t>(num_elements));
    for (size_t i = 0; i < input_data.size(); ++i) input_data[i] = static_cast<float>(i + 1);

    builder.MakeInput<float>("x_in", shape, input_data);
    builder.AddNode("x_sigmoid", "Sigmoid", {"x_in"}, {"x"});
    builder.MakeInputBool("cond", {1});
    builder.MakeOutput<float>("if_out", shape);

    std::vector<float> then_const_data(static_cast<size_t>(num_elements), 7.0f);
    std::vector<float> else_const_data(static_cast<size_t>(num_elements), -3.0f);

    GraphProto then_g = then_constant
                            ? MakeConstantBranchSubgraph("then_branch", "then_out", shape,
                                                         TensorProto::FLOAT, then_const_data)
                            : MakeMulBranchSubgraph("then_branch", "x", "then_out", shape,
                                                    TensorProto::FLOAT, 2.0f, "then_const");
    GraphProto else_g = else_constant
                            ? MakeConstantBranchSubgraph("else_branch", "else_out", shape,
                                                         TensorProto::FLOAT, else_const_data)
                            : MakeMulBranchSubgraph("else_branch", "x", "else_out", shape,
                                                    TensorProto::FLOAT, -1.0f, "else_const");

    builder.AddNode(
        "if_node",
        "If",
        {"cond"},
        {"if_out"},
        "",
        {
            MakeBranchAttribute("then_branch", std::move(then_g)),
            MakeBranchAttribute("else_branch", std::move(else_g)),
        });
  };
}

}  // namespace

static void RunIfTest(const GetTestModelFn& model_fn,
                      ExpectedEPNodeAssignment expected_ep_assignment,
                      const std::string& backend_name = "cpu",
                      int opset = 19,
                      float fp32_abs_err = 1e-5f,
                      bool enable_htp_fp16_precision = false) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";

  if (enable_htp_fp16_precision) {
#if defined(_WIN32)
    SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
#endif
#if defined(__linux__) && !defined(__aarch64__)
    provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif
    provider_options["enable_htp_fp16_precision"] = "1";
  }

  RunQnnModelTest(model_fn,
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  fp32_abs_err);
}

//
// CPU tests
//

// Positive: dynamic condition, both branches dynamic, identical shape/dtype.
TEST_F(QnnCPUBackendTests, If_Fp32_DynamicCond_BasicBranches) {
  RunIfTest(BuildIfTestCase({4}, {4}, TensorProto::FLOAT, TensorProto::FLOAT,
                            "if_out", "then_out", "else_out"),
            ExpectedEPNodeAssignment::All);
}

// Negative: branches output different shapes.
TEST_F(QnnCPUBackendTests, If_Fp32_DynamicCond_ShapeMismatch_DeclinesFusion) {
  RunIfTest(BuildIfTestCase({4}, {2}, TensorProto::FLOAT, TensorProto::FLOAT,
                            "if_out", "then_out", "else_out"),
            ExpectedEPNodeAssignment::Some);
}

// Negative: branch terminus name collides with the If output name.
TEST_F(QnnCPUBackendTests, If_Fp32_DynamicCond_NameCollision_DeclinesFusion) {
  RunIfTest(BuildIfTestCase({4}, {4}, TensorProto::FLOAT, TensorProto::FLOAT,
                            "if_out", "if_out", "if_out"),
            ExpectedEPNodeAssignment::Some);
}

// Then branch is pure constant (0 nodes, 1 initializer); else branch is dynamic.
TEST_F(QnnCPUBackendTests, If_Fp32_DynamicCond_ThenConstant_ElseDynamic) {
  RunIfTest(BuildIfMixedTestCase(/*then_constant=*/true, /*else_constant=*/false),
            ExpectedEPNodeAssignment::All);
}

// Then branch is dynamic; else branch is pure constant.
TEST_F(QnnCPUBackendTests, If_Fp32_DynamicCond_ThenDynamic_ElseConstant) {
  RunIfTest(BuildIfMixedTestCase(/*then_constant=*/false, /*else_constant=*/true),
            ExpectedEPNodeAssignment::All);
}

// Both branches pure constant.
TEST_F(QnnCPUBackendTests, If_Fp32_DynamicCond_BothBranchesConstant) {
  RunIfTest(BuildIfMixedTestCase(/*then_constant=*/true, /*else_constant=*/true),
            ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

//
// HTP tests (FP32 model executed at FP16 precision).
//

TEST_F(QnnHTPBackendTests, If_FP32_as_FP16_BasicBranches) {
  RunIfTest(BuildIfTestCase({1, 2, 3}, {1, 2, 3}, TensorProto::FLOAT, TensorProto::FLOAT,
                            "if_out", "then_out", "else_out", {1, 2, 3}),
            ExpectedEPNodeAssignment::All,
            "htp", 19, 0.008f, /*enable_htp_fp16_precision=*/true);
}

TEST_F(QnnHTPBackendTests, If_FP32_as_FP16_ThenConstant_ElseDynamic) {
  RunIfTest(BuildIfMixedTestCase(/*then_constant=*/true, /*else_constant=*/false, {1, 2, 3}),
            ExpectedEPNodeAssignment::All,
            "htp", 19, 0.008f, true);
}

TEST_F(QnnHTPBackendTests, If_FP32_as_FP16_ThenDynamic_ElseConstant) {
  RunIfTest(BuildIfMixedTestCase(/*then_constant=*/false, /*else_constant=*/true, {1, 2, 3}),
            ExpectedEPNodeAssignment::All,
            "htp", 19, 0.008f, true);
}

TEST_F(QnnHTPBackendTests, If_FP32_as_FP16_BothBranchesConstant) {
  RunIfTest(BuildIfMixedTestCase(/*then_constant=*/true, /*else_constant=*/true, {1, 2, 3}),
            ExpectedEPNodeAssignment::All,
            "htp", 19, 0.008f, true);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

#if defined(_M_ARM64)

//
// GPU tests
//

TEST_F(QnnGPUBackendTests, If_Fp32_BasicBranches) {
  RunIfTest(BuildIfTestCase({1, 2, 3}, {1, 2, 3}, TensorProto::FLOAT, TensorProto::FLOAT,
                            "if_out", "then_out", "else_out", {1, 2, 3}),
            ExpectedEPNodeAssignment::All,
            "gpu", 19, 0.008f);
}

TEST_F(QnnGPUBackendTests, If_Fp32_ThenConstant_ElseDynamic) {
  RunIfTest(BuildIfMixedTestCase(/*then_constant=*/true, /*else_constant=*/false, {1, 2, 3}),
            ExpectedEPNodeAssignment::All,
            "gpu", 19, 0.008f);
}

TEST_F(QnnGPUBackendTests, If_Fp32_ThenDynamic_ElseConstant) {
  RunIfTest(BuildIfMixedTestCase(/*then_constant=*/false, /*else_constant=*/true, {1, 2, 3}),
            ExpectedEPNodeAssignment::All,
            "gpu", 19, 0.008f);
}

TEST_F(QnnGPUBackendTests, If_Fp32_BothBranchesConstant) {
  RunIfTest(BuildIfMixedTestCase(/*then_constant=*/true, /*else_constant=*/true, {1, 2, 3}),
            ExpectedEPNodeAssignment::All,
            "gpu", 19, 0.008f);
}

#endif  // defined(_M_ARM64)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
