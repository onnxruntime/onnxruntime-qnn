// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

#include "test/providers/qnn/qnn_test_utils.h"

#include <onnx/onnx_pb.h>
#include "gtest/gtest.h"

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

// Builds an If-only model (no upstream QNN-eligible producer) so that
// ExpectedEPNodeAssignment::None is a meaningful rejection assertion.
// `x` is a graph input consumed directly by both branches.
GetTestModelFn BuildIfOnlyTestCase(const std::string& if_output_name,
                                   const std::string& then_output_name,
                                   const std::string& else_output_name,
                                   const std::vector<int64_t>& shape = {4}) {
  return [=](ModelTestBuilder& builder) {
    int64_t num_elements = 1;
    for (auto d : shape) num_elements *= d;
    std::vector<float> data(static_cast<size_t>(num_elements));
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<float>(i + 1);

    builder.MakeInput<float>("x", shape, data);
    builder.MakeInputBool("cond", {1});
    builder.MakeOutput<float>(if_output_name, shape);

    GraphProto then_g = MakeMulBranchSubgraph(
        "then_branch", "x", then_output_name, shape, TensorProto::FLOAT, 2.0f, "then_const");
    GraphProto else_g = MakeMulBranchSubgraph(
        "else_branch", "x", else_output_name, shape, TensorProto::FLOAT, -1.0f, "else_const");

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

// Builds an If model where each branch wraps its compute op in a QDQ pair (DQ -> Mul -> Q),
// matching the shape real quantized models produce. Uses uint8 graph IO (x_q in, if_out_q out)
// to avoid standalone outer-graph Q/DQ nodes that QNN EP refuses to partition by themselves.
//   x_q (uint8 graph input) ---+
//                              |
//                              v
//   cond ----------> If(then: DQ(x_q)->Mul(2)->Q,  else: DQ(x_q)->Mul(-1)->Q) -> if_out_q (uint8 graph output)
GetTestModelFn BuildIfQDQTestCase(const std::vector<int64_t>& shape,
                                  float x_scale,
                                  uint8_t x_zero_point,
                                  float out_scale,
                                  uint8_t out_zero_point) {
  return [=](ModelTestBuilder& builder) {
    int64_t num_elements = 1;
    for (auto d : shape) num_elements *= d;
    std::vector<uint8_t> data(static_cast<size_t>(num_elements));
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<uint8_t>(i + 1);

    builder.MakeInput<uint8_t>("x_q", shape, data);
    builder.MakeInputBool("cond", {1});
    builder.MakeOutput<uint8_t>("if_out_q", shape);

    auto build_qdq_mul_branch = [&](const std::string& branch_name,
                                    const std::string& branch_output_name,
                                    float mul_const) {
      GraphProto branch;
      branch.set_name(branch_name);

      ValueInfoProto* out = branch.add_output();
      out->set_name(branch_output_name);
      TypeProto* tp = out->mutable_type();
      tp->mutable_tensor_type()->set_elem_type(TensorProto::UINT8);
      for (int64_t d : shape) {
        tp->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(d);
      }

      const std::string dq_scale_name = branch_name + "_dq_scale";
      const std::string dq_zp_name = branch_name + "_dq_zp";
      const std::string mul_const_name = branch_name + "_mul_const";
      const std::string q_scale_name = branch_name + "_q_scale";
      const std::string q_zp_name = branch_name + "_q_zp";

      auto add_scalar_fp = [&](const std::string& name, float v) {
        TensorProto* t = branch.add_initializer();
        t->set_name(name);
        t->set_data_type(TensorProto::FLOAT);
        t->add_float_data(v);
      };
      auto add_scalar_u8 = [&](const std::string& name, uint8_t v) {
        TensorProto* t = branch.add_initializer();
        t->set_name(name);
        t->set_data_type(TensorProto::UINT8);
        const uint8_t buf[1] = {v};
        t->set_raw_data(buf, sizeof(buf));
      };

      add_scalar_fp(dq_scale_name, x_scale);
      add_scalar_u8(dq_zp_name, x_zero_point);
      add_scalar_fp(mul_const_name, mul_const);
      add_scalar_fp(q_scale_name, out_scale);
      add_scalar_u8(q_zp_name, out_zero_point);

      const std::string dq_out_name = branch_name + "_dq_out";
      const std::string mul_out_name = branch_name + "_mul_out";

      NodeProto* dq = branch.add_node();
      dq->set_name(branch_name + "_dq");
      dq->set_op_type("DequantizeLinear");
      dq->set_domain("");
      dq->add_input("x_q");
      dq->add_input(dq_scale_name);
      dq->add_input(dq_zp_name);
      dq->add_output(dq_out_name);

      NodeProto* mul = branch.add_node();
      mul->set_name(branch_name + "_mul");
      mul->set_op_type("Mul");
      mul->set_domain("");
      mul->add_input(dq_out_name);
      mul->add_input(mul_const_name);
      mul->add_output(mul_out_name);

      NodeProto* q = branch.add_node();
      q->set_name(branch_name + "_q");
      q->set_op_type("QuantizeLinear");
      q->set_domain("");
      q->add_input(mul_out_name);
      q->add_input(q_scale_name);
      q->add_input(q_zp_name);
      q->add_output(branch_output_name);

      return branch;
    };

    GraphProto then_g = build_qdq_mul_branch("then_branch", "then_out_q", 2.0f);
    GraphProto else_g = build_qdq_mul_branch("else_branch", "else_out_q", -1.0f);

    builder.AddNode(
        "if_node",
        "If",
        {"cond"},
        {"if_out_q"},
        "",
        {
            MakeBranchAttribute("then_branch", std::move(then_g)),
            MakeBranchAttribute("else_branch", std::move(else_g)),
        });
  };
}

// Like BuildIfTestCase, but the upstream producer of `x` is Trilu (not in the QNN
// op factory) instead of Sigmoid. ORT places Trilu on the CPU EP, so the implicit
// input `x` consumed by both branches must cross a CPU -> QNN partition boundary.
// Exercises ProcessInputs's manual implicit-input registration on the cross-partition path.
GetTestModelFn BuildIfCrossPartitionImplicitInputTestCase(const std::vector<int64_t>& shape,
                                                          const std::string& if_output_name,
                                                          const std::string& then_output_name,
                                                          const std::string& else_output_name) {
  return [=](ModelTestBuilder& builder) {
    int64_t num_elements = 1;
    for (auto d : shape) num_elements *= d;
    std::vector<float> data(static_cast<size_t>(num_elements));
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<float>(i + 1);

    builder.MakeInput<float>("x_in", shape, data);
    builder.AddNode("x_trilu", "Trilu", {"x_in"}, {"x"});
    builder.MakeInputBool("cond", {1});
    builder.MakeOutput<float>(if_output_name, shape);

    GraphProto then_g = MakeMulBranchSubgraph(
        "then_branch", "x", then_output_name, shape, TensorProto::FLOAT, 2.0f, "then_const");
    GraphProto else_g = MakeMulBranchSubgraph(
        "else_branch", "x", else_output_name, shape, TensorProto::FLOAT, -1.0f, "else_const");

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

// Negative: branch terminus name collides with the If output name.
// Uses an If-only model so ExpectedEPNodeAssignment::None is a meaningful assertion
// (no other QNN-eligible op upstream that would trivially satisfy ::Some).
TEST_F(QnnCPUBackendTests, If_Fp32_DynamicCond_NameCollision_DeclinesFusion) {
  RunIfTest(BuildIfOnlyTestCase("if_out", "if_out", "if_out"),
            ExpectedEPNodeAssignment::None);
}

// Negative: branches reuse the same internal tensor name (the const initializer).
// EP must decline — flattening into one QNN graph would silently mis-wire.
TEST_F(QnnCPUBackendTests, If_Fp32_DynamicCond_SharedInternalName_DeclinesFusion) {
  auto build = [](ModelTestBuilder& builder) {
    const std::vector<int64_t> shape = {4};
    builder.MakeInput<float>("x", shape, {1.f, 2.f, 3.f, 4.f});
    builder.MakeInputBool("cond", {1});
    builder.MakeOutput<float>("if_out", shape);
    // Both branches reuse "shared_const" as the initializer name.
    GraphProto then_g = MakeMulBranchSubgraph(
        "then_branch", "x", "then_out", shape, TensorProto::FLOAT, 2.0f, "shared_const");
    GraphProto else_g = MakeMulBranchSubgraph(
        "else_branch", "x", "else_out", shape, TensorProto::FLOAT, -1.0f, "shared_const");
    builder.AddNode("if_node", "If", {"cond"}, {"if_out"}, "",
                    {MakeBranchAttribute("then_branch", std::move(then_g)),
                     MakeBranchAttribute("else_branch", std::move(else_g))});
  };
  RunIfTest(build, ExpectedEPNodeAssignment::None);
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

// Cross-partition implicit input: QNN EP declines (Trilu on CPU, If's `x` has no QNN producer).
TEST_F(QnnCPUBackendTests, If_Fp32_DynamicCond_CrossPartitionImplicitInput) {
  RunIfTest(BuildIfCrossPartitionImplicitInputTestCase(
                {2, 2}, "if_out", "then_out", "else_out"),
            ExpectedEPNodeAssignment::None);
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

// HTP variant of the cross-partition decline check.
TEST_F(QnnHTPBackendTests, If_FP32_as_FP16_CrossPartitionImplicitInput) {
  RunIfTest(BuildIfCrossPartitionImplicitInputTestCase(
                {1, 2, 3}, "if_out", "then_out", "else_out"),
            ExpectedEPNodeAssignment::None,
            "htp", 19, 0.008f, /*enable_htp_fp16_precision=*/true);
}

// HTP QDQ shape: each branch wraps Mul in DQ -> Mul -> Q, matching real quantized models.
// Currently disabled: QDQ fusion inside If branches is not yet supported. TranslateBranch
// dispatches branch ops directly without wrapping them in OrtNodeUnits, so QNN HTP rejects
// the standalone DequantizeLinear inside the branch with backendValidateOpConfig error 3110.
TEST_F(QnnHTPBackendTests, DISABLED_If_QDQ_U8_BranchesWrapMul) {
  RunIfTest(BuildIfQDQTestCase(/*shape=*/{1, 2, 3},
                               /*x_scale=*/0.01f, /*x_zero_point=*/128,
                               /*out_scale=*/0.02f, /*out_zero_point=*/128),
            ExpectedEPNodeAssignment::All,
            "htp", 19, 0.05f);
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
