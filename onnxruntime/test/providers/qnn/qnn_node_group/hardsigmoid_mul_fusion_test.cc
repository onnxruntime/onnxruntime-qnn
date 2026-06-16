// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

namespace {

// Builds: root -> HardSigmoid -> Mul(<ordered inputs>) -> output, which the QNN EP
// recognizes as HardSwish(x) = x * HardSigmoid(x) and fuses into a single
// QNN_OP_ELEMENT_WISE_NEURON (HardSwish) node.
//
// `hardsigmoid_first` controls the Mul input ordering:
//   true  -> Mul(hsig_out, input)   (HardSigmoid output is Inputs()[0])
//   false -> Mul(input, hsig_out)   (HardSigmoid output is Inputs()[1])
// Both orderings are mathematically equivalent and must fuse identically.
GetTestModelFn BuildHardSigmoidMulTestCase(const TestInputDef<float>& input_def, bool hardsigmoid_first) {
  return [input_def, hardsigmoid_first](ModelTestBuilder& builder) -> void {
    MakeTestInput<float>(builder, "input", input_def);

    // HardSigmoid uses QNN's required alpha=1/6, beta=0.5 so the fusion is eligible.
    std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
    attrs.push_back(MakeAttribute("alpha", 1.0f / 6.0f));
    attrs.push_back(MakeAttribute("beta", 0.5f));
    builder.AddNode("HardSigmoid", "HardSigmoid", {"input"}, {"hsig_out"}, kOnnxDomain, attrs);

    if (hardsigmoid_first) {
      builder.AddNode("Mul", "Mul", {"hsig_out", "input"}, {"output"}, kOnnxDomain);
    } else {
      builder.AddNode("Mul", "Mul", {"input", "hsig_out"}, {"output"}, kOnnxDomain);
    }
    builder.MakeOutput("output");
  };
}

ProviderOptions GetHtpProviderOptions(const std::filesystem::path& json_qnn_graph_dir) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["enable_htp_fp16_precision"] = "1";
#if defined(__linux__) && !defined(__aarch64__)
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();
  return provider_options;
}

// Runs the model and asserts the HardSigmoid+Mul pair fused into a single HardSwish:
// the standalone Mul (ElementWiseMultiply) must be gone, replaced by one
// ElementWiseNeuron (HardSwish).
void RunAndAssertFused(const TestInputDef<float>& input_def, bool hardsigmoid_first,
                       const std::filesystem::path& json_qnn_graph_dir) {
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetHtpProviderOptions(json_qnn_graph_dir);

  RunQnnModelTest(BuildHardSigmoidMulTestCase(input_def, hardsigmoid_first),
                  provider_options,
                  /*opset_version=*/18,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/0.01f);  // fp16 (QNN) vs fp32 (CPU EP).

  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseMultiply", /*count=*/0);
  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseNeuron", /*count=*/1);
}

}  // namespace

// HardSigmoid -> Mul(input, hsig_out): HardSigmoid output is the SECOND Mul input.
// This is the ordering the original same_root_input check already handled.
TEST_F(QnnHTPBackendTests, HardSigmoidMulFusion_NormalOrder_Fuses) {
#if defined(_WIN32)
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
#endif
  auto input_def = TestInputDef<float>({1, 2, 2, 4}, false, GetFloatDataInRange(-5.0f, 5.0f, 16));
  RunAndAssertFused(input_def, /*hardsigmoid_first=*/false, "HardSigmoidMulFusion_NormalOrder");
}

// HardSigmoid -> Mul(hsig_out, input): HardSigmoid output is the FIRST Mul input.
// Reproduces the copy-paste bug: the same_root_input check compared Mul.Inputs()[0]
// against the HardSigmoid input on both sides of the ||, so this (valid) ordering
// failed to match and the pattern was NOT fused. Existing accuracy tests use this
// ordering but only assert EP assignment (which passes whether or not fusion occurs),
// so the missed fusion went undetected. This asserts the fusion actually happens.
TEST_F(QnnHTPBackendTests, HardSigmoidMulFusion_ReversedOrder_Fuses) {
#if defined(_WIN32)
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
#endif
  auto input_def = TestInputDef<float>({1, 2, 2, 4}, false, GetFloatDataInRange(-5.0f, 5.0f, 16));
  RunAndAssertFused(input_def, /*hardsigmoid_first=*/true, "HardSigmoidMulFusion_ReversedOrder");
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
