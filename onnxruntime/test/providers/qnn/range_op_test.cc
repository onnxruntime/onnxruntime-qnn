// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <string>

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// ONNX Range has no native QNN op. All three inputs must be graph initializers
// and the op is constant-folded by the QNN EP at graph-build time on any
// backend (CPU or HTP).
template <typename T>
static void RunRangeOpTest(T start, T limit, T delta,
                           int opset_version,
                           ExpectedEPNodeAssignment expected_ep_assignment,
                           const std::string& backend_name = "cpu",
                           bool make_dynamic = false) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";
#if defined(__linux__) && !defined(__aarch64__)
  if (backend_name == "htp") {
    provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
  }
#endif

  // make_dynamic=true: pass start as a non-initializer to trigger the dynamic-input rejection path.
  const bool is_init = !make_dynamic;
  std::vector<TestInputDef<T>> scalar_inputs = {
      TestInputDef<T>({}, is_init, {start}),
      TestInputDef<T>({}, /*is_initializer=*/true, {limit}),
      TestInputDef<T>({}, /*is_initializer=*/true, {delta}),
  };
  std::vector<TestInputDef<int64_t>> unused_inputs;

  RunQnnModelTest(BuildOpTestCase<T, int64_t>("Range_node", "Range",
                                              scalar_inputs,
                                              unused_inputs,
                                              /*attrs=*/{}),
                  provider_options,
                  opset_version,
                  EPVerificationParams{expected_ep_assignment});
}

// ---------------------------------------------------------------------------
// CPU backend tests — run on x86/x64 Windows and Linux.
// ---------------------------------------------------------------------------

TEST_F(QnnCPUBackendTests, Range_float32_ascending) {
  RunRangeOpTest<float>(0.0f, 5.0f, 1.0f, 11, ExpectedEPNodeAssignment::All, "cpu");
}

TEST_F(QnnCPUBackendTests, Range_float32_fractional_delta) {
  RunRangeOpTest<float>(1.0f, 5.0f, 0.5f, 11, ExpectedEPNodeAssignment::All, "cpu");
}

TEST_F(QnnCPUBackendTests, Range_float32_descending) {
  RunRangeOpTest<float>(10.0f, 0.0f, -2.0f, 11, ExpectedEPNodeAssignment::All, "cpu");
}

TEST_F(QnnCPUBackendTests, Range_float32_empty) {
  RunRangeOpTest<float>(3.0f, 3.0f, 1.0f, 11, ExpectedEPNodeAssignment::All, "cpu");
}

TEST_F(QnnCPUBackendTests, Range_int32_ascending) {
  RunRangeOpTest<int32_t>(0, 8, 1, 11, ExpectedEPNodeAssignment::All, "cpu");
}

TEST_F(QnnCPUBackendTests, Range_int32_step) {
  RunRangeOpTest<int32_t>(2, 20, 3, 11, ExpectedEPNodeAssignment::All, "cpu");
}

TEST_F(QnnCPUBackendTests, Range_int32_negative_delta) {
  RunRangeOpTest<int32_t>(10, 0, -2, 11, ExpectedEPNodeAssignment::All, "cpu");
}

TEST_F(QnnCPUBackendTests, Range_int32_empty) {
  RunRangeOpTest<int32_t>(5, 5, 1, 11, ExpectedEPNodeAssignment::All, "cpu");
}

TEST_F(QnnCPUBackendTests, Range_int64_ascending) {
  RunRangeOpTest<int64_t>(0, 16, 1, 11, ExpectedEPNodeAssignment::All, "cpu");
}

// Dynamic input must be rejected (fallback to CPU EP).
TEST_F(QnnCPUBackendTests, Range_dynamic_input_rejected) {
  RunRangeOpTest<float>(0.0f, 5.0f, 1.0f, 11, ExpectedEPNodeAssignment::None, "cpu",
                        /*make_dynamic=*/true);
}

// delta == 0 must be rejected. ORT's native CPU Range kernel itself rejects delta == 0
// (throws during the CPU-EP baseline run inside RunAndVerifyOutputsWithEP), before QNN
// EP's own node-assignment check is ever reached. Verify the expected failure so this
// test actively guards against silent regressions (mirrors gru_test.cc's layout1_forward
// negative tests).
TEST_F(QnnCPUBackendTests, Range_delta_zero_rejected) {
  EXPECT_THROW(
      RunRangeOpTest<float>(0.0f, 5.0f, 0.0f, 11, ExpectedEPNodeAssignment::None, "cpu"),
      std::exception);
}

// ---------------------------------------------------------------------------
// HTP backend tests — only register on ARM64 or Linux (matches the repo
// convention used by cumsum_test.cc, etc.; HTP runtime requires those hosts).
// ---------------------------------------------------------------------------
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

TEST_F(QnnHTPBackendTests, Range_float32_ascending) {
  RunRangeOpTest<float>(0.0f, 5.0f, 1.0f, 11, ExpectedEPNodeAssignment::All, "htp");
}

TEST_F(QnnHTPBackendTests, Range_float32_fractional_delta) {
  RunRangeOpTest<float>(1.0f, 5.0f, 0.5f, 11, ExpectedEPNodeAssignment::All, "htp");
}

TEST_F(QnnHTPBackendTests, Range_float32_descending) {
  RunRangeOpTest<float>(10.0f, 0.0f, -2.0f, 11, ExpectedEPNodeAssignment::All, "htp");
}

TEST_F(QnnHTPBackendTests, Range_float32_empty) {
  RunRangeOpTest<float>(3.0f, 3.0f, 1.0f, 11, ExpectedEPNodeAssignment::All, "htp");
}

TEST_F(QnnHTPBackendTests, Range_int32_ascending) {
  RunRangeOpTest<int32_t>(0, 8, 1, 11, ExpectedEPNodeAssignment::All, "htp");
}

TEST_F(QnnHTPBackendTests, Range_int32_step) {
  RunRangeOpTest<int32_t>(2, 20, 3, 11, ExpectedEPNodeAssignment::All, "htp");
}

TEST_F(QnnHTPBackendTests, Range_int32_negative_delta) {
  RunRangeOpTest<int32_t>(10, 0, -2, 11, ExpectedEPNodeAssignment::All, "htp");
}

TEST_F(QnnHTPBackendTests, Range_int32_empty) {
  RunRangeOpTest<int32_t>(5, 5, 1, 11, ExpectedEPNodeAssignment::All, "htp");
}

TEST_F(QnnHTPBackendTests, Range_int64_ascending) {
  RunRangeOpTest<int64_t>(0, 16, 1, 11, ExpectedEPNodeAssignment::All, "htp");
}

// Dynamic input must be rejected (fallback to CPU EP).
TEST_F(QnnHTPBackendTests, Range_dynamic_input_rejected) {
  RunRangeOpTest<float>(0.0f, 5.0f, 1.0f, 11, ExpectedEPNodeAssignment::None, "htp",
                        /*make_dynamic=*/true);
}

// delta == 0 must be rejected. See the CPU-side test above for why this needs EXPECT_THROW.
// On Linux aarch64 the test framework skips FP16/FP32 HTP tests on arch <= 68 without throwing
// (RunQnnModelTest returns via GTEST_SKIP before the CPU baseline runs), so no EXPECT_THROW there.
TEST_F(QnnHTPBackendTests, Range_delta_zero_rejected) {
#if defined(__linux__) && defined(__aarch64__)
  RunRangeOpTest<float>(0.0f, 5.0f, 0.0f, 11, ExpectedEPNodeAssignment::None, "htp");
#else
  EXPECT_THROW(
      RunRangeOpTest<float>(0.0f, 5.0f, 0.0f, 11, ExpectedEPNodeAssignment::None, "htp"),
      std::exception);
#endif
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
