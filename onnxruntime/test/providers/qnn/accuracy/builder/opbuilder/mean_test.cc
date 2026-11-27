// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_ENABLE_ACCURACY_TESTS && \
    (defined(__linux__) || defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC))

#include <string>
#include "gtest/gtest.h"
#include "test/providers/qnn/infra/specs/builder/opbuilder/mean_test_utils.h"

namespace onnxruntime {
namespace test {

// Retain the legacy backend-availability SetUp, as well as numerical-helper skips.
class QnnAcc_Mean_AccuracyTest : public QnnHTPBackendTests,
                                 public ::testing::WithParamInterface<MeanSpec> {};

TEST_P(QnnAcc_Mean_AccuracyTest, Case) {
  const auto& spec = GetParam();
  const auto options = MeanProviderOptions(spec);
  if (spec.qdq) {
    TestQDQModelAccuracy(BuildMeanF32(spec), BuildMeanQDQ(spec), options, spec.opset,
                         ExpectedEPNodeAssignment::All);
  } else {
    RunQnnModelTest(BuildMeanF32(spec), options, spec.opset,
                    EPVerificationParams{ExpectedEPNodeAssignment::All,
                                         ElementwiseAbsoluteVerifier(1e-5f)});
  }
}

INSTANTIATE_TEST_SUITE_P(
    , QnnAcc_Mean_AccuracyTest, ::testing::ValuesIn(kMeanSpecs),
    [](const ::testing::TestParamInfo<MeanSpec>& i) { return std::string(i.param.name); });

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_ENABLE_ACCURACY_TESTS
