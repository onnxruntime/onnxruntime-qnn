// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_ENABLE_ACCURACY_TESTS

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "test/providers/qnn/infra/specs/builder/opbuilder/mod_specs.h"
#include "test/providers/qnn/qnn_test_utils.h"

namespace onnxruntime::test {
namespace {

GetTestModelFn BuildModModel(const ModSpec& spec) {
  return BuildOpTestCase<int64_t>(
      "Mod_node", "Mod",
      {TestInputDef<int64_t>({1, 4, 5}, false, spec.dividend),
       TestInputDef<int64_t>({1, 5}, spec.divisor_is_initializer, spec.divisor)},
      {}, {});
}

ProviderOptions ModProviderOptions() {
  ProviderOptions options;
  options["backend_type"] = "htp";
  options["offload_graph_io_quantization"] = "0";
  return options;
}

void RunModAccuracy(const ModSpec& spec) {
  RunQnnModelTest(
      BuildModModel(spec), ModProviderOptions(), 13,
      EPVerificationParams{ExpectedEPNodeAssignment::All});
}

}  // namespace

class QnnAcc_Mod_AccuracyTest : public QnnHTPBackendTests,
                                public ::testing::WithParamInterface<ModSpec> {};

TEST_P(QnnAcc_Mod_AccuracyTest, Case) {
  RunModAccuracy(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    , QnnAcc_Mod_AccuracyTest, ::testing::ValuesIn(GetModSpecs()),
    [](const ::testing::TestParamInfo<ModSpec>& info) { return std::string(info.param.name); });

}  // namespace onnxruntime::test

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_ENABLE_ACCURACY_TESTS
