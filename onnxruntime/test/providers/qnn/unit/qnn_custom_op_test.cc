// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Function-level unit tests for the UDO custom-op domain parser (ParseCustomOpDomains)
// and the placeholder op/kernel (QnnUdoPlaceholderOp / QnnUdoPlaceholderKernel). None of
// these need a real QNN backend or EP-internal linkage, so these tests run in every build
// (no QNN_EP_INTERNAL_SYMBOL_ACCESS guard) — same category as qnn_node_group_utils_test.cc.

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD)

#include <cstring>
#include <string>
#include <vector>

#include "core/providers/qnn/custom_op/qnn_custom_op.h"
#include "core/providers/qnn/custom_op/qnn_custom_op_domain_parser.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace test {

namespace {

// Constructs an Ort::Logger whose cached severity is FATAL, so every ORT_CXX_LOG call
// short-circuits on the severity gate and never dereferences the wrapped null OrtLogger*.
// Production code (ParseCustomOpDomains) logs via plain ORT_CXX_LOG on the reference it is
// given; passing a default-constructed Ort::Logger() would crash on the first WARNING-level
// call (cached severity is VERBOSE, so the null-check gate never short-circuits). This is a
// local copy of qnn_unit_test_utils.h::MakeNullLogger() — that header is gated behind
// QNN_EP_INTERNAL_SYMBOL_ACCESS for unrelated reasons (it also pulls in EP-internal
// qnn_model_wrapper.h), so it cannot be reused directly by this ungated test file.
Ort::Logger MakeNullLogger() {
  static_assert(sizeof(Ort::Logger) == 2 * sizeof(void*),
                "Ort::Logger layout changed — update MakeNullLogger()");
  Ort::Logger logger{std::nullptr_t{}};
  OrtLoggingLevel fatal = ORT_LOGGING_LEVEL_FATAL;
  std::memcpy(reinterpret_cast<char*>(&logger) + sizeof(const OrtLogger*),
              &fatal, sizeof(OrtLoggingLevel));
  return logger;
}

}  // namespace

// =============================================================================
// ParseCustomOpDomains
// =============================================================================

TEST(QnnUnit_CustomOpDomainParserTest, SingleDomainSingleOp) {
  Ort::Logger logger = MakeNullLogger();
  std::vector<onnxruntime::CustomOpDomainSpec> specs;
  onnxruntime::ParseCustomOpDomains("udo_domain:MyAdd", specs, logger);
  ASSERT_EQ(specs.size(), 1u);
  EXPECT_EQ(specs[0].domain, "udo_domain");
  ASSERT_EQ(specs[0].op_types.size(), 1u);
  EXPECT_EQ(specs[0].op_types[0], "MyAdd");
}

TEST(QnnUnit_CustomOpDomainParserTest, SingleDomainMultipleOps) {
  Ort::Logger logger = MakeNullLogger();
  std::vector<onnxruntime::CustomOpDomainSpec> specs;
  onnxruntime::ParseCustomOpDomains("my_domain:OpA,OpB,OpC", specs, logger);
  ASSERT_EQ(specs.size(), 1u);
  EXPECT_EQ(specs[0].domain, "my_domain");
  ASSERT_EQ(specs[0].op_types.size(), 3u);
  EXPECT_EQ(specs[0].op_types[0], "OpA");
  EXPECT_EQ(specs[0].op_types[1], "OpB");
  EXPECT_EQ(specs[0].op_types[2], "OpC");
}

TEST(QnnUnit_CustomOpDomainParserTest, MultipleDomains) {
  Ort::Logger logger = MakeNullLogger();
  std::vector<onnxruntime::CustomOpDomainSpec> specs;
  onnxruntime::ParseCustomOpDomains("domainA:OpX;domainB:OpY,OpZ", specs, logger);
  ASSERT_EQ(specs.size(), 2u);
  EXPECT_EQ(specs[0].domain, "domainA");
  ASSERT_EQ(specs[0].op_types.size(), 1u);
  EXPECT_EQ(specs[0].op_types[0], "OpX");
  EXPECT_EQ(specs[1].domain, "domainB");
  ASSERT_EQ(specs[1].op_types.size(), 2u);
  EXPECT_EQ(specs[1].op_types[0], "OpY");
  EXPECT_EQ(specs[1].op_types[1], "OpZ");
}

TEST(QnnUnit_CustomOpDomainParserTest, EmptyString) {
  Ort::Logger logger = MakeNullLogger();
  std::vector<onnxruntime::CustomOpDomainSpec> specs;
  onnxruntime::ParseCustomOpDomains("", specs, logger);
  EXPECT_TRUE(specs.empty());
}

TEST(QnnUnit_CustomOpDomainParserTest, MissingColon_Skipped) {
  Ort::Logger logger = MakeNullLogger();
  std::vector<onnxruntime::CustomOpDomainSpec> specs;
  // Entry without ':' is malformed and should be skipped; valid entry still processed.
  onnxruntime::ParseCustomOpDomains("bad_entry;good_domain:Op1", specs, logger);
  ASSERT_EQ(specs.size(), 1u);
  EXPECT_EQ(specs[0].domain, "good_domain");
}

TEST(QnnUnit_CustomOpDomainParserTest, EmptyDomain_Skipped) {
  Ort::Logger logger = MakeNullLogger();
  std::vector<onnxruntime::CustomOpDomainSpec> specs;
  onnxruntime::ParseCustomOpDomains(":MyOp;valid_domain:MyOp2", specs, logger);
  ASSERT_EQ(specs.size(), 1u);
  EXPECT_EQ(specs[0].domain, "valid_domain");
}

TEST(QnnUnit_CustomOpDomainParserTest, EmptyOpTypeList_Skipped) {
  Ort::Logger logger = MakeNullLogger();
  std::vector<onnxruntime::CustomOpDomainSpec> specs;
  onnxruntime::ParseCustomOpDomains("domain_no_ops:;valid_domain:Op1", specs, logger);
  ASSERT_EQ(specs.size(), 1u);
  EXPECT_EQ(specs[0].domain, "valid_domain");
}

// =============================================================================
// QnnUdoPlaceholderOp / QnnUdoPlaceholderKernel
// =============================================================================

TEST(QnnUnit_CustomOpTest, PlaceholderOp_Metadata) {
  onnxruntime::qnn::QnnUdoPlaceholderOp op{"MyAdd", "QNNExecutionProvider"};
  EXPECT_STREQ(op.GetName(), "MyAdd");
  EXPECT_STREQ(op.GetExecutionProviderType(), "QNNExecutionProvider");
  EXPECT_EQ(op.GetInputTypeCount(), 1u);
  EXPECT_EQ(op.GetOutputTypeCount(), 1u);
  EXPECT_EQ(op.GetInputCharacteristic(0),
            OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC);
  EXPECT_EQ(op.GetOutputCharacteristic(0),
            OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC);
  EXPECT_FALSE(op.GetVariadicInputHomogeneity());
  EXPECT_FALSE(op.GetVariadicOutputHomogeneity());
}

TEST(QnnUnit_CustomOpTest, PlaceholderKernel_ComputeReturnsError) {
  onnxruntime::qnn::QnnUdoPlaceholderKernel kernel;
  OrtStatusPtr status = kernel.ComputeV2(/*context=*/nullptr);
  ASSERT_NE(status, nullptr) << "Expected an error status but got nullptr (success)";
  // Confirm it's ORT_FAIL and the message mentions 'fused'.
  EXPECT_EQ(Ort::GetApi().GetErrorCode(status), ORT_FAIL);
  std::string msg = Ort::GetApi().GetErrorMessage(status);
  EXPECT_NE(msg.find("fused"), std::string::npos) << "Error message: " << msg;
  Ort::GetApi().ReleaseStatus(status);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
