// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Function-level unit tests for ParseCustomOpDomains, ParseOpPackages,
// QnnUdoPlaceholderOp, and QnnUdoPlaceholderKernel.
//
// All tests are gated on QNN_EP_INTERNAL_SYMBOL_ACCESS: in coverage builds the test
// binary is link-time bound to onnxruntime_providers_qnn.so, so all EP-internal symbols
// (ParseOpPackages, ParseCustomOpDomains) are directly reachable. In non-coverage builds
// the EP is loaded via dlopen and these symbols are not accessible from the test binary.

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "core/providers/qnn/custom_op/qnn_custom_op.h"
#include "core/providers/qnn/custom_op/qnn_custom_op_domain_parser.h"
#include "core/providers/qnn/builder/op_package/op_package_parser.h"
#include "core/providers/qnn/ort_api.h"
#include "test/providers/qnn/unit/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {

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

// =============================================================================
// ParseOpPackages
// =============================================================================

// Verifies that ParseOpPackages handles a Windows drive-letter path correctly.
// On Windows, the colon-delimited entry contains an extra ':' from the drive letter
// (e.g., "MyOp:C:\\path\\foo.dll:Symbol"). The parser must merge the drive letter and
// the rest of the path into a single token without producing a dangling string_view.
// On other platforms, the same code path is exercised with a POSIX-style path so that
// a regression in the cross-platform parsing logic is caught everywhere.
TEST(QnnUnit_OpPackageParserTest, ParseOpPackages_AbsolutePath) {
  Ort::Logger logger = MakeNullLogger();
  std::vector<onnxruntime::qnn::OpPackage> op_packages;

#if defined(_WIN32)
  // Create a real placeholder file at a Windows-style absolute path so std::filesystem::exists
  // returns true and the drive-letter merge branch is exercised.
  std::filesystem::path tmp_dir = std::filesystem::temp_directory_path();
  std::filesystem::path tmp_dll = tmp_dir / "ort_qnn_parse_oppkg_test.dll";
  std::ofstream(tmp_dll).put('\0');  // create empty placeholder
  ASSERT_TRUE(std::filesystem::exists(tmp_dll));

  const std::string entry = "MyOp:" + tmp_dll.string() + ":MyAddOpPackageInterfaceProvider";
  onnxruntime::ParseOpPackages(entry, op_packages, logger);
  ASSERT_EQ(op_packages.size(), 1u);
  EXPECT_EQ(op_packages[0].op_type, "MyOp");
  EXPECT_EQ(op_packages[0].path, tmp_dll.string());
  EXPECT_EQ(op_packages[0].interface, "MyAddOpPackageInterfaceProvider");
  EXPECT_TRUE(op_packages[0].target.empty());

  // Variant with explicit ":CPU" target.
  op_packages.clear();
  const std::string entry_with_target = entry + ":CPU";
  onnxruntime::ParseOpPackages(entry_with_target, op_packages, logger);
  ASSERT_EQ(op_packages.size(), 1u);
  EXPECT_EQ(op_packages[0].path, tmp_dll.string());
  EXPECT_EQ(op_packages[0].target, "CPU");

  std::filesystem::remove(tmp_dll);
#else
  // POSIX path — exercises the same parsing pipeline (without the Windows merge branch).
  const std::string entry = "MyOp:/tmp/foo.so:MyAddOpPackageInterfaceProvider";
  onnxruntime::ParseOpPackages(entry, op_packages, logger);
  ASSERT_EQ(op_packages.size(), 1u);
  EXPECT_EQ(op_packages[0].op_type, "MyOp");
  EXPECT_EQ(op_packages[0].path, "/tmp/foo.so");
  EXPECT_EQ(op_packages[0].interface, "MyAddOpPackageInterfaceProvider");
  EXPECT_TRUE(op_packages[0].target.empty());

  op_packages.clear();
  onnxruntime::ParseOpPackages(entry + ":CPU", op_packages, logger);
  ASSERT_EQ(op_packages.size(), 1u);
  EXPECT_EQ(op_packages[0].target, "CPU");
#endif
}

#if defined(_WIN32)
// Regression test for the Windows drive-letter merge: parsing of the config string must be
// deterministic in the input — same string → same parse, regardless of filesystem state.
// If the merge were gated on std::filesystem::exists(), a missing DLL would silently mis-parse
// `MyOp:C:\path\foo.dll:Symbol` as 4 tokens with "C" landing in the path slot.
TEST(QnnUnit_OpPackageParserTest, ParseOpPackages_AbsolutePath_NotYetOnDisk) {
  Ort::Logger logger = MakeNullLogger();
  std::vector<onnxruntime::qnn::OpPackage> op_packages;

  // Path that does NOT exist on disk — only the token shape (single-letter drive prefix) drives the merge.
  const std::string non_existent_path = "C:\\does\\not\\exist\\ort_qnn_parse_oppkg_not_on_disk.dll";
  ASSERT_FALSE(std::filesystem::exists(non_existent_path));

  const std::string entry = "MyOp:" + non_existent_path + ":MyAddOpPackageInterfaceProvider";
  onnxruntime::ParseOpPackages(entry, op_packages, logger);
  ASSERT_EQ(op_packages.size(), 1u);
  EXPECT_EQ(op_packages[0].op_type, "MyOp");
  EXPECT_EQ(op_packages[0].path, non_existent_path);
  EXPECT_EQ(op_packages[0].interface, "MyAddOpPackageInterfaceProvider");
  EXPECT_TRUE(op_packages[0].target.empty());

  // Variant with explicit ":CPU" target — the merge must leave room for the trailing target token.
  op_packages.clear();
  onnxruntime::ParseOpPackages(entry + ":CPU", op_packages, logger);
  ASSERT_EQ(op_packages.size(), 1u);
  EXPECT_EQ(op_packages[0].path, non_existent_path);
  EXPECT_EQ(op_packages[0].target, "CPU");
}
#endif

// Verifies that ParseOpPackages preserves a relative path as-is. Relative paths must NOT
// trigger the Windows drive-letter merge branch (which is gated on splitStrings[1] being a
// single ASCII letter), so the parser should pass the path through to op_packages unchanged.
TEST(QnnUnit_OpPackageParserTest, ParseOpPackages_RelativePath) {
  Ort::Logger logger = MakeNullLogger();
  std::vector<onnxruntime::qnn::OpPackage> op_packages;

#if defined(_WIN32)
  // No drive letter → no extra ':' → no merge needed. Path passes through verbatim.
  const std::string entry = "MyOp:foo.dll:MyAddOpPackageInterfaceProvider";
#else
  const std::string entry = "MyOp:foo.so:MyAddOpPackageInterfaceProvider";
#endif
  onnxruntime::ParseOpPackages(entry, op_packages, logger);
  ASSERT_EQ(op_packages.size(), 1u);
  EXPECT_EQ(op_packages[0].op_type, "MyOp");
#if defined(_WIN32)
  EXPECT_EQ(op_packages[0].path, "foo.dll");
#else
  EXPECT_EQ(op_packages[0].path, "foo.so");
#endif
  EXPECT_EQ(op_packages[0].interface, "MyAddOpPackageInterfaceProvider");
  EXPECT_TRUE(op_packages[0].target.empty());

  op_packages.clear();
  onnxruntime::ParseOpPackages(entry + ":CPU", op_packages, logger);
  ASSERT_EQ(op_packages.size(), 1u);
  EXPECT_EQ(op_packages[0].target, "CPU");
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
