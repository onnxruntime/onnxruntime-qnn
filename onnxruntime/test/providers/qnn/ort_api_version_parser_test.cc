// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "core/providers/qnn/ort_api_version_parser.h"

namespace onnxruntime {
namespace test {

using onnxruntime::qnn::detail::ParseRuntimeOrtApiVersion;

TEST(QnnOrtApiVersionParserTest, NullPointerReturnsZero) {
  EXPECT_EQ(ParseRuntimeOrtApiVersion(nullptr), 0u);
}

TEST(QnnOrtApiVersionParserTest, EmptyStringReturnsZero) {
  EXPECT_EQ(ParseRuntimeOrtApiVersion(""), 0u);
}

TEST(QnnOrtApiVersionParserTest, MissingMinorReturnsZero) {
  EXPECT_EQ(ParseRuntimeOrtApiVersion("1"), 0u);
  EXPECT_EQ(ParseRuntimeOrtApiVersion("1."), 0u);
}

TEST(QnnOrtApiVersionParserTest, ValidTwoComponent) {
  EXPECT_EQ(ParseRuntimeOrtApiVersion("1.24"), 24u);
}

TEST(QnnOrtApiVersionParserTest, ValidThreeComponent) {
  EXPECT_EQ(ParseRuntimeOrtApiVersion("1.24.0"), 24u);
  EXPECT_EQ(ParseRuntimeOrtApiVersion("1.26.5"), 26u);
}

TEST(QnnOrtApiVersionParserTest, PrereleaseSuffixIgnoredAfterMinor) {
  EXPECT_EQ(ParseRuntimeOrtApiVersion("1.24.5-rc1"), 24u);
  EXPECT_EQ(ParseRuntimeOrtApiVersion("1.24-rc1"), 24u);
}

TEST(QnnOrtApiVersionParserTest, MajorNotOneReturnsZero) {
  EXPECT_EQ(ParseRuntimeOrtApiVersion("2.0.0"), 0u);
  EXPECT_EQ(ParseRuntimeOrtApiVersion("0.99.0"), 0u);
}

TEST(QnnOrtApiVersionParserTest, NonNumericReturnsZero) {
  EXPECT_EQ(ParseRuntimeOrtApiVersion("abc"), 0u);
  EXPECT_EQ(ParseRuntimeOrtApiVersion("1.x"), 0u);
}

TEST(QnnOrtApiVersionParserTest, LeadingPlusOrSpaceRejected) {
  EXPECT_EQ(ParseRuntimeOrtApiVersion(" 1.24"), 0u);
  EXPECT_EQ(ParseRuntimeOrtApiVersion("+1.24"), 0u);
}

TEST(QnnOrtApiVersionParserTest, NegativeMinorRejected) {
  EXPECT_EQ(ParseRuntimeOrtApiVersion("1.-1"), 0u);
}

TEST(QnnOrtApiVersionParserTest, MissingDotAfterMajorRejected) {
  EXPECT_EQ(ParseRuntimeOrtApiVersion("124"), 0u);
}

}  // namespace test
}  // namespace onnxruntime
