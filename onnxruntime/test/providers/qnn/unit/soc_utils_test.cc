// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Function-level unit tests for soc_utils.cc — specifically SocModelFromName().
// No QNN backend or physical hardware required.

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include "core/providers/qnn/soc_utils.h"

using namespace onnxruntime::qnn::soc;

// ---------------------------------------------------------------------------
// Exact return values for representative chips
// ---------------------------------------------------------------------------

TEST(QnnUnit_SocUtilsTest, SocModelFromName_SM8750_Returns69) {
  EXPECT_EQ(SocModelFromName("SM8750"), 69u);
}

TEST(QnnUnit_SocUtilsTest, SocModelFromName_SM8850_Returns87) {
  EXPECT_EQ(SocModelFromName("SM8850"), 87u);
}

TEST(QnnUnit_SocUtilsTest, SocModelFromName_SM8650_Returns57) {
  EXPECT_EQ(SocModelFromName("SM8650"), 57u);
}

TEST(QnnUnit_SocUtilsTest, SocModelFromName_SM8550_Returns43) {
  EXPECT_EQ(SocModelFromName("SM8550"), 43u);
}

TEST(QnnUnit_SocUtilsTest, SocModelFromName_SM8450_Returns36) {
  EXPECT_EQ(SocModelFromName("SM8450"), 36u);
}

TEST(QnnUnit_SocUtilsTest, SocModelFromName_SM8350_Returns30) {
  EXPECT_EQ(SocModelFromName("SM8350"), 30u);
}

TEST(QnnUnit_SocUtilsTest, SocModelFromName_SC8380XP_Returns60) {
  EXPECT_EQ(SocModelFromName("SC8380XP"), 60u);
}

TEST(QnnUnit_SocUtilsTest, SocModelFromName_SDM845_Returns0) {
  // SDM845 predates HTP; not in the supported-arch map.
  EXPECT_EQ(SocModelFromName("SDM845"), 0u);
}

// ---------------------------------------------------------------------------
// Case-insensitivity
// ---------------------------------------------------------------------------

TEST(QnnUnit_SocUtilsTest, SocModelFromName_Lowercase_Returns69) {
  EXPECT_EQ(SocModelFromName("sm8750"), 69u);
}

TEST(QnnUnit_SocUtilsTest, SocModelFromName_MixedCase_Returns69) {
  EXPECT_EQ(SocModelFromName("Sm8750"), 69u);
}

TEST(QnnUnit_SocUtilsTest, SocModelFromName_AllCaps_Returns69) {
  EXPECT_EQ(SocModelFromName("SM8750"), 69u);
}

// ---------------------------------------------------------------------------
// Unknown / edge inputs → 0 (QNN_SOC_MODEL_UNKNOWN)
// ---------------------------------------------------------------------------

TEST(QnnUnit_SocUtilsTest, SocModelFromName_UnknownName_Returns0) {
  EXPECT_EQ(SocModelFromName("FOOBAR"), 0u);
}

TEST(QnnUnit_SocUtilsTest, SocModelFromName_EmptyString_Returns0) {
  EXPECT_EQ(SocModelFromName(""), 0u);
}

TEST(QnnUnit_SocUtilsTest, SocModelFromName_NumericString_Returns0) {
  // Numeric strings are not in the name table; ParseSocModel() handles them
  // via stoi() separately. SocModelFromName() itself returns 0 for them.
  EXPECT_EQ(SocModelFromName("69"), 0u);
}

TEST(QnnUnit_SocUtilsTest, SocModelFromName_SM8250_Returns0) {
  // SM8250 is not a named entry in Qnn_SocModel_t; must return 0.
  EXPECT_EQ(SocModelFromName("SM8250"), 0u);
}

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
