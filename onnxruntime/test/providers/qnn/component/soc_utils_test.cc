// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Function-level unit tests for soc_utils.cc — specifically SocModelFromName().
// No QNN backend or physical hardware required.

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include "core/providers/qnn/soc_utils.h"

using namespace onnxruntime::qnn::soc;

// One representative chip per supported HTP arch tier (V68 through V81).
TEST(QnnUnit_SocUtilsTest, SocModelFromName_SupportedChips) {
  EXPECT_EQ(SocModelFromName("SM8350"), 30u);    // V68 — Snapdragon 888
  EXPECT_EQ(SocModelFromName("SM8450"), 36u);    // V69 — Snapdragon 8 Gen 1
  EXPECT_EQ(SocModelFromName("SM8550"), 43u);    // V73 — Snapdragon 8 Gen 2
  EXPECT_EQ(SocModelFromName("SM8650"), 57u);    // V75 — Snapdragon 8 Gen 3
  EXPECT_EQ(SocModelFromName("SC8380XP"), 60u);  // V75 — Snapdragon X Elite
  EXPECT_EQ(SocModelFromName("SM8750"), 69u);    // V79 — Snapdragon 8 Elite
  EXPECT_EQ(SocModelFromName("SM8850"), 87u);    // V81
}

// Lowercase input is normalized before lookup.
TEST(QnnUnit_SocUtilsTest, SocModelFromName_Lowercase) {
  EXPECT_EQ(SocModelFromName("sm8750"), 69u);
  EXPECT_EQ(SocModelFromName("sm8550"), 43u);
  EXPECT_EQ(SocModelFromName("sc8380xp"), 60u);
}

// Mixed-case input is normalized before lookup.
TEST(QnnUnit_SocUtilsTest, SocModelFromName_MixedCase) {
  EXPECT_EQ(SocModelFromName("Sm8750"), 69u);
  EXPECT_EQ(SocModelFromName("sM8650"), 57u);
}

// Unrecognized / out-of-scope inputs return 0 (QNN_SOC_MODEL_UNKNOWN).
TEST(QnnUnit_SocUtilsTest, SocModelFromName_UnknownReturns0) {
  EXPECT_EQ(SocModelFromName("FOOBAR"), 0u);  // unrecognized name
  EXPECT_EQ(SocModelFromName(""), 0u);        // empty string
  EXPECT_EQ(SocModelFromName("69"), 0u);      // numeric string — use stoi path
  EXPECT_EQ(SocModelFromName("SDM845"), 0u);  // pre-HTP chip, not in table
  EXPECT_EQ(SocModelFromName("SM8250"), 0u);  // not in Qnn_SocModel_t
}

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
