// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Function-level unit tests for soc_utils.cc.
// No QNN backend or physical hardware required.

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include "core/providers/qnn/soc_utility/soc_utils.h"
namespace onnxruntime {
namespace test {

// One representative chip per supported HTP arch tier (V68 through V81).
TEST(QnnUnit_SocUtilsTest, MapSocModelFromSocName_SupportedChips) {
  EXPECT_EQ(qnn::soc::MapSocModelFromSocName("SM8350"), 30u);    // V68 — Snapdragon 888
  EXPECT_EQ(qnn::soc::MapSocModelFromSocName("SM8450"), 36u);    // V69 — Snapdragon 8 Gen 1
  EXPECT_EQ(qnn::soc::MapSocModelFromSocName("SM8550"), 43u);    // V73 — Snapdragon 8 Gen 2
  EXPECT_EQ(qnn::soc::MapSocModelFromSocName("SM8650"), 57u);    // V75 — Snapdragon 8 Gen 3
  EXPECT_EQ(qnn::soc::MapSocModelFromSocName("SC8380XP"), 60u);  // V75 — Snapdragon X Elite
  EXPECT_EQ(qnn::soc::MapSocModelFromSocName("SM8750"), 69u);    // V79 — Snapdragon 8 Elite
  EXPECT_EQ(qnn::soc::MapSocModelFromSocName("SM8850"), 87u);    // V81
}

// Lowercase input is normalized before lookup.
TEST(QnnUnit_SocUtilsTest, MapSocModelFromSocName_Lowercase) {
  EXPECT_EQ(qnn::soc::MapSocModelFromSocName("sm8750"), 69u);
  EXPECT_EQ(qnn::soc::MapSocModelFromSocName("sm8550"), 43u);
  EXPECT_EQ(qnn::soc::MapSocModelFromSocName("sc8380xp"), 60u);
}

// Mixed-case input is normalized before lookup.
TEST(QnnUnit_SocUtilsTest, MapSocModelFromSocName_MixedCase) {
  EXPECT_EQ(qnn::soc::MapSocModelFromSocName("Sm8750"), 69u);
  EXPECT_EQ(qnn::soc::MapSocModelFromSocName("sM8650"), 57u);
}

// Unrecognized / out-of-scope inputs return 0 (QNN_SOC_MODEL_UNKNOWN).
TEST(QnnUnit_SocUtilsTest, MapSocModelFromSocName_UnknownReturns0) {
  EXPECT_EQ(qnn::soc::MapSocModelFromSocName("FOOBAR"), 0u);  // unrecognized name
  EXPECT_EQ(qnn::soc::MapSocModelFromSocName(""), 0u);        // empty string
  EXPECT_EQ(qnn::soc::MapSocModelFromSocName("69"), 0u);      // numeric string — use stoi path
  EXPECT_EQ(qnn::soc::MapSocModelFromSocName("SDM845"), 0u);  // pre-HTP chip, not in table
  EXPECT_EQ(qnn::soc::MapSocModelFromSocName("SM8250"), 0u);  // not in Qnn_SocModel_t
}

// Enumerate all SoC in table.
TEST(QnnUnit_SocUtilsTest, MapHtpArchFromSocModel_KnownSoc) {
  EXPECT_EQ(qnn::soc::MapHtpArchFromSocModel(30), 68u);
  EXPECT_EQ(qnn::soc::MapHtpArchFromSocModel(34), 68u);
  EXPECT_EQ(qnn::soc::MapHtpArchFromSocModel(36), 69u);
  EXPECT_EQ(qnn::soc::MapHtpArchFromSocModel(37), 68u);
  EXPECT_EQ(qnn::soc::MapHtpArchFromSocModel(42), 69u);
  EXPECT_EQ(qnn::soc::MapHtpArchFromSocModel(43), 73u);
  EXPECT_EQ(qnn::soc::MapHtpArchFromSocModel(57), 75u);
  EXPECT_EQ(qnn::soc::MapHtpArchFromSocModel(60), 73u);
  EXPECT_EQ(qnn::soc::MapHtpArchFromSocModel(68), 73u);
  EXPECT_EQ(qnn::soc::MapHtpArchFromSocModel(69), 79u);
  EXPECT_EQ(qnn::soc::MapHtpArchFromSocModel(70), 73u);
  EXPECT_EQ(qnn::soc::MapHtpArchFromSocModel(87), 81u);
  EXPECT_EQ(qnn::soc::MapHtpArchFromSocModel(88), 81u);
  EXPECT_EQ(qnn::soc::MapHtpArchFromSocModel(103), 85u);
}

TEST(QnnUnit_SocUtilsTest, MapHtpArchFromSocModel_UnknownSoc) {
  EXPECT_EQ(qnn::soc::MapHtpArchFromSocModel(29), 0u);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
