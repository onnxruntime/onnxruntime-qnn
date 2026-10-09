// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <array>
#include <string>
#include <vector>

#include "core/providers/qnn/cache_compatibility/qnn_cache_compatibility_manager.h"

namespace onnxruntime {
namespace test {

TEST(QnnUnit_CacheCompatibilityManagerTest, DeserializeCompatibilityInfo_AcceptsValidV1AndV2) {
  qnn::QnnCacheCompatibilityManager manager(nullptr);

  qnn::QnnCompatibilityInfo v1_info = QNN_COMPATIBILITY_INFO_INIT;
  const Ort::Status v1_status = manager.DeserializeCompatibilityInfo(
      "1:2.0.0:1.22.0:3.0.0:73:0", v1_info);
  ASSERT_TRUE(v1_status.IsOK()) << v1_status.GetErrorMessage();
  EXPECT_EQ(v1_info.version, qnn::QnnCompatibilityInfoVersion::QNN_COMPATIBILITY_INFO_V1);
  EXPECT_EQ(std::get<qnn::QnnCompatibilityInfoV1>(v1_info.info).htp_arch, 73u);

  qnn::QnnCompatibilityInfo v2_info = QNN_COMPATIBILITY_INFO_INIT;
  const Ort::Status v2_status = manager.DeserializeCompatibilityInfo(
      "v2:1:2.0.0:1.22.0:73,75:43,60:0,8:0", v2_info);
  ASSERT_TRUE(v2_status.IsOK()) << v2_status.GetErrorMessage();
  EXPECT_EQ(v2_info.version, qnn::QnnCompatibilityInfoVersion::QNN_COMPATIBILITY_INFO_V2);
  const auto& parsed_v2 = std::get<qnn::QnnCompatibilityInfoV2>(v2_info.info);
  EXPECT_EQ(parsed_v2.htp_archs, (std::vector<uint32_t>{73u, 75u}));
  EXPECT_EQ(parsed_v2.soc_models, (std::vector<uint32_t>{43u, 60u}));
  EXPECT_EQ(parsed_v2.vtcm_mbs, (std::vector<uint32_t>{0u, 8u}));
}

TEST(QnnUnit_CacheCompatibilityManagerTest, DeserializeCompatibilityInfo_RejectsMalformedV1Numbers) {
  constexpr std::array<const char*, 6> invalid_values = {
      "abc", "12abc", " 10", "+10", "-1", "4294967296"};
  qnn::QnnCacheCompatibilityManager manager(nullptr);

  for (const char* invalid_value : invalid_values) {
    SCOPED_TRACE(invalid_value);
    qnn::QnnCompatibilityInfo info = QNN_COMPATIBILITY_INFO_INIT;
    const std::string serialized =
        std::string(invalid_value) + ":2.0.0:1.22.0:3.0.0:73:0";

    const Ort::Status status = manager.DeserializeCompatibilityInfo(serialized, info);
    EXPECT_FALSE(status.IsOK());
    EXPECT_NE(status.GetErrorMessage().find("malformed numeric value"), std::string::npos);
  }
}

TEST(QnnUnit_CacheCompatibilityManagerTest, DeserializeCompatibilityInfo_RejectsMalformedV2Numbers) {
  constexpr std::array<const char*, 6> invalid_values = {
      "abc", "12abc", " 10", "+10", "-1", "4294967296"};
  qnn::QnnCacheCompatibilityManager manager(nullptr);

  for (const char* invalid_value : invalid_values) {
    SCOPED_TRACE(invalid_value);
    qnn::QnnCompatibilityInfo info = QNN_COMPATIBILITY_INFO_INIT;
    const std::string serialized =
        std::string("v2:1:2.0.0:1.22.0:73,") + invalid_value + ":43,60:0,8:0";

    const Ort::Status status = manager.DeserializeCompatibilityInfo(serialized, info);
    EXPECT_FALSE(status.IsOK());
    EXPECT_NE(status.GetErrorMessage().find("malformed numeric value"), std::string::npos);
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
