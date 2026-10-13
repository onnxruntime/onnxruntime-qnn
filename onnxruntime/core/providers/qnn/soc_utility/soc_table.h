// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace onnxruntime {
namespace qnn {
namespace soc {

struct SocInfo {
  std::string soc_name;
  uint32_t soc_model;
  uint32_t htp_arch;
};

// Sorted by soc_model in ascending order.
static const std::vector<SocInfo> kSocInfos = {
    SocInfo{"SM8350", 30, 68},    // Snapdragon 888
    SocInfo{"SM8325", 34, 68},    //
    SocInfo{"SM8450", 36, 69},    // Snapdragon 8 Gen 1
    SocInfo{"SC8280X", 37, 68},   // Snapdragon 8cx Gen 3
    SocInfo{"SM8475", 42, 69},    // Snapdragon 8+ Gen 1
    SocInfo{"SM8550", 43, 73},    // Snapdragon 8 Gen 2
    SocInfo{"SM8650", 57, 75},    // Snapdragon 8 Gen 3
    SocInfo{"SC8380XP", 60, 73},  // Snapdragon X Elite
    SocInfo{"SM8635", 68, 73},    // Snapdragon 8s Gen 3
    SocInfo{"SM8750", 69, 79},    // Snapdragon 8 Elite
    SocInfo{"SM7675", 70, 73},    // Snapdragon 7+ Gen 3
    SocInfo{"SM8850", 87, 81},    // Snapdragon 8 Elite Gen 5
    SocInfo{"SC8480XP", 88, 81}   // Snapdragon X2 Elite
};

}  // namespace soc
}  // namespace qnn
}  // namespace onnxruntime
