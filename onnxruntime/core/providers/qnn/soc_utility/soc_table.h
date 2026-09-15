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
    SocInfo{"SM8350", 30, 68},    // Lahaina
    SocInfo{"SM8325", 34, 68},    // Lahaina 4G Variant
    SocInfo{"SM8450", 36, 69},    // Waipio
    SocInfo{"SC8280X", 37, 68},   // Windows Makena
    SocInfo{"SM8475", 42, 69},    // Palima
    SocInfo{"SM8550", 43, 73},    // Kailua
    SocInfo{"SM8650", 57, 75},    // Lanai
    SocInfo{"SC8380XP", 60, 73},  // Windows Hamoa
    SocInfo{"SM8635", 68, 73},    // Palawan
    SocInfo{"SM8750", 69, 79},    // Pakala
    SocInfo{"SM7675", 70, 73},    // Lamma
    SocInfo{"SM8850", 87, 81},    // Kaanapali
    SocInfo{"SC8480XP", 88, 81},  // Windows Glymur
    SocInfo{"SM8975", 103, 85}    // Hawi
};

}  // namespace soc
}  // namespace qnn
}  // namespace onnxruntime
