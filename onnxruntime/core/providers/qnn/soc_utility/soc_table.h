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
  std::vector<int> soc_ids;
};

// Sorted by soc_model in ascending order.
static const std::vector<SocInfo> kSocInfos = {
    SocInfo{"SM8350", 30, 68, {415, 439}},                      // Snapdragon 888
    SocInfo{"SM8325", 34, 68, {501}},                           //
    SocInfo{"SM8450", 36, 69, {457, 482, 552}},                 // Snapdragon 8 Gen 1
    SocInfo{"SC8280X", 37, 68, {435}},                          // Snapdragon 8cx Gen 3
    SocInfo{"SM8475", 42, 69, {530, 531, 540}},                 // Snapdragon 8+ Gen 1
    SocInfo{"SM8550", 43, 73, {536, 519, 600, 601}},            // Snapdragon 8 Gen 2
    SocInfo{"SM8650", 57, 75, {557, 577, 696}},                 // Snapdragon 8 Gen 3
    SocInfo{"SC8380XP", 60, 73, {538, 555, 709, 710}},          // Snapdragon X Elite
    SocInfo{"SM8635", 68, 73, {614, 642}},                      // Snapdragon 8s Gen 3
    SocInfo{"SM8750", 69, 79, {618, 639}},                      // Snapdragon 8 Elite
    SocInfo{"SM7675", 70, 73, {632, 643}},                      // Snapdragon 7+ Gen 3
    SocInfo{"SM8850", 87, 81, {660, 661, 722, 723, 743, 782}},  // Snapdragon 8 Elite Gen 5
    SocInfo{"SC8480XP", 88, 81, {662, 698}},                    // Snapdragon X2 Elite Extreme
    SocInfo{"SC8440XP", 94, 81, {693}},                         // Snapdragon X2 Elite
};

}  // namespace soc
}  // namespace qnn
}  // namespace onnxruntime
