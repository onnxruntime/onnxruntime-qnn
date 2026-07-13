// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <variant>
#include <vector>

#include "QnnTypes.h"

namespace onnxruntime {
namespace qnn {

// Extended Qnn_Version_t to support comparing between two versions.
struct QnnVersion : Qnn_Version_t {
  QnnVersion& operator=(const Qnn_Version_t& other) {
    major = other.major;
    minor = other.minor;
    patch = other.patch;
    return *this;
  }

  bool operator==(const QnnVersion& other) const {
    return major == other.major && minor == other.minor;
  }

  bool operator<(const QnnVersion& other) const {
    return major != other.major ? major < other.major : minor < other.minor;
  }

  bool operator<=(const QnnVersion& other) const {
    return major != other.major ? major <= other.major : minor <= other.minor;
  }

  bool operator>(const QnnVersion& other) const {
    return major != other.major ? major > other.major : minor > other.minor;
  }

  bool operator>=(const QnnVersion& other) const {
    return major != other.major ? major >= other.major : minor >= other.minor;
  }
};

// ============================================================================
// BACKWARD-COMPATIBILITY CONTRACT — READ BEFORE MODIFYING
//
// The QnnCompatibilityInfoV* structs below are serialized into on-disk QNN
// context cache binaries. Older binaries produced by prior releases must remain
// readable by newer code. Therefore an existing versioned info struct
// (QnnCompatibilityInfoV1, QnnCompatibilityInfoV2, ...) MUST NOT be modified:
// do not add, remove, reorder, rename, or change the type of any field.
//
// To change the schema, EVOLVE the info instead of revising the existing one:
//   1. Add a new struct QnnCompatibilityInfoV<N+1> with the desired layout.
//   2. Add a new QNN_COMPATIBILITY_INFO_V<N+1> value to
//      QnnCompatibilityInfoVersion.
//   3. Add the new struct to the std::variant in QnnCompatibilityInfo.
//   4. Update QNN_COMPATIBILITY_INFO_INIT to construct the new version, and
//      keep read paths for all older versions intact.
//
// This preserves the ability to load caches written by any earlier version.
// ============================================================================

// V1: legacy context binary.
//
// - `context_blob_version` is retained here only to keep V1 intact;
struct QnnCompatibilityInfoV1 {
  uint32_t backend_id = 0;
  QnnVersion sdk_version = QNN_VERSION_INIT;
  QnnVersion backend_api_version = QNN_VERSION_INIT;
  QnnVersion context_blob_version = QNN_VERSION_INIT;  // Deprecated;
  uint32_t htp_arch = 0;
  bool is_htp_usr_drv = false;
};

// V2: Flexible Context Binary.
//
// - Drops `context_blob_version`. Adds parallel arrays for `htp_arch` / `soc_model` / `vtcm_mb` so a single record
//   can describe multiple SoC variants embedded in one FCB.
// - Caller must enforce `soc_models.size()` == `htp_archs.size()` == `vtcm_mbs.size()`.
// - Length 1 indicates single-SoC cache, and length N indicates FCB cache with N SoCs.
// - `soc_model` will not be participated in compatibility decision as runtime SoC cannot be queried anymore. Keeping
//   it in the struct for completeness.
// - `vtcm_mb=0` indicates unset by user.
struct QnnCompatibilityInfoV2 {
  uint32_t backend_id = 0;
  QnnVersion sdk_version = QNN_VERSION_INIT;
  QnnVersion backend_api_version = QNN_VERSION_INIT;
  std::vector<uint32_t> htp_archs;
  std::vector<uint32_t> soc_models;
  std::vector<uint32_t> vtcm_mbs;
  bool is_htp_usr_drv = false;
};

// Version constants.
enum struct QnnCompatibilityInfoVersion : uint32_t {
  QNN_COMPATIBILITY_INFO_V1 = 1,
  QNN_COMPATIBILITY_INFO_V2 = 2
};

// Versioned wrapper.
//
// - `version` is kept to make the struct self-contained despite duplicate with std::variant.index().
struct QnnCompatibilityInfo {
  QnnCompatibilityInfoVersion version;
  std::variant<QnnCompatibilityInfoV1, QnnCompatibilityInfoV2> info;
};

// QnnCompatibilityInfo initializer macro.
#define QNN_COMPATIBILITY_INFO_INIT \
  {qnn::QnnCompatibilityInfoVersion::QNN_COMPATIBILITY_INFO_V2, qnn::QnnCompatibilityInfoV2()}

}  // namespace qnn
}  // namespace onnxruntime
