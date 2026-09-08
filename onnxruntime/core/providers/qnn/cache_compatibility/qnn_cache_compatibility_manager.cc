// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/cache_compatibility/qnn_cache_compatibility_manager.h"

#include <string>
#include <string_view>
#include <variant>
#include <type_traits>

#include "HTP/QnnHtpDevice.h"
#include "QnnTypes.h"

#include "core/providers/qnn/builder/qnn_backend_manager.h"
#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/cache_compatibility/qnn_cache_compatibility_info.h"
#include "core/providers/qnn/htp_usr_drv_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

namespace {

// Boundary of HTP V6x and V7x arch incompatibility.
constexpr uint32_t kHtpV6xAndV7xBreakageArch = 73;
// Adopt 0 to indicate VTCM size unset by user.
constexpr uint32_t kVtcmMbUnset = 0;

// Corresponding index for each field in V1 compatibility info.
enum class CompatibilityInfoIndexV1 : size_t {
  BACKEND_ID = 0,
  SDK_VERSION,
  BACKEND_API_VERSION,
  CONTEXT_BLOB_VERSION,
  HTP_ARCH,
  IS_HTP_USR_DRV,

  SIZE  // Sentinel.
};

// Corresponding index for each field in V2 compatibility info.
enum class CompatibilityInfoIndexV2 : size_t {
  VERSION = 0,
  BACKEND_ID,
  SDK_VERSION,
  BACKEND_API_VERSION,
  HTP_ARCHS,
  SOC_MODELS,
  VTCM_MBS,
  IS_HTP_USR_DRV,

  SIZE  // Sentinel.
};

template <typename DType>
Ort::Status DeserializeString(std::string_view str, /*out*/ DType& val) {
  try {
    val = static_cast<DType>(std::stoi(std::string(str)));
  } catch (const std::invalid_argument& /*ex*/) {
    return MAKE_EP_FAIL("Unrecognized compatibility info format: malformed numeric value.");
  } catch (const std::out_of_range& /*ex*/) {
    return MAKE_EP_FAIL("Unrecognized compatibility info format: malformed numeric value.");
  }
  return Ort::Status();
}

template <typename DType>
Ort::Status DeserializeString(std::string_view str, /*out*/ std::vector<DType>& val) {
  for (auto split_str : utils::SplitString(str, ",")) {
    val.push_back(0);
    RETURN_IF_ERROR(DeserializeString(split_str, val.back()));
  }
  return Ort::Status();
}

Ort::Status DeserializeVersion(std::string_view str, /*out*/ QnnVersion& version) {
  auto parts = utils::SplitString(str, ".");
  RETURN_IF(parts.size() != 3, "Unrecognized compatibility info format: malformed version.");
  RETURN_IF_ERROR(DeserializeString(parts[0], version.major));
  RETURN_IF_ERROR(DeserializeString(parts[1], version.minor));
  RETURN_IF_ERROR(DeserializeString(parts[2], version.patch));
  return Ort::Status();
}

Ort::Status GetHtpUsrDrvInfo(QnnBackendManager* qnn_backend_manager,
                             const uint32_t htp_arch,
                             /*out*/ QnnVersion& sdk_version,
                             /*out*/ bool& is_htp_usr_drv) {
  RETURN_IF_ERROR(htp_usr_drv::IsHtpUsrDrvEnabled(qnn_backend_manager->GetBackendLibDir(),
                                                  htp_arch,
                                                  is_htp_usr_drv));

  // There is no way to query HNRD's backend API version with current APIs. Fortunately, since backend API versions are
  // bumped along with SDK versions, adopt SDK versions in HNRD scenarios, which can be extracted from driver's file
  // version. Note that backend API versions are not entirely discarded to align with QNN behavior.
  if (is_htp_usr_drv) {
    sdk_version = htp_usr_drv::GetHtpUsrDrvVersion();
    RETURN_IF(sdk_version.major == 0 && sdk_version.minor == 0 && sdk_version.patch == 0,
              "Failed to get HtpUsrDrv file version.");
  } else {
    const std::string& sdk_version_str = qnn_backend_manager->GetSdkVersion();
    auto split_versions = utils::SplitString(sdk_version_str, ".");
    RETURN_IF(split_versions.size() != 4 || split_versions[0].substr(0, 1) != "v",
              "Expected SDK version in vMajor.Minor.Patch.BuildId format.");
    sdk_version = QnnVersion{static_cast<uint32_t>(std::stoi(std::string(split_versions[0].substr(1)))),
                             static_cast<uint32_t>(std::stoi(std::string(split_versions[1]))),
                             static_cast<uint32_t>(std::stoi(std::string(split_versions[2])))};
  }

  return Ort::Status();
}

std::string SerializeVersion(const QnnVersion& version) {
  return std::to_string(version.major) + "." + std::to_string(version.minor) + "." + std::to_string(version.patch);
}

std::string SerializeArray(const std::vector<uint32_t>& arr) {
  std::string arr_str;
  for (size_t idx = 0; idx < arr.size(); ++idx) {
    if (idx != 0) {
      arr_str += ",";
    }
    arr_str += std::to_string(arr[idx]);
  }
  return arr_str;
}

template <typename INFO_VER, typename CallbackFunc>
OrtCompiledModelCompatibility ValidateCompatibilityWithCallback(const INFO_VER& cache_info,
                                                                const INFO_VER& runtime_info,
                                                                CallbackFunc validate_htp_compatibility) {
  if (cache_info.backend_id != runtime_info.backend_id) {
    return OrtCompiledModelCompatibility_EP_UNSUPPORTED;
  }

  // Deliberately leave all branches without reduction for readability and maintainability.
  if (!cache_info.is_htp_usr_drv && !runtime_info.is_htp_usr_drv) {
    if (cache_info.backend_api_version <= runtime_info.backend_api_version) {
      return validate_htp_compatibility(cache_info, runtime_info);
    } else {
      return OrtCompiledModelCompatibility_EP_UNSUPPORTED;
    }
  } else if (!cache_info.is_htp_usr_drv && runtime_info.is_htp_usr_drv) {
    if (cache_info.sdk_version <= runtime_info.sdk_version) {
      return validate_htp_compatibility(cache_info, runtime_info);
    } else {
      return OrtCompiledModelCompatibility_EP_UNSUPPORTED;
    }
  } else if (cache_info.is_htp_usr_drv && !runtime_info.is_htp_usr_drv) {
    // Unexpected usage of context binary generated by user driver path but executing on traditional path.
    return OrtCompiledModelCompatibility_EP_UNSUPPORTED;
  } else {  // (info.is_htp_usr_drv && runtime_info.is_htp_usr_drv)
    if (cache_info.sdk_version <= runtime_info.sdk_version) {
      return validate_htp_compatibility(cache_info, runtime_info);
    } else {
      return OrtCompiledModelCompatibility_EP_UNSUPPORTED;
    }
  }

  // Should never reach.
  return OrtCompiledModelCompatibility_EP_UNSUPPORTED;
}

}  // namespace

Ort::Status QnnCacheCompatibilityManager::DeserializeCompatibilityInfo(const std::string& info_string,
                                                                       QnnCompatibilityInfo& info) {
  size_t pos = info_string.find(":");
  RETURN_IF(pos == std::string::npos, "Unrecognized compatibility info format.");
  const std::string first_slice = info_string.substr(0, pos);

  if (first_slice[0] != 'v') {
    info.version = QnnCompatibilityInfoVersion::QNN_COMPATIBILITY_INFO_V1;
    info.info = QnnCompatibilityInfoV1();
    return DeserializeCompatibilityInfoV1(info_string, std::get<QnnCompatibilityInfoV1>(info.info));
  } else if (first_slice == "v2") {
    info.version = QnnCompatibilityInfoVersion::QNN_COMPATIBILITY_INFO_V2;
    info.info = QnnCompatibilityInfoV2();
    return DeserializeCompatibilityInfoV2(info_string, std::get<QnnCompatibilityInfoV2>(info.info));
  } else {
    return MAKE_EP_FAIL(("Unrecognized compatibility info version: " + first_slice).c_str());
  }
}

Ort::Status QnnCacheCompatibilityManager::DeserializeCompatibilityInfoV1(const std::string& info_string,
                                                                         QnnCompatibilityInfoV1& info) {
  // V1 format: <BackendId>:<SDK>:<BackendApi>:<ContextBlob>:<HtpArch>:<IsHtpUsrDrv>.
  auto split_info_strings = utils::SplitString(info_string, ":");
  RETURN_IF(split_info_strings.size() != 6, "Unrecognized compatibility info format: not enough fields.");

  RETURN_IF_ERROR(DeserializeString(split_info_strings[0], info.backend_id));
  RETURN_IF_ERROR(DeserializeVersion(split_info_strings[1], info.sdk_version));
  RETURN_IF_ERROR(DeserializeVersion(split_info_strings[2], info.backend_api_version));
  RETURN_IF_ERROR(DeserializeVersion(split_info_strings[3], info.context_blob_version));
  RETURN_IF_ERROR(DeserializeString(split_info_strings[4], info.htp_arch));
  info.is_htp_usr_drv = split_info_strings[5] == "1";

  return Ort::Status();
}

Ort::Status QnnCacheCompatibilityManager::DeserializeCompatibilityInfoV2(const std::string& info_string,
                                                                         QnnCompatibilityInfoV2& info) {
  // V2 format: v2:<BackendId>:<SDK>:<BackendApi>:<HtpArch0>,...:<SocModel0>,...:<VtcmMb0>,...:<IsHtpUsrDrv>.
  auto split_info_strings = utils::SplitString(info_string, ":");
  RETURN_IF(split_info_strings.size() != 8, "Unrecognized compatibility info format: not enough fields.");

  RETURN_IF_ERROR(DeserializeString(split_info_strings[1], info.backend_id));
  RETURN_IF_ERROR(DeserializeVersion(split_info_strings[2], info.sdk_version));
  RETURN_IF_ERROR(DeserializeVersion(split_info_strings[3], info.backend_api_version));
  RETURN_IF_ERROR(DeserializeString(split_info_strings[4], info.htp_archs));
  RETURN_IF_ERROR(DeserializeString(split_info_strings[5], info.soc_models));
  RETURN_IF_ERROR(DeserializeString(split_info_strings[6], info.vtcm_mbs));
  info.is_htp_usr_drv = split_info_strings[7] == "1";

  RETURN_IF_NOT(info.htp_archs.size() == info.soc_models.size() && info.htp_archs.size() == info.vtcm_mbs.size(),
                "Unrecognized compatibility info format: unequal size in relative fields.");

  return Ort::Status();
}

Ort::Status QnnCacheCompatibilityManager::GetCompatibilityInfo(QnnCompatibilityInfo& info) {
  RETURN_IF(info.version != QnnCompatibilityInfoVersion::QNN_COMPATIBILITY_INFO_V2,
            "Support getting V2 compatibility info only.");

  QnnCompatibilityInfoV2& info_v2 = std::get<QnnCompatibilityInfoV2>(info.info);

  info_v2.backend_id = qnn_backend_manager_->GetBackendId();
  info_v2.backend_api_version = qnn_backend_manager_->GetBackendApiVersion();

  if (info_v2.htp_archs.size() == 0) {
    info_v2.htp_archs.push_back(static_cast<uint32_t>(qnn_backend_manager_->GetHtpArch()));
  }
  if (info_v2.soc_models.size() == 0) {
    info_v2.soc_models.push_back(qnn_backend_manager_->GetSocModel());
  } else {
    RETURN_IF(info_v2.soc_models.size() != info_v2.htp_archs.size(),
              "Unequal size of `htp_archs` and `soc_models` in the given compatibility info.");
  }
  if (info_v2.vtcm_mbs.size() > 0) {
    RETURN_IF(info_v2.vtcm_mbs.size() != info_v2.htp_archs.size(),
              "Unequal size of `htp_archs` and `vtcm_mbs` in the given compatibility info.");
  } else {
    info_v2.vtcm_mbs.assign(info_v2.htp_archs.size(), kVtcmMbUnset);
  }

  // This function, GetCompatibilityInfo, will be invoked by QnnEp to acquire compatibility info for both legacy
  // context binary and multi-SoC DLC. More than one HTP arch value occurs for multi-SoC DLC usage, which can be only
  // prepared on x86 environments where HNRD is not supported. Thus, HTP arch is don't care and useless in such case,
  // passing any one (e.g., the first one to align with legacy context binary) is fine.
  RETURN_IF_ERROR(GetHtpUsrDrvInfo(qnn_backend_manager_,
                                   info_v2.htp_archs[0],
                                   info_v2.sdk_version,
                                   info_v2.is_htp_usr_drv));

  return Ort::Status();
}

Ort::Status QnnCacheCompatibilityManager::GetRuntimeCompatibilityInfoV1(QnnCompatibilityInfoV1& info) {
  info.backend_id = qnn_backend_manager_->GetBackendId();
  info.backend_api_version = qnn_backend_manager_->GetBackendApiVersion();

  info.htp_arch = static_cast<uint32_t>(qnn_backend_manager_->GetHtpArch());
  RETURN_IF(info.htp_arch == static_cast<uint32_t>(QNN_HTP_DEVICE_ARCH_NONE), "Unable to acquire runtime HTP arch.");

  RETURN_IF_ERROR(GetHtpUsrDrvInfo(qnn_backend_manager_,
                                   info.htp_arch,
                                   info.sdk_version,
                                   info.is_htp_usr_drv));

  return Ort::Status();
}

Ort::Status QnnCacheCompatibilityManager::GetRuntimeCompatibilityInfoV2(QnnCompatibilityInfoV2& info) {
  info.backend_id = qnn_backend_manager_->GetBackendId();
  info.backend_api_version = qnn_backend_manager_->GetBackendApiVersion();

  info.htp_archs.push_back(static_cast<uint32_t>(qnn_backend_manager_->GetHtpArch()));
  RETURN_IF(info.htp_archs[0] == static_cast<uint32_t>(QNN_HTP_DEVICE_ARCH_NONE),
            "Unable to acquire runtime HTP arch.");

  info.vtcm_mbs.push_back(static_cast<uint32_t>(qnn_backend_manager_->GetVtcmSize()));
  RETURN_IF(info.vtcm_mbs[0] == 0, "Unable to acquire runtime VTCM size.");

  RETURN_IF_ERROR(GetHtpUsrDrvInfo(qnn_backend_manager_,
                                   info.htp_archs[0],
                                   info.sdk_version,
                                   info.is_htp_usr_drv));

  return Ort::Status();
}

Ort::Status QnnCacheCompatibilityManager::SerializeCompatibilityInfo(const QnnCompatibilityInfo& info,
                                                                     std::string& info_string) {
  if (info.version == QnnCompatibilityInfoVersion::QNN_COMPATIBILITY_INFO_V1) {
    return SerializeCompatibilityInfoV1(std::get<QnnCompatibilityInfoV1>(info.info), info_string);
  } else if (info.version == QnnCompatibilityInfoVersion::QNN_COMPATIBILITY_INFO_V2) {
    return SerializeCompatibilityInfoV2(std::get<QnnCompatibilityInfoV2>(info.info), info_string);
  } else {
    return MAKE_EP_FAIL(("Unrecognized QnnCompatibilityInfoVersion: " +
                         std::to_string(static_cast<uint32_t>(info.version)))
                            .c_str());
  }
}

Ort::Status QnnCacheCompatibilityManager::SerializeCompatibilityInfoV1(const QnnCompatibilityInfoV1& info,
                                                                       std::string& info_string) {
  // Skip serialization if the given info is only default initialized (i.e., no update at all).
  QnnCompatibilityInfoV1 empty_info;
  RETURN_IF(info.sdk_version == empty_info.sdk_version ||
                info.backend_api_version == empty_info.backend_api_version ||
                info.htp_arch == empty_info.htp_arch,
            "No compatibility info to be serialized.");

  const std::string backend_id_string = std::to_string(info.backend_id);
  const std::string sdk_version_string = SerializeVersion(info.sdk_version);
  const std::string backend_api_version_string = SerializeVersion(info.backend_api_version);
  const std::string context_blob_version_string = SerializeVersion(info.context_blob_version);
  const std::string htp_arch_string = std::to_string(info.htp_arch);
  const std::string is_htp_usr_drv_string = info.is_htp_usr_drv ? "1" : "0";

  info_string = (backend_id_string + ":" +
                 sdk_version_string + ":" +
                 backend_api_version_string + ":" +
                 context_blob_version_string + ":" +
                 htp_arch_string + ":" +
                 is_htp_usr_drv_string);

  return Ort::Status();
}

Ort::Status QnnCacheCompatibilityManager::SerializeCompatibilityInfoV2(const QnnCompatibilityInfoV2& info,
                                                                       std::string& info_string) {
  // Skip serialization if the given info is only default initialized (i.e., no update at all).
  QnnCompatibilityInfoV2 empty_info;
  RETURN_IF(info.sdk_version == empty_info.sdk_version ||
                info.backend_api_version == empty_info.backend_api_version ||
                info.htp_archs == empty_info.htp_archs ||
                info.soc_models == empty_info.soc_models ||
                info.vtcm_mbs == empty_info.vtcm_mbs,
            "No compatibility info to be serialized.");

  const std::string backend_id_string = std::to_string(info.backend_id);
  const std::string sdk_version_string = SerializeVersion(info.sdk_version);
  const std::string backend_api_version_string = SerializeVersion(info.backend_api_version);
  const std::string htp_archs_string = SerializeArray(info.htp_archs);
  const std::string soc_models_string = SerializeArray(info.soc_models);
  const std::string vtcm_mbs_string = SerializeArray(info.vtcm_mbs);
  const std::string is_htp_usr_drv_string = info.is_htp_usr_drv ? "1" : "0";

  info_string = ("v2:" +
                 backend_id_string + ":" +
                 sdk_version_string + ":" +
                 backend_api_version_string + ":" +
                 htp_archs_string + ":" +
                 soc_models_string + ":" +
                 vtcm_mbs_string + ":" +
                 is_htp_usr_drv_string);

  return Ort::Status();
}

Ort::Status QnnCacheCompatibilityManager::ValidateCompatibilityInfo(const QnnCompatibilityInfo& info,
                                                                    OrtCompiledModelCompatibility& compatibility) {
  if (info.version == QnnCompatibilityInfoVersion::QNN_COMPATIBILITY_INFO_V1) {
    return ValidateCompatibilityInfoV1(std::get<QnnCompatibilityInfoV1>(info.info), compatibility);
  } else if (info.version == QnnCompatibilityInfoVersion::QNN_COMPATIBILITY_INFO_V2) {
    return ValidateCompatibilityInfoV2(std::get<QnnCompatibilityInfoV2>(info.info), compatibility);
  } else {
    return MAKE_EP_FAIL(("Unrecognized QnnCompatibilityInfoVersion: " +
                         std::to_string(static_cast<uint32_t>(info.version)))
                            .c_str());
  }
}

Ort::Status QnnCacheCompatibilityManager::ValidateCompatibilityInfoV1(const QnnCompatibilityInfoV1& info,
                                                                      OrtCompiledModelCompatibility& compatibility) {
  compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;

  // Get runtime info to be compared with the given one.
  QnnCompatibilityInfoV1 runtime_info;
  RETURN_IF_ERROR(GetRuntimeCompatibilityInfoV1(runtime_info));

  // The comparison order is as below:
  //   1. backend ID
  //   2. backend API version (if no HNRD involved) / SDK version (if HNRD involved)
  //   3. (Deprecated) context blob version
  //   4. HTP arch
  //
  // Notes:
  //   - Context blob version is deprecated from compatibility info due to the peak memory concern in the previous
  //     approach which acquires the runtime context blob version through creating a fake context binary. Nevertheless,
  //     the removal in fact does not impact the compatibility validation since the combination of a new context binary
  //     generated by an old SDK is impossible. Thus, comparing backend API version / SDK version is enough to cover
  //     context blob version.

  auto validate_htp_compatibility = [](const QnnCompatibilityInfoV1& cache_info,
                                       const QnnCompatibilityInfoV1& runtime_info) {
    if (cache_info.htp_arch < runtime_info.htp_arch) {
      return OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION;
    } else if (cache_info.htp_arch == runtime_info.htp_arch) {
      return OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL;
    } else {
      return OrtCompiledModelCompatibility_EP_UNSUPPORTED;
    }
  };

  compatibility = ValidateCompatibilityWithCallback(info, runtime_info, validate_htp_compatibility);

  return Ort::Status();
}

Ort::Status QnnCacheCompatibilityManager::ValidateCompatibilityInfoV2(const QnnCompatibilityInfoV2& info,
                                                                      OrtCompiledModelCompatibility& compatibility) {
  compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;

  // Guard the given compatibility info having equal array sizes.
  RETURN_IF(info.vtcm_mbs.size() > 0 && info.vtcm_mbs.size() != info.htp_archs.size(),
            "Unequal size of `htp_archs` and `vtcm_mbs` in the given compatibility info.");

  // Get runtime info to be compared with the given one.
  QnnCompatibilityInfoV2 runtime_info;
  RETURN_IF_ERROR(GetRuntimeCompatibilityInfoV2(runtime_info));

  // The comparison order is as below:
  //   1. backend ID
  //   2. backend API version (if no HNRD involved) / SDK version (if HNRD involved)
  //   3. (Deprecated) context blob version
  //   4. HTP arch / VTCM size.
  //
  // Notes:
  //   - HTP arch 6x (e.g., v68) is considered incompatible with HTP arch 7x (e.g., v73).
  //   - Only the exact match (i.e., identical HTP arch and VTCM size) is considered optimal.

  auto validate_htp_compatibility = [](const QnnCompatibilityInfoV2& cache_info,
                                       const QnnCompatibilityInfoV2& runtime_info) {
    OrtCompiledModelCompatibility most_optimal_compatibility = OrtCompiledModelCompatibility_EP_UNSUPPORTED;

    const uint32_t runtime_htp_arch = runtime_info.htp_archs[0];
    const uint32_t runtime_vtcm_mb = runtime_info.vtcm_mbs[0];

    for (size_t idx = 0; idx < cache_info.htp_archs.size(); ++idx) {
      OrtCompiledModelCompatibility cache_compatibility;

      const uint32_t cache_htp_arch = cache_info.htp_archs[idx];
      const uint32_t cache_vtcm_mb = cache_info.vtcm_mbs.size() > 0 ? cache_info.vtcm_mbs[idx] : kVtcmMbUnset;

      if ((cache_htp_arch < kHtpV6xAndV7xBreakageArch && runtime_htp_arch >= kHtpV6xAndV7xBreakageArch) ||
          cache_htp_arch > runtime_htp_arch ||
          cache_vtcm_mb > runtime_vtcm_mb) {
        cache_compatibility = OrtCompiledModelCompatibility_EP_UNSUPPORTED;
      } else if (cache_htp_arch == runtime_htp_arch && cache_vtcm_mb == runtime_vtcm_mb) {
        cache_compatibility = OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL;
      } else {
        cache_compatibility = OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION;
      }

      // The comparison leverages the enum definition of OrtCompiledModelCompatibility, where
      // SUPPORTED_OPTIMAL < SUPPORTED_PREFER_RECOMPILATION < UNSUPPORTED. Although the smallest one is NOT_APPLICABLE,
      // this compatibility level is not used here.
      if (cache_compatibility < most_optimal_compatibility) {
        most_optimal_compatibility = cache_compatibility;
      }
    }

    return most_optimal_compatibility;
  };

  compatibility = ValidateCompatibilityWithCallback(info, runtime_info, validate_htp_compatibility);

  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime
