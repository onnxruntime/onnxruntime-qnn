// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "CPU/QnnCpuCommon.h"
#include "HTP/QnnHtpCommon.h"
#include "QnnSdkBuildId.h"
#include "gtest/gtest.h"
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_ep_device_ep_metadata_keys.h"
#include "onnxruntime_session_options_config_keys.h"

#include "test/providers/qnn/qnn_test_utils.h"

#define ORT_MODEL_FOLDER ORT_TSTR("testdata/")

// Defined in test_main.cc.
extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

// Utility class to help create environment using HNRD for testing.
// Expected usage is used along with smart pointer to automatically restore temporarily moved libraries.
class HnrdTestHandle {
 public:
  HnrdTestHandle(uint32_t htp_arch) : htp_arch_(htp_arch) {
    // Move Prepare/Skel/Stub libraries to a temporary directory to trigger HNRD.
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    temp_dir_ = std::string("temp_") + info->test_suite_name() + "-" + info->name();

    std::filesystem::create_directory(temp_dir_);
    for (const std::string& lib : GetRelatedLibs()) {
      if (std::filesystem::exists(lib)) {
        std::filesystem::rename(lib, temp_dir_ / lib);
      }
    }
  }

  ~HnrdTestHandle() {
    // Move libraries back from temporary directory for later testcases.
    for (const std::string& lib : GetRelatedLibs()) {
      if (std::filesystem::exists(temp_dir_ / lib)) {
        std::filesystem::rename(temp_dir_ / lib, lib);
      }
    }

    std::filesystem::remove(temp_dir_);
  }

 private:
  std::vector<std::string> GetRelatedLibs() {
#ifdef _WIN32
    return {"QnnHtpPrepare.dll",
            "libQnnHtpV" + std::to_string(htp_arch_) + "Skel.so",
            "QnnHtpV" + std::to_string(htp_arch_) + "Stub.dll"};
#else
    return {"libQnnHtpPrepare.so",
            "libQnnHtpV" + std::to_string(htp_arch_) + "Skel.so",
            "libQnnHtpV" + std::to_string(htp_arch_) + "Stub.so"};
#endif
  }

  uint32_t htp_arch_;
  std::filesystem::path temp_dir_;
};

#if defined(_WIN32) && defined(_M_ARM64)
TEST_F(QnnHTPBackendTests, ModelCompatibility_SelfValidate_CbTradRtTrad) {
  const ORTCHAR_T* input_model_file = ORT_MODEL_FOLDER "mul_1.onnx";
  std::filesystem::path output_model_file("mul_1_ctx.onnx");
  std::filesystem::remove(output_model_file);

  ProviderOptions qnn_options = {{"backend_type", "htp"}, {"vtcm_mb", "8"}, {"num_graph_prepare_threads", "1"}};

  {
    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, output_model_file.string().c_str());
    so.AddConfigEntry(kOrtSessionOptionsFailOnSuboptimalCompiledModel, "1");

    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, qnn_options);

    ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, input_model_file, so));
    ASSERT_TRUE(std::filesystem::exists(output_model_file));
  }

  Ort::SessionOptions so;
  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, qnn_options);

  ScopedOrtSession scoped(std::move(registered_ep_device),
                          Ort::Session(*ort_env, output_model_file.wstring().c_str(), so));

  std::filesystem::remove(output_model_file);
}

// TODO: Re-enable once CI supports HNRD. One can still run the test on local WoS machine.
TEST_F(QnnHTPBackendTests, DISABLED_ModelCompatibility_SelfValidate_CbTradRtHnrd) {
  QNN_SKIP_TEST_IF_NO_PLATFORM_ATTRS();

  const ORTCHAR_T* input_model_file = ORT_MODEL_FOLDER "mul_1.onnx";
  std::filesystem::path output_model_file("mul_1_ctx.onnx");
  std::filesystem::remove(output_model_file);

  ProviderOptions qnn_options = {{"backend_type", "htp"}, {"vtcm_mb", "8"}, {"num_graph_prepare_threads", "1"}};

  {
    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, output_model_file.string().c_str());
    so.AddConfigEntry(kOrtSessionOptionsFailOnSuboptimalCompiledModel, "1");

    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, qnn_options);

    ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, input_model_file, so));
    ASSERT_TRUE(std::filesystem::exists(output_model_file));
  }

  QnnHtpDevice_Arch_t htp_arch = QnnHTPBackendTests::GetPlatformAttributes().htp_arch;
  auto hnrd_test_handle = std::make_unique<HnrdTestHandle>(static_cast<uint32_t>(htp_arch));

  try {
    {
      Ort::SessionOptions so;
      RegisteredEpDeviceUniquePtr registered_ep_device;
      RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, qnn_options);

      ScopedOrtSession scoped(std::move(registered_ep_device),
                              Ort::Session(*ort_env, output_model_file.wstring().c_str(), so));
    }
    // Compare to ModelCompatibility_SelfValidate_CbHnrdRtTrad, this testcase could get here if driver is as new as
    // compiled SDK.
  } catch (const Ort::Exception& e) {
    std::string message(e.what());
    ASSERT_TRUE(message.find("Compiled model is not supported by execution provider") != std::string::npos);
  }

  std::filesystem::remove(output_model_file);
}

// TODO: Re-enable once CI supports HNRD. One can still run the test on local WoS machine.
TEST_F(QnnHTPBackendTests, DISABLED_ModelCompatibility_SelfValidate_CbHnrdRtTrad) {
  QNN_SKIP_TEST_IF_NO_PLATFORM_ATTRS();

  const ORTCHAR_T* input_model_file = ORT_MODEL_FOLDER "mul_1.onnx";
  std::filesystem::path output_model_file("mul_1_ctx.onnx");
  std::filesystem::remove(output_model_file);

  ProviderOptions qnn_options = {{"backend_type", "htp"}, {"vtcm_mb", "8"}, {"num_graph_prepare_threads", "1"}};

  QnnHtpDevice_Arch_t htp_arch = QnnHTPBackendTests::GetPlatformAttributes().htp_arch;
  auto hnrd_test_handle = std::make_unique<HnrdTestHandle>(static_cast<uint32_t>(htp_arch));

  {
    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, output_model_file.string().c_str());
    so.AddConfigEntry(kOrtSessionOptionsFailOnSuboptimalCompiledModel, "1");

    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, qnn_options);

    ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, input_model_file, so));
    ASSERT_TRUE(std::filesystem::exists(output_model_file));
  }

  hnrd_test_handle.reset();

  try {
    {
      Ort::SessionOptions so;
      RegisteredEpDeviceUniquePtr registered_ep_device;
      RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, qnn_options);

      ScopedOrtSession scoped(std::move(registered_ep_device),
                              Ort::Session(*ort_env, output_model_file.wstring().c_str(), so));
    }
    FAIL() << "Expect compiled model not supported by execution provider.";  // Should not get here.
  } catch (const Ort::Exception& e) {
    std::string message(e.what());
    ASSERT_TRUE(message.find("Compiled model is not supported by execution provider") != std::string::npos);
  }

  std::filesystem::remove(output_model_file);
}

// TODO: Re-enable once CI supports HNRD. One can still run the test on local WoS machine.
TEST_F(QnnHTPBackendTests, DISABLED_ModelCompatibility_SelfValidate_CbHnrdRtHnrd) {
  QNN_SKIP_TEST_IF_NO_PLATFORM_ATTRS();

  const ORTCHAR_T* input_model_file = ORT_MODEL_FOLDER "mul_1.onnx";
  std::filesystem::path output_model_file("mul_1_ctx.onnx");
  std::filesystem::remove(output_model_file);

  ProviderOptions qnn_options = {{"backend_type", "htp"}, {"vtcm_mb", "8"}, {"num_graph_prepare_threads", "1"}};

  QnnHtpDevice_Arch_t htp_arch = QnnHTPBackendTests::GetPlatformAttributes().htp_arch;
  auto hnrd_test_handle = std::make_unique<HnrdTestHandle>(static_cast<uint32_t>(htp_arch));

  {
    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, output_model_file.string().c_str());
    so.AddConfigEntry(kOrtSessionOptionsFailOnSuboptimalCompiledModel, "1");

    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, qnn_options);

    ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, input_model_file, so));
    ASSERT_TRUE(std::filesystem::exists(output_model_file));
  }

  Ort::SessionOptions so;
  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, qnn_options);

  ScopedOrtSession scoped(std::move(registered_ep_device),
                          Ort::Session(*ort_env, output_model_file.wstring().c_str(), so));

  std::filesystem::remove(output_model_file);
}
#endif  // defined(_WIN32) && defined(_M_ARM64)

struct CompatibilityTestInfoV1 {
  uint32_t backend_id = QNN_BACKEND_ID_HTP;
  std::string sdk_build_id = QNN_SDK_BUILD_ID;  // In format of "v<major>.<minor>.<patch>.<build_id>".
  uint32_t backend_api_version_major = QNN_HTP_API_VERSION_MAJOR;
  uint32_t backend_api_version_minor = QNN_HTP_API_VERSION_MINOR;
  uint32_t backend_api_version_patch = QNN_HTP_API_VERSION_PATCH;
  uint32_t context_blob_version_major = QNN_HTP_CONTEXT_BLOB_VERSION_MAJOR;
  uint32_t context_blob_version_minor = QNN_HTP_CONTEXT_BLOB_VERSION_MINOR;
  uint32_t context_blob_version_patch = QNN_HTP_CONTEXT_BLOB_VERSION_PATCH;
  uint32_t htp_arch = 0;
  bool is_htp_usr_drv = false;

  std::string ToString() const {
    if (sdk_build_id.empty()) {
      return "";
    }
    size_t idx = sdk_build_id.rfind(".");
    std::string sdk_version = sdk_build_id.substr(1, idx - 1);

    return (std::to_string(backend_id) + ":" +
            sdk_version + ":" +
            std::to_string(backend_api_version_major) + "." +
            std::to_string(backend_api_version_minor) + "." +
            std::to_string(backend_api_version_patch) + ":" +
            "0.0.0:" +  // Context blob version is deprecated.
            std::to_string(htp_arch) + ":" +
            (is_htp_usr_drv ? "1" : "0"));
  }
};

struct CompatibilityTestInfoV2 {
  uint32_t backend_id = QNN_BACKEND_ID_HTP;
  std::string sdk_build_id = QNN_SDK_BUILD_ID;  // In format of "v<major>.<minor>.<patch>.<build_id>".
  uint32_t backend_api_version_major = QNN_HTP_API_VERSION_MAJOR;
  uint32_t backend_api_version_minor = QNN_HTP_API_VERSION_MINOR;
  uint32_t backend_api_version_patch = QNN_HTP_API_VERSION_PATCH;
  std::vector<uint32_t> htp_archs;
  std::vector<uint32_t> soc_models;
  std::vector<uint32_t> vtcm_mbs;
  bool is_htp_usr_drv = false;

  void FillPlatformInfo() {
    const QnnHTPBackendTests::QnnPlatformAttributes& platform_attrs = QnnHTPBackendTests::GetPlatformAttributes();
    htp_archs.push_back(static_cast<uint32_t>(platform_attrs.htp_arch));
    soc_models.push_back(0);  // In fact don't care.
    vtcm_mbs.push_back(platform_attrs.vtcm_size_mb);
  }

  std::string ToString() const {
    if (sdk_build_id.empty()) {
      return "";
    }
    size_t idx = sdk_build_id.rfind(".");
    std::string sdk_version = sdk_build_id.substr(1, idx - 1);

    auto serialize_array = [](const std::vector<uint32_t>& arr) {
      // Provide a default value to avoid too early error out during parsing.
      if (arr.empty()) {
        return std::string("0");
      }

      std::string arr_str;
      for (size_t idx = 0; idx < arr.size(); ++idx) {
        if (idx != 0) {
          arr_str += ",";
        }
        arr_str += std::to_string(arr[idx]);
      }
      return arr_str;
    };

    return ("v2:" +
            std::to_string(backend_id) + ":" +
            sdk_version + ":" +
            std::to_string(backend_api_version_major) + "." +
            std::to_string(backend_api_version_minor) + "." +
            std::to_string(backend_api_version_patch) + ":" +
            serialize_array(htp_archs) + ":" +
            serialize_array(soc_models) + ":" +
            serialize_array(vtcm_mbs) + ":" +
            (is_htp_usr_drv ? "1" : "0"));
  }
};

struct MallocAllocator : OrtAllocator {
  MallocAllocator() {
    OrtAllocator::Alloc = [](OrtAllocator* this_, size_t size) {
      return static_cast<MallocAllocator*>(this_)->Alloc(size);
    };
  }

  void* Alloc(size_t size) {
    return malloc(size);
  }
};

TEST_F(QnnHTPBackendTests, ModelCompatibility_GetCompatibility) {
  QNN_SKIP_TEST_IF_NO_PLATFORM_ATTRS();
  auto platform_attrs = QnnHTPBackendTests::GetPlatformAttributes();
#if defined(__aarch64__) || defined(_M_ARM64)
  const uint32_t htp_arch = static_cast<uint32_t>(platform_attrs.htp_arch);
#else
  const uint32_t htp_arch = 73;
#endif

  const ORTCHAR_T* input_model_file = ORT_MODEL_FOLDER "mul_1.onnx";
  const ORTCHAR_T* output_model_file = ORT_TSTR("mul_1_ctx.onnx");
  std::filesystem::remove(output_model_file);

  ProviderOptions qnn_options = {{"backend_type", "htp"}, {"htp_arch", std::to_string(htp_arch)}, {"vtcm_mb", "8"}};

  CONDITIONAL_SKIP_TEST_ON_LINUX_ARM64(qnn_options, QNN_HTP_DEVICE_ARCH_V68, "FP16");

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  // By default, 8 is used, which will impact time to run all
  // unit tests due to overhead of thread creation/destruction
  qnn_options["num_graph_prepare_threads"] = "1";
#endif

  {
    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, std::filesystem::path(output_model_file).string().c_str());

    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, qnn_options);

    ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, input_model_file, so));
    ASSERT_TRUE(std::filesystem::exists(output_model_file));
  }

  {
    Ort::SessionOptions so;
    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, qnn_options);

    ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, output_model_file, so));
    auto& session = scoped.session();

    // Extract generated compatibility info from model metadata.
    OrtModelMetadata* model_metadata = nullptr;
    ASSERT_EQ(nullptr, Ort::GetApi().SessionGetModelMetadata(session, &model_metadata));

    MallocAllocator allocator;
    std::string key = std::string(kOrtModelMetadata_EpCompatibilityInfoPrefix) + kQnnExecutionProvider;
    char* val = nullptr;
    ASSERT_EQ(nullptr,
              Ort::GetApi().ModelMetadataLookupCustomMetadataMap(model_metadata, &allocator, key.c_str(), &val));

    CompatibilityTestInfoV2 expected_info;
    // Override SDK-dependent fields from runtime SDK.
    expected_info.sdk_build_id = platform_attrs.sdk_version;
    expected_info.backend_api_version_major = platform_attrs.backend_api_version.major;
    expected_info.backend_api_version_minor = platform_attrs.backend_api_version.minor;
    expected_info.backend_api_version_patch = platform_attrs.backend_api_version.patch;
    // Set platform related fields.
    expected_info.htp_archs.push_back(htp_arch);
    expected_info.soc_models.push_back(0);
    expected_info.vtcm_mbs.push_back(8);

    ASSERT_TRUE(val != nullptr && expected_info.ToString() == val);

    free(val);
    Ort::GetApi().ReleaseModelMetadata(model_metadata);
  }

  std::filesystem::remove(output_model_file);
}

#if defined(_WIN32) && defined(_M_ARM64)
template <typename INFO_VER>
static void TestModelCompatibilityApiValidate(const INFO_VER& test_info,
                                              const OrtCompiledModelCompatibility expected_compatibility) {
  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions so;
  RegisterQnnEpLibrary(registered_ep_device,
                       so,
                       kQnnExecutionProvider,
                       {{"backend_type", "htp"}, {"num_graph_prepare_threads", "1"}});

  const OrtEpDevice* const* ep_devices = nullptr;
  size_t num_ep_devices = 0;
  Ort::GetApi().GetEpDevices(*GetOrtEnv(), &ep_devices, &num_ep_devices);
  const OrtEpDevice* qcom_npu_device = nullptr;
  for (size_t i = 0; i < num_ep_devices; i++) {
    const char* name = Ort::GetApi().EpDevice_EpName(ep_devices[i]);
    const OrtHardwareDevice* ep_hw_device = Ort::GetApi().EpDevice_Device(ep_devices[i]);
    if (name && std::string(name) == kQnnExecutionProvider &&
        Ort::GetApi().HardwareDevice_Type(ep_hw_device) == OrtHardwareDeviceType_NPU) {
      qcom_npu_device = ep_devices[i];
    }
  }

  if (qcom_npu_device == nullptr) {
    GTEST_SKIP() << "No QNN NPU EP device found";
  }

  OrtCompiledModelCompatibility out_status = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
  Ort::GetApi().GetModelCompatibilityForEpDevices(&qcom_npu_device, 1, test_info.ToString().c_str(), &out_status);
  ASSERT_EQ(out_status, expected_compatibility);
}

TEST_F(QnnHTPBackendTests, ModelCompatibility_V1_ApiValidate) {
  CompatibilityTestInfoV1 test_info;
  test_info.htp_arch = static_cast<uint32_t>(QnnHTPBackendTests::GetPlatformAttributes().htp_arch);

  TestModelCompatibilityApiValidate(test_info, OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL);
}

TEST_F(QnnHTPBackendTests, ModelCompatibility_V1_ApiValidate_DiffBackend) {
  CompatibilityTestInfoV1 test_info;
  test_info.backend_id = QNN_BACKEND_ID_CPU;

  TestModelCompatibilityApiValidate(test_info, OrtCompiledModelCompatibility_EP_UNSUPPORTED);
}

TEST_F(QnnHTPBackendTests, ModelCompatibility_V1_ApiValidate_CbTradRtTrad_CbOldApiVersion) {
  CompatibilityTestInfoV1 test_info;
  test_info.backend_api_version_major = 0;
  test_info.backend_api_version_minor = 0;
  test_info.backend_api_version_patch = 0;
  test_info.htp_arch = static_cast<uint32_t>(QnnHTPBackendTests::GetPlatformAttributes().htp_arch);

  TestModelCompatibilityApiValidate(test_info, OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL);
}

TEST_F(QnnHTPBackendTests, ModelCompatibility_V1_ApiValidate_CbTradRtTrad_CbNewApiVersion) {
  CompatibilityTestInfoV1 test_info;
  test_info.backend_api_version_major = 9999;
  test_info.backend_api_version_minor = 9999;
  test_info.backend_api_version_patch = 9999;

  TestModelCompatibilityApiValidate(test_info, OrtCompiledModelCompatibility_EP_UNSUPPORTED);
}

// TODO: Re-enable once CI supports HNRD. One can still run the test on local WoS machine.
TEST_F(QnnHTPBackendTests, DISABLED_ModelCompatibility_V1_ApiValidate_CbTradRtHnrd_CbOldSdkVersion) {
  QNN_SKIP_TEST_IF_NO_PLATFORM_ATTRS();

  QnnHtpDevice_Arch_t htp_arch = QnnHTPBackendTests::GetPlatformAttributes().htp_arch;
  auto hnrd_test_handle = std::make_unique<HnrdTestHandle>(static_cast<uint32_t>(htp_arch));

  CompatibilityTestInfoV1 test_info;
  test_info.sdk_build_id = "v0.0.0.0";
  test_info.htp_arch = static_cast<uint32_t>(htp_arch);

  TestModelCompatibilityApiValidate(test_info, OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL);
}

// TODO: Re-enable once CI supports HNRD. One can still run the test on local WoS machine.
TEST_F(QnnHTPBackendTests, DISABLED_ModelCompatibility_V1_ApiValidate_CbTradRtHnrd_CbNewSdkVersion) {
  QNN_SKIP_TEST_IF_NO_PLATFORM_ATTRS();

  QnnHtpDevice_Arch_t htp_arch = QnnHTPBackendTests::GetPlatformAttributes().htp_arch;
  auto hnrd_test_handle = std::make_unique<HnrdTestHandle>(static_cast<uint32_t>(htp_arch));

  CompatibilityTestInfoV1 test_info;
  test_info.sdk_build_id = "v9999.9999.9999.9999";

  TestModelCompatibilityApiValidate(test_info, OrtCompiledModelCompatibility_EP_UNSUPPORTED);
}

// TODO: Re-enable once CI supports HNRD. One can still run the test on local WoS machine.
TEST_F(QnnHTPBackendTests, DISABLED_ModelCompatibility_V1_ApiValidate_CbHnrdRtHnrd_CbOldSdkVersion) {
  QNN_SKIP_TEST_IF_NO_PLATFORM_ATTRS();

  QnnHtpDevice_Arch_t htp_arch = QnnHTPBackendTests::GetPlatformAttributes().htp_arch;
  auto hnrd_test_handle = std::make_unique<HnrdTestHandle>(static_cast<uint32_t>(htp_arch));

  CompatibilityTestInfoV1 test_info;
  test_info.sdk_build_id = "v0.0.0.0";
  test_info.htp_arch = static_cast<uint32_t>(htp_arch);
  test_info.is_htp_usr_drv = true;

  TestModelCompatibilityApiValidate(test_info, OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL);
}

// TODO: Re-enable once CI supports HNRD. One can still run the test on local WoS machine.
TEST_F(QnnHTPBackendTests, DISABLED_ModelCompatibility_V1_ApiValidate_CbHnrdRtHnrd_CbNewSdkVersion) {
  QNN_SKIP_TEST_IF_NO_PLATFORM_ATTRS();

  QnnHtpDevice_Arch_t htp_arch = QnnHTPBackendTests::GetPlatformAttributes().htp_arch;
  auto hnrd_test_handle = std::make_unique<HnrdTestHandle>(static_cast<uint32_t>(htp_arch));

  CompatibilityTestInfoV1 test_info;
  test_info.sdk_build_id = "v9999.9999.9999.9999";
  test_info.is_htp_usr_drv = true;

  TestModelCompatibilityApiValidate(test_info, OrtCompiledModelCompatibility_EP_UNSUPPORTED);
}

TEST_F(QnnHTPBackendTests, ModelCompatibility_V1_ApiValidate_CbOldHtpArch) {
  CompatibilityTestInfoV1 test_info;
  test_info.htp_arch = 0;

  TestModelCompatibilityApiValidate(test_info, OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION);
}

TEST_F(QnnHTPBackendTests, ModelCompatibility_V1_ApiValidate_CbNewHtpArch) {
  CompatibilityTestInfoV1 test_info;
  test_info.htp_arch = 9999;

  TestModelCompatibilityApiValidate(test_info, OrtCompiledModelCompatibility_EP_UNSUPPORTED);
}

TEST_F(QnnHTPBackendTests, ModelCompatibility_V2_ApiValidate) {
  CompatibilityTestInfoV2 test_info;
  test_info.FillPlatformInfo();

  TestModelCompatibilityApiValidate(test_info, OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL);
}

TEST_F(QnnHTPBackendTests, ModelCompatibility_V2_ApiValidate_DiffBackend) {
  CompatibilityTestInfoV2 test_info;
  test_info.backend_id = QNN_BACKEND_ID_CPU;

  TestModelCompatibilityApiValidate(test_info, OrtCompiledModelCompatibility_EP_UNSUPPORTED);
}

TEST_F(QnnHTPBackendTests, ModelCompatibility_V2_ApiValidate_CbTradRtTrad_CbOldApiVersion) {
  CompatibilityTestInfoV2 test_info;
  test_info.backend_api_version_major = 0;
  test_info.backend_api_version_minor = 0;
  test_info.backend_api_version_patch = 0;
  test_info.FillPlatformInfo();

  TestModelCompatibilityApiValidate(test_info, OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL);
}

TEST_F(QnnHTPBackendTests, ModelCompatibility_V2_ApiValidate_CbTradRtTrad_CbNewApiVersion) {
  CompatibilityTestInfoV2 test_info;
  test_info.backend_api_version_major = 9999;
  test_info.backend_api_version_minor = 9999;
  test_info.backend_api_version_patch = 9999;

  TestModelCompatibilityApiValidate(test_info, OrtCompiledModelCompatibility_EP_UNSUPPORTED);
}

// TODO: Re-enable once CI supports HNRD. One can still run the test on local WoS machine.
TEST_F(QnnHTPBackendTests, DISABLED_ModelCompatibility_V2_ApiValidate_CbTradRtHnrd_CbOldSdkVersion) {
  QNN_SKIP_TEST_IF_NO_PLATFORM_ATTRS();

  QnnHtpDevice_Arch_t htp_arch = QnnHTPBackendTests::GetPlatformAttributes().htp_arch;
  auto hnrd_test_handle = std::make_unique<HnrdTestHandle>(static_cast<uint32_t>(htp_arch));

  CompatibilityTestInfoV2 test_info;
  test_info.sdk_build_id = "v0.0.0.0";
  test_info.FillPlatformInfo();

  TestModelCompatibilityApiValidate(test_info, OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL);
}

// TODO: Re-enable once CI supports HNRD. One can still run the test on local WoS machine.
TEST_F(QnnHTPBackendTests, DISABLED_ModelCompatibility_V2_ApiValidate_CbTradRtHnrd_CbNewSdkVersion) {
  QNN_SKIP_TEST_IF_NO_PLATFORM_ATTRS();

  QnnHtpDevice_Arch_t htp_arch = QnnHTPBackendTests::GetPlatformAttributes().htp_arch;
  auto hnrd_test_handle = std::make_unique<HnrdTestHandle>(static_cast<uint32_t>(htp_arch));

  CompatibilityTestInfoV2 test_info;
  test_info.sdk_build_id = "v9999.9999.9999.9999";

  TestModelCompatibilityApiValidate(test_info, OrtCompiledModelCompatibility_EP_UNSUPPORTED);
}

// TODO: Re-enable once CI supports HNRD. One can still run the test on local WoS machine.
TEST_F(QnnHTPBackendTests, DISABLED_ModelCompatibility_V2_ApiValidate_CbHnrdRtHnrd_CbOldSdkVersion) {
  QNN_SKIP_TEST_IF_NO_PLATFORM_ATTRS();

  QnnHtpDevice_Arch_t htp_arch = QnnHTPBackendTests::GetPlatformAttributes().htp_arch;
  auto hnrd_test_handle = std::make_unique<HnrdTestHandle>(static_cast<uint32_t>(htp_arch));

  CompatibilityTestInfoV2 test_info;
  test_info.sdk_build_id = "v0.0.0.0";
  test_info.is_htp_usr_drv = true;
  test_info.FillPlatformInfo();

  TestModelCompatibilityApiValidate(test_info, OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL);
}

// TODO: Re-enable once CI supports HNRD. One can still run the test on local WoS machine.
TEST_F(QnnHTPBackendTests, DISABLED_ModelCompatibility_V2_ApiValidate_CbHnrdRtHnrd_CbNewSdkVersion) {
  QNN_SKIP_TEST_IF_NO_PLATFORM_ATTRS();

  QnnHtpDevice_Arch_t htp_arch = QnnHTPBackendTests::GetPlatformAttributes().htp_arch;
  auto hnrd_test_handle = std::make_unique<HnrdTestHandle>(static_cast<uint32_t>(htp_arch));

  CompatibilityTestInfoV2 test_info;
  test_info.sdk_build_id = "v9999.9999.9999.9999";
  test_info.is_htp_usr_drv = true;

  TestModelCompatibilityApiValidate(test_info, OrtCompiledModelCompatibility_EP_UNSUPPORTED);
}

TEST_F(QnnHTPBackendTests, ModelCompatibility_V2_ApiValidate_CbHtpArchLessThan73_RtHtpArchAtLeast73) {
  QNN_SKIP_TEST_IF_NO_PLATFORM_ATTRS();
  if (QnnHTPBackendTests::GetPlatformAttributes().htp_arch < 73) {
    GTEST_SKIP() << "Skip as this testcase requires runtime HTP arch >= 73.";
  }

  CompatibilityTestInfoV2 test_info;
  test_info.FillPlatformInfo();
  test_info.htp_archs[0] = 0;

  TestModelCompatibilityApiValidate(test_info, OrtCompiledModelCompatibility_EP_UNSUPPORTED);
}

TEST_F(QnnHTPBackendTests, ModelCompatibility_V2_ApiValidate_CbHtpArch73_RtHtpArch81) {
  QNN_SKIP_TEST_IF_NO_PLATFORM_ATTRS();
  if (QnnHTPBackendTests::GetPlatformAttributes().htp_arch <= 73) {
    GTEST_SKIP() << "Skip as this testcase requires runtime HTP arch > 73.";
  }

  CompatibilityTestInfoV2 test_info;
  test_info.FillPlatformInfo();
  test_info.htp_archs[0] = 73;

  TestModelCompatibilityApiValidate(test_info, OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION);
}

TEST_F(QnnHTPBackendTests, ModelCompatibility_V2_ApiValidate_CbNewHtpArch) {
  CompatibilityTestInfoV2 test_info;
  test_info.FillPlatformInfo();
  test_info.htp_archs[0] = 9999;

  TestModelCompatibilityApiValidate(test_info, OrtCompiledModelCompatibility_EP_UNSUPPORTED);
}

TEST_F(QnnHTPBackendTests, ModelCompatibility_V2_ApiValidate_CbLessVtcm) {
  CompatibilityTestInfoV2 test_info;
  test_info.FillPlatformInfo();
  test_info.vtcm_mbs[0] = 0;

  TestModelCompatibilityApiValidate(test_info, OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION);
}

TEST_F(QnnHTPBackendTests, ModelCompatibility_V2_ApiValidate_CbSameHtpArchDiffVtcm) {
  CompatibilityTestInfoV2 test_info;
  test_info.FillPlatformInfo();
  // Append the same info again and manually modify VTCM.
  test_info.FillPlatformInfo();
  test_info.vtcm_mbs[0] = 0;

  TestModelCompatibilityApiValidate(test_info, OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL);
}

TEST_F(QnnHTPBackendTests, ModelCompatibility_V2_ApiValidate_CbMoreVtcm) {
  CompatibilityTestInfoV2 test_info;
  test_info.FillPlatformInfo();
  test_info.vtcm_mbs[0] = 9999;

  TestModelCompatibilityApiValidate(test_info, OrtCompiledModelCompatibility_EP_UNSUPPORTED);
}
#endif  // defined(_WIN32) && defined(_M_ARM64)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
