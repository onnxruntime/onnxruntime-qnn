// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include <gsl/gsl>
#include <gsl/util>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_session_options_config_keys.h"

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/model_test_builder.h"

#define ORT_MODEL_FOLDER ORT_TSTR("testdata/")

// Defined in test_main.cc.
extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

#if (!defined(__aarch64__) && !defined(_M_ARM64) && !defined(_M_ARM64EC)) || defined(QNN_HTP_CROSS_DEVICE_PREPARE_AVAILABLE)

namespace {

void CompileModelWithPerSocOptions(const ProviderOptions& per_soc_options,
                                   bool is_embed_mode = true) {
  const ORTCHAR_T* input_model_file = ORT_MODEL_FOLDER "nhwc_resize_sizes_opset18.quant.onnx";
  const ORTCHAR_T* output_model_file = ORT_TSTR("model_ctx.onnx");
  const ORTCHAR_T* output_bin_file = ORT_TSTR("model_ctx_qnn.bin");
  std::filesystem::remove(output_model_file);
  std::filesystem::remove(output_bin_file);
  auto cleanup = gsl::finally([&output_model_file, &output_bin_file]() {
    std::filesystem::remove(output_model_file);
    std::filesystem::remove(output_bin_file);
  });

  {
    ProviderOptions provider_options(per_soc_options);
    provider_options["backend_type"] = "htp";
    provider_options["offload_graph_io_quantization"] = "0";  // Avoid IO QDQ CPU fallback.
    provider_options["num_graph_prepare_threads"] = "1";
#if defined(QNN_HTP_CROSS_DEVICE_PREPARE_AVAILABLE)
    provider_options["enable_htp_cross_device_prepare"] = "1";
#endif

    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, is_embed_mode ? "1" : "0");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, std::filesystem::path(output_model_file).string().c_str());

    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

    ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, input_model_file, so));
    ASSERT_TRUE(std::filesystem::exists(output_model_file));
    if (!is_embed_mode) {
      ASSERT_TRUE(std::filesystem::exists(output_bin_file));
    }
  }

#if defined(QNN_HTP_CROSS_DEVICE_PREPARE_AVAILABLE)
  {
    ProviderOptions provider_options = {{"backend_type", "htp"}, {"disable_file_mapped_weights", "1"}};
    Ort::SessionOptions so;
    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

    ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, output_model_file, so));
  }
#endif  // defined(QNN_HTP_CROSS_DEVICE_PREPARE_AVAILABLE)
}

struct WeightSharingWarningLog {
  std::mutex mutex;
  std::vector<std::string> messages;

  static void ORT_API_CALL Capture(void* param, OrtLoggingLevel severity, const char*, const char*, const char*,
                                   const char* message) {
    if (severity == ORT_LOGGING_LEVEL_WARNING &&
        std::string(message).find("Multi-SoC weight sharing: model ") != std::string::npos) {
      auto& log = *static_cast<WeightSharingWarningLog*>(param);
      std::lock_guard<std::mutex> lock(log.mutex);
      log.messages.emplace_back(message);
    }
  }
};

void CompileModelWithWeightSharing(const std::vector<ProviderOptions>& per_soc_options,
                                   const std::filesystem::path& model_file = {},
                                   const std::filesystem::path& output_model_prefix = {},
                                   RegisteredEpDeviceUniquePtr* shared_ep_device = nullptr,
                                   bool fail_binary_write = false,
                                   WeightSharingWarningLog* log = nullptr) {
  ASSERT_FALSE(per_soc_options.empty()) << "At least one set of per-SoC options is required.";
  const size_t num_models = per_soc_options.size();
  if (fail_binary_write) {
    ASSERT_GT(num_models, 1u) << "Binary-write failure requires a deferred producer and a terminal session.";
  }
  const std::filesystem::path input_model_file =
      model_file.empty() ? std::filesystem::path(ORT_MODEL_FOLDER "nhwc_resize_sizes_opset18.quant.onnx") : model_file;
  const std::string context_model_prefix = output_model_prefix.empty() ? "model_ctx" : output_model_prefix.string();
  const size_t first_copied_model_idx = output_model_prefix.empty() ? 1 : 0;

  std::vector<std::filesystem::path> input_model_files;
  std::vector<std::filesystem::path> output_model_files;
  std::vector<std::filesystem::path> output_bin_files;
  input_model_files.reserve(num_models);
  output_model_files.reserve(num_models);
  output_bin_files.reserve(num_models);
  for (size_t model_idx = 0; model_idx < num_models; ++model_idx) {
    const std::string model_number = std::to_string(model_idx + 1);
    if (!output_model_prefix.empty()) {
      // Give every model a distinct path across failure and recovery batches.
      input_model_files.emplace_back(context_model_prefix + "_input" + model_number + ".onnx");
    } else if (model_idx == 0) {
      input_model_files.push_back(input_model_file);
    } else {
      input_model_files.emplace_back("input_model_file" + model_number + ".onnx");
    }
    output_model_files.emplace_back(context_model_prefix + model_number + ".onnx");
    output_bin_files.emplace_back(context_model_prefix + model_number + "_qnn.bin");
  }

  auto cleanup = gsl::finally([&input_model_files, &output_model_files, &output_bin_files, first_copied_model_idx]() {
    for (size_t model_idx = first_copied_model_idx; model_idx < input_model_files.size(); ++model_idx) {
      std::filesystem::remove(input_model_files[model_idx]);
    }
    for (const auto& output_model_file : output_model_files) {
      std::filesystem::remove(output_model_file);
    }
    for (const auto& output_bin_file : output_bin_files) {
      std::filesystem::remove(output_bin_file);
    }
  });

  for (size_t model_idx = 0; model_idx < num_models; ++model_idx) {
    std::filesystem::remove(output_model_files[model_idx]);
    std::filesystem::remove(output_bin_files[model_idx]);
    if (model_idx >= first_copied_model_idx) {
      // Distinct model paths avoid duplicate graph names in the shared QNN context.
      ASSERT_TRUE(std::filesystem::copy_file(input_model_file,
                                             input_model_files[model_idx],
                                             std::filesystem::copy_options::overwrite_existing));
    }
  }

  auto get_provider_options = [](const ProviderOptions& soc_options) {
    ProviderOptions provider_options(soc_options);
    provider_options["backend_type"] = "htp";
    provider_options["offload_graph_io_quantization"] = "0";  // Avoid IO QDQ CPU fallback.
    provider_options["num_graph_prepare_threads"] = "1";
#if defined(QNN_HTP_CROSS_DEVICE_PREPARE_AVAILABLE)
    provider_options["enable_htp_cross_device_prepare"] = "1";
#endif

    return provider_options;
  };

  {
    // Keep every session alive until the final one compiles and dumps the shared context.
    // Sessions must be destroyed before the EP library is unregistered.
    RegisteredEpDeviceUniquePtr local_registered_ep_device;
    auto& registered_ep_device = shared_ep_device != nullptr ? *shared_ep_device : local_registered_ep_device;
    std::vector<Ort::Session> sessions;
    sessions.reserve(num_models);
    for (size_t model_idx = 0; model_idx < num_models; ++model_idx) {
      Ort::SessionOptions so;
      so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
      so.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "0");
      so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, output_model_files[model_idx].string().c_str());
      so.AddConfigEntry(kOrtSessionOptionShareEpContexts, "1");
      if (model_idx + 1 == num_models) {
        so.AddConfigEntry(kOrtSessionOptionStopShareEpContexts, "1");
      }
      if (log != nullptr) {
        so.SetLogSeverityLevel(ORT_LOGGING_LEVEL_WARNING);
        Ort::ThrowOnError(Ort::GetApi().SetUserLoggingFunction(so, WeightSharingWarningLog::Capture, log));
      }

      const ProviderOptions provider_options = get_provider_options(per_soc_options[model_idx]);
      if (registered_ep_device == nullptr) {
        RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);
        ASSERT_NE(registered_ep_device.get(), nullptr);
      } else {
        so.AppendExecutionProvider_V2(*ort_env, {Ort::ConstEpDevice(registered_ep_device.get())}, provider_options);
      }
      if (fail_binary_write && model_idx + 1 == num_models) {
        ASSERT_TRUE(std::filesystem::is_regular_file(output_model_files.front()));
        ASSERT_FALSE(std::filesystem::exists(output_bin_files.front()));
        // A directory at the shared binary path makes terminal export fail inside CreateEPContextNodes.
        ASSERT_TRUE(std::filesystem::create_directory(output_bin_files.front()));
      }
      sessions.emplace_back(*ort_env, input_model_files[model_idx].c_str(), so);
    }

    ASSERT_TRUE(std::filesystem::is_regular_file(output_bin_files.front()));
    for (size_t model_idx = 0; model_idx < num_models; ++model_idx) {
      ASSERT_TRUE(std::filesystem::exists(output_model_files[model_idx]));
      if (model_idx != 0) {
        ASSERT_FALSE(std::filesystem::exists(output_bin_files[model_idx]));
      }

      ONNX_NAMESPACE::ModelProto context_model;
      std::ifstream stream(output_model_files[model_idx], std::ios::binary);
      ASSERT_TRUE(stream.good());
      ASSERT_TRUE(context_model.ParseFromIstream(&stream));
      bool references_shared_binary = false;
      for (const auto& node : context_model.graph().node()) {
        if (node.op_type() != "EPContext") {
          continue;
        }
        for (const auto& attr : node.attribute()) {
          if (attr.name() == "ep_cache_context") {
            EXPECT_EQ(attr.s(), output_bin_files.front().filename().string());
            references_shared_binary = true;
          }
        }
      }
      ASSERT_TRUE(references_shared_binary);
    }
  }

#if defined(QNN_HTP_CROSS_DEVICE_PREPARE_AVAILABLE)
  {
    ProviderOptions provider_options = {{"backend_type", "htp"}, {"disable_file_mapped_weights", "1"}};
    RegisteredEpDeviceUniquePtr local_registered_ep_device;
    auto& registered_ep_device = shared_ep_device != nullptr ? *shared_ep_device : local_registered_ep_device;
    std::vector<Ort::Session> sessions;
    sessions.reserve(num_models);
    for (size_t model_idx = 0; model_idx < num_models; ++model_idx) {
      Ort::SessionOptions so;
      so.AddConfigEntry(kOrtSessionOptionShareEpContexts, "1");
      if (model_idx + 1 == num_models) {
        so.AddConfigEntry(kOrtSessionOptionStopShareEpContexts, "1");
      }

      if (registered_ep_device == nullptr) {
        RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);
        ASSERT_NE(registered_ep_device.get(), nullptr);
      } else {
        so.AppendExecutionProvider_V2(*ort_env, {Ort::ConstEpDevice(registered_ep_device.get())}, provider_options);
      }
      sessions.emplace_back(*ort_env, output_model_files[model_idx].c_str(), so);
    }
  }
#endif  // defined(QNN_HTP_CROSS_DEVICE_PREPARE_AVAILABLE)
}

}  // namespace

TEST_F(QnnHTPBackendTests, EPContextMultiSoc_DefaultHtpConfigs) {
  ProviderOptions per_soc_options;
  per_soc_options["htp_arch"] = "68,73,81";
#ifdef _WIN32
  per_soc_options["soc_model"] = "37,60,88";
#else
  per_soc_options["soc_model"] = "30,43,87";
#endif

  CompileModelWithPerSocOptions(per_soc_options);
}

TEST_F(QnnHTPBackendTests, EPContextMultiSoc_DefaultHtpConfigs_NonEmbed) {
  ProviderOptions per_soc_options;
  per_soc_options["htp_arch"] = "68,73,81";
#ifdef _WIN32
  per_soc_options["soc_model"] = "37,60,88";
#else
  per_soc_options["soc_model"] = "30,43,87";
#endif

  CompileModelWithPerSocOptions(per_soc_options, /*is_embed_mode*/ false);
}

TEST_F(QnnHTPBackendTests, EPContextMultiSoc_HtpArchOnly) {
  ProviderOptions per_soc_options;
  per_soc_options["htp_arch"] = "68,73,81";

  CompileModelWithPerSocOptions(per_soc_options);
}

TEST_F(QnnHTPBackendTests, EPContextMultiSoc_HtpArchOnly_SocModelUnknown) {
  ProviderOptions per_soc_options;
  per_soc_options["htp_arch"] = "68,73,81";
  per_soc_options["soc_model"] = "0,0,0";

  CompileModelWithPerSocOptions(per_soc_options);
}

TEST_F(QnnHTPBackendTests, EPContextMultiSoc_SocModelOnly) {
  ProviderOptions per_soc_options;
#ifdef _WIN32
  per_soc_options["soc_model"] = "37,60,88";
#else
  per_soc_options["soc_model"] = "30,43,87";
#endif

  CompileModelWithPerSocOptions(per_soc_options);
}

TEST_F(QnnHTPBackendTests, EPContextMultiSoc_SingleHtpConfigs) {
  ProviderOptions per_soc_options;
  per_soc_options["htp_arch"] = "68,73,81";
#ifdef _WIN32
  per_soc_options["soc_model"] = "37,60,88";
#else
  per_soc_options["soc_model"] = "30,43,87";
#endif
  per_soc_options["vtcm_mb"] = "8";

  CompileModelWithPerSocOptions(per_soc_options);
}

TEST_F(QnnHTPBackendTests, EPContextMultiSoc_MultipleHtpConfigs) {
  ProviderOptions per_soc_options;
  per_soc_options["htp_arch"] = "68,73,81";
#ifdef _WIN32
  per_soc_options["soc_model"] = "37,60,88";
#else
  per_soc_options["soc_model"] = "30,43,87";
#endif
  per_soc_options["vtcm_mb"] = "8,8,8";

  CompileModelWithPerSocOptions(per_soc_options);
}

TEST_F(QnnHTPBackendTests, EPContextMultiSoc_MismatchHtpArchSocModel) {
  ProviderOptions per_soc_options;
  per_soc_options["htp_arch"] = "68,73,81";
#ifdef _WIN32
  per_soc_options["soc_model"] = "37,60";
#else
  per_soc_options["soc_model"] = "30,43";
#endif

  try {
    CompileModelWithPerSocOptions(per_soc_options);
    FAIL() << "Expecting mismatched length of soc_model and htp_arch should fail.";
  } catch (const Ort::Exception& ex) {
    ASSERT_EQ(ex.GetOrtErrorCode(), ORT_FAIL);
    ASSERT_THAT(
        ex.what(),
        testing::HasSubstr("Expecting soc_model and htp_arch having equal number of values in multi-SoC EP context."));
  }
}

TEST_F(QnnHTPBackendTests, EPContextMultiSoc_Legacy) {
  ProviderOptions per_soc_options;
  per_soc_options["htp_arch"] = "73";
#ifdef _WIN32
  per_soc_options["soc_model"] = "60";
#else
  per_soc_options["soc_model"] = "43";
#endif

  CompileModelWithPerSocOptions(per_soc_options);
}

TEST_F(QnnHTPBackendTests, EPContextMultiSoc_HtpArch_68_73_81_NotAllArchSupported) {
  QNN_SKIP_TEST_ON_LINUX_X86_64("Skip as unable to deliberately fallback on Linux x86.");

  ModelTestBuilder helper;

  // Build a graph that the first Add node will fail for op validation on V68 only, impacting the overall partition.
  auto model_builder = [](ModelTestBuilder& builder) {
    std::vector<float> data(2 * 3, 1.0f);
    QuantParams<uint8_t> q_param = GetDataQuantParams<uint8_t>(gsl::make_span(data));

    MakeTestInput(builder, "input1", TestInputDef<float>({2, 3}, false, data));
    MakeTestInput(builder, "input2", TestInputDef<float>({2, 3}, false, data));
    MakeTestInput(builder, "input3", TestInputDef<float>({2, 3}, false, data));

    builder.AddNode("add1", "Add", {"input1", "input2"}, {"add1_out"});

    std::string add2_input1_qdq = AddQDQNodePair<uint8_t>(builder,
                                                          "add2_in1_qdq",
                                                          "add1_out",
                                                          q_param.scale,
                                                          q_param.zero_point);
    std::string add2_input2_qdq = AddQDQNodePair<uint8_t>(builder,
                                                          "add2_in2_qdq",
                                                          "input3",
                                                          q_param.scale,
                                                          q_param.zero_point);

    builder.AddNode("add2", "Add", {add2_input1_qdq, add2_input2_qdq}, {"add2_out"});
    AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder,
                                                   "add2_qdq_out",
                                                   "add2_out",
                                                   q_param.scale,
                                                   q_param.zero_point);
  };
  model_builder(helper);

  const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
  opset_id_proto->set_domain("");
  opset_id_proto->set_version(13);
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  std::string model_data;
  helper.model_.SerializeToString(&model_data);
  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  ProviderOptions provider_options = {{"backend_type", "htp"},
                                      {"offload_graph_io_quantization", "0"},
                                      {"num_graph_prepare_threads", "1"}};
#if defined(QNN_HTP_CROSS_DEVICE_PREPARE_AVAILABLE)
  provider_options["enable_htp_cross_device_prepare"] = "1";
#endif
  provider_options["htp_arch"] = "68,73,81";
#ifdef _WIN32
  provider_options["soc_model"] = "37,60,88";
#else
  provider_options["soc_model"] = "30,43,87";
#endif

  std::filesystem::path output_model_file("model_ctx.onnx");
  std::filesystem::remove(output_model_file);

  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, output_model_file.string().c_str());
  so.AddConfigEntry(kOrtSessionOptionsRecordEpGraphAssignmentInfo, "1");  // For verifying node assignment.

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  ScopedOrtSession scoped(std::move(registered_ep_device),
                          Ort::Session(*ort_env, model_data_span.data(), model_data_span.size(), so));
  ASSERT_TRUE(std::filesystem::exists(output_model_file));
  std::filesystem::remove(output_model_file);

  // Verify only the first Add node is CPU fallback.
  std::vector<Ort::ConstEpAssignedSubgraph> ep_assigned_subgraphs = scoped.session().GetEpGraphAssignmentInfo();
  for (const auto& ep_assigned_subgraph : ep_assigned_subgraphs) {
    for (const auto& ep_assigned_node : ep_assigned_subgraph.GetNodes()) {
      if (ep_assigned_node.GetName() == "add1") {
        ASSERT_NE(ep_assigned_subgraph.GetEpName(), kQnnExecutionProvider);
      } else {
        ASSERT_EQ(ep_assigned_subgraph.GetEpName(), kQnnExecutionProvider);
      }
    }
  }
}

TEST_F(QnnHTPBackendTests, EPContextMultiSoc_WeightSharing_DefaultHtpConfigs) {
  ProviderOptions per_soc_options;
  per_soc_options["htp_arch"] = "68,73,81";
#ifdef _WIN32
  per_soc_options["soc_model"] = "37,60,88";
#else
  per_soc_options["soc_model"] = "30,43,87";
#endif

  CompileModelWithWeightSharing({per_soc_options, per_soc_options});
}

TEST_F(QnnHTPBackendTests, EPContextMultiSoc_WeightSharing_DefaultHtpConfigs_ThreeSessions) {
  ProviderOptions per_soc_options;
  per_soc_options["htp_arch"] = "68,73,81";
#ifdef _WIN32
  per_soc_options["soc_model"] = "37,60,88";
#else
  per_soc_options["soc_model"] = "30,43,87";
#endif

  CompileModelWithWeightSharing({per_soc_options, per_soc_options, per_soc_options});
}

TEST_F(QnnHTPBackendTests, EPContextMultiSoc_WeightSharing_DiffHtpConfigs) {
  ProviderOptions per_soc_options1;
  per_soc_options1["htp_arch"] = "68,73,81";
#ifdef _WIN32
  per_soc_options1["soc_model"] = "37,60,88";
#else
  per_soc_options1["soc_model"] = "30,43,87";
#endif
  per_soc_options1["vtcm_mb"] = "4";

  ProviderOptions per_soc_options2(per_soc_options1);
  per_soc_options2["vtcm_mb"] = "8";

  CompileModelWithWeightSharing({per_soc_options1, per_soc_options2});
}

TEST_F(QnnHTPBackendTests, EPContextMultiSoc_WeightSharing_MismatchSocTargets) {
  ProviderOptions per_soc_options1;
  per_soc_options1["htp_arch"] = "68,73,81";
#ifdef _WIN32
  per_soc_options1["soc_model"] = "37,60,88";
#else
  per_soc_options1["soc_model"] = "30,43,87";
#endif

  ProviderOptions per_soc_options2;
  per_soc_options2["htp_arch"] = "81,73,68";
#ifdef _WIN32
  per_soc_options2["soc_model"] = "88,60,37";
#else
  per_soc_options2["soc_model"] = "87,43,30";
#endif

  try {
    CompileModelWithWeightSharing({per_soc_options1, per_soc_options2});
    FAIL() << "Expecting mismatched htp_arch/soc_model across sessions should fail.";
  } catch (const Ort::Exception& ex) {
    ASSERT_EQ(ex.GetOrtErrorCode(), ORT_EP_FAIL);
    ASSERT_THAT(
        ex.what(),
        testing::HasSubstr("Mismatched htp_arch/soc_model between current session and the previous one."));
  }
}

TEST_F(QnnHTPBackendTests, EPContextMultiSoc_WeightSharing_LessSocTargets) {
  ProviderOptions per_soc_options1;
  per_soc_options1["htp_arch"] = "68,73,81";
#ifdef _WIN32
  per_soc_options1["soc_model"] = "37,60,88";
#else
  per_soc_options1["soc_model"] = "30,43,87";
#endif

  ProviderOptions per_soc_options2;
  per_soc_options2["htp_arch"] = "73,81";
#ifdef _WIN32
  per_soc_options2["soc_model"] = "60,88";
#else
  per_soc_options2["soc_model"] = "43,87";
#endif

  try {
    CompileModelWithWeightSharing({per_soc_options1, per_soc_options2});
    FAIL() << "Expecting mismatched htp_arch/soc_model across sessions should fail.";
  } catch (const Ort::Exception& ex) {
    ASSERT_EQ(ex.GetOrtErrorCode(), ORT_EP_FAIL);
    ASSERT_THAT(
        ex.what(),
        testing::HasSubstr("Mismatched htp_arch/soc_model between current session and the previous one."));
  }
}

TEST_F(QnnHTPBackendTests, EPContextMultiSoc_WeightSharing_MoreSocTargets) {
  ProviderOptions per_soc_options1;
  per_soc_options1["htp_arch"] = "73,81";
#ifdef _WIN32
  per_soc_options1["soc_model"] = "60,88";
#else
  per_soc_options1["soc_model"] = "43,87";
#endif

  ProviderOptions per_soc_options2;
  per_soc_options2["htp_arch"] = "68,73,81";
#ifdef _WIN32
  per_soc_options2["soc_model"] = "37,60,88";
#else
  per_soc_options2["soc_model"] = "30,43,87";
#endif

  try {
    CompileModelWithWeightSharing({per_soc_options1, per_soc_options2});
    FAIL() << "Expecting mismatched htp_arch/soc_model across sessions should fail.";
  } catch (const Ort::Exception& ex) {
    ASSERT_EQ(ex.GetOrtErrorCode(), ORT_EP_FAIL);
    ASSERT_THAT(
        ex.what(),
        testing::HasSubstr("Mismatched htp_arch/soc_model between current session and the previous one."));
  }
}

TEST_F(QnnHTPBackendTests, EPContextMultiSoc_WeightSharing_MatchHtpArchMismatchSocModel) {
  ProviderOptions per_soc_options1;
  per_soc_options1["htp_arch"] = "73,81";
#ifdef _WIN32
  per_soc_options1["soc_model"] = "60,88";
#else
  per_soc_options1["soc_model"] = "43,87";
#endif

  ProviderOptions per_soc_options2;
  per_soc_options2["htp_arch"] = "73,81";
#ifdef _WIN32
  per_soc_options2["soc_model"] = "43,88";
#else
  per_soc_options2["soc_model"] = "60,87";
#endif

  try {
    CompileModelWithWeightSharing({per_soc_options1, per_soc_options2});
    FAIL() << "Expecting mismatched htp_arch/soc_model across sessions should fail.";
  } catch (const Ort::Exception& ex) {
    ASSERT_EQ(ex.GetOrtErrorCode(), ORT_EP_FAIL);
    ASSERT_THAT(
        ex.what(),
        testing::HasSubstr("Mismatched htp_arch/soc_model between current session and the previous one."));
  }
}

TEST_F(QnnHTPBackendTests, EPContextMultiSoc_WeightSharing_SpillFill) {
  ProviderOptions per_soc_options;
  per_soc_options["htp_arch"] = "68,73,81";
#ifdef _WIN32
  per_soc_options["soc_model"] = "37,60,88";
#else
  per_soc_options["soc_model"] = "30,43,87";
#endif
  per_soc_options["enable_htp_spill_fill_buffer"] = "1";

  CompileModelWithWeightSharing({per_soc_options, per_soc_options});
}

// Check shared context is correctly cleaned-up after failing to write EP Context binary at stop-share session.
TEST_F(QnnHTPBackendTests, EPContextMultiSoc_WeightSharing_ExportFailure_ThenSuccessfulBatch) {
  ProviderOptions per_soc_options;
  per_soc_options["htp_arch"] = "73,81";
#ifdef _WIN32
  per_soc_options["soc_model"] = "60,88";
#else
  per_soc_options["soc_model"] = "43,87";
#endif

  ProviderOptions recovery_options(per_soc_options);
  recovery_options["htp_arch"] = "81,73";
#ifdef _WIN32
  recovery_options["soc_model"] = "88,60";
#else
  recovery_options["soc_model"] = "87,43";
#endif

  const std::filesystem::path test_dir = "multi-soc_weight-sharing_export-failure-recovery";
  std::filesystem::remove_all(test_dir);
  ASSERT_TRUE(std::filesystem::create_directory(test_dir));
  auto cleanup = gsl::finally([&test_dir]() { std::filesystem::remove_all(test_dir); });

  // Keep the EP registered across both batches so unloading cannot hide stale shared state.
  RegisteredEpDeviceUniquePtr registered_ep_device;
  const std::filesystem::path failed_prefix = test_dir / "failed_ctx";
  const std::filesystem::path recovery_prefix = test_dir / "recovery_ctx";
  const std::filesystem::path failed_binary = test_dir / "failed_ctx1_qnn.bin";

  try {
    CompileModelWithWeightSharing({per_soc_options, per_soc_options},
                                  {},
                                  failed_prefix,
                                  &registered_ep_device,
                                  /*fail_binary_write=*/true);
    FAIL() << "Expected writing EP Context binary to fail.";
  } catch (const Ort::Exception& ex) {
    ASSERT_EQ(ex.GetOrtErrorCode(), ORT_EP_FAIL);
    ASSERT_THAT(ex.what(), testing::HasSubstr("Failed to open context cache file."));
  }
  ASSERT_NE(registered_ep_device.get(), nullptr);
  ASSERT_FALSE(std::filesystem::exists(failed_binary));  // The helper removes the blocking directory.

  // Different targets reject any cached snapshots retained from the failed batch.
  // The helper verifies the recovery binary path and every wrapper's ep_cache_context basename.
  CompileModelWithWeightSharing({recovery_options, recovery_options},
                                {},
                                recovery_prefix,
                                &registered_ep_device);
  EXPECT_FALSE(std::filesystem::exists(failed_binary));
}

// Check shared context is correctly cleaned-up after failing to compile at stop-share session.
TEST_F(QnnHTPBackendTests, EPContextMultiSoc_WeightSharing_TargetMismatch_ThenSuccessfulBatch) {
  ProviderOptions per_soc_options;
  per_soc_options["htp_arch"] = "73,81";
#ifdef _WIN32
  per_soc_options["soc_model"] = "60,88";
#else
  per_soc_options["soc_model"] = "43,87";
#endif

  ProviderOptions recovery_options(per_soc_options);
  recovery_options["htp_arch"] = "81,73";
#ifdef _WIN32
  recovery_options["soc_model"] = "88,60";
#else
  recovery_options["soc_model"] = "87,43";
#endif

  const std::filesystem::path test_dir = "multi-soc_weight-sharing_target-mismatch-recovery";
  std::filesystem::remove_all(test_dir);
  ASSERT_TRUE(std::filesystem::create_directory(test_dir));
  auto cleanup = gsl::finally([&test_dir]() { std::filesystem::remove_all(test_dir); });

  // Keep the EP registered across both batches so unloading cannot hide stale shared state.
  RegisteredEpDeviceUniquePtr registered_ep_device;
  const std::filesystem::path failed_prefix = test_dir / "failed_ctx";
  const std::filesystem::path recovery_prefix = test_dir / "recovery_ctx";
  const std::filesystem::path failed_binary = test_dir / "failed_ctx1_qnn.bin";

  try {
    CompileModelWithWeightSharing({per_soc_options, recovery_options}, {}, failed_prefix, &registered_ep_device);
    FAIL() << "Expected mismatched SoC targets to fail.";
  } catch (const Ort::Exception& ex) {
    ASSERT_EQ(ex.GetOrtErrorCode(), ORT_EP_FAIL);
    ASSERT_THAT(ex.what(), testing::HasSubstr("Mismatched htp_arch/soc_model between current session and the previous one."));
  }
  ASSERT_NE(registered_ep_device.get(), nullptr);
  ASSERT_FALSE(std::filesystem::exists(failed_binary));

  // Reversing the targets also makes any retained producer snapshot incompatible with recovery.
  CompileModelWithWeightSharing({recovery_options, recovery_options},
                                {},
                                recovery_prefix,
                                &registered_ep_device);
  EXPECT_FALSE(std::filesystem::exists(failed_binary));
}

namespace {

#ifdef QNN_HTP_NATIVE_BQ_AVAILABLE
void CreateNativeBqModel(const std::filesystem::path& path) {
  ModelTestBuilder builder;

  builder.MakeInput<float>("input0", {1, 32, 64}, 0.0f, 0.25f);
  const auto input = AddQDQNodePair<uint16_t>(builder, "input_qdq", "input0", 0.00390625f, 32768);

  // Symmetric INT4 weights, block size 32, and 32 output channels meet all native-BQ constraints except arch.
  std::vector<uint8_t> packed_weights(32 * 2 * 16);
  for (size_t channel = 0; channel < 32; ++channel) {
    const uint8_t packed = channel % 2 == 0 ? 0x99 : 0x77;
    std::fill_n(packed_weights.begin() + channel * 32, 32, packed);
  }

  builder.MakeInitializer<uint8_t>("B", {32, 2, 16}, packed_weights);
  builder.MakeInitializer<float>("scales", {32, 2}, std::vector<float>(64, 0.125f));
  builder.MakeInitializer<uint8_t>("zero_points", {32, 1}, std::vector<uint8_t>(32, 0x88));
  builder.AddNode("matmul_nbits", "MatMulNBits", {input, "B", "scales", "zero_points"}, {"matmul_output"}, kMSDomain,
                  {builder.MakeScalarAttribute("K", int64_t{64}),
                   builder.MakeScalarAttribute("N", int64_t{32}),
                   builder.MakeScalarAttribute("bits", int64_t{4}),
                   builder.MakeScalarAttribute("block_size", int64_t{32}),
                   builder.MakeScalarAttribute("accuracy_level", int64_t{1})});

  AddQDQNodePairWithOutputAsGraphOutput<uint16_t>(builder, "output_qdq", "matmul_output", 0.015625f, 32768);

  auto* ms_opset = builder.model_.add_opset_import();
  ms_opset->set_domain(kMSDomain);
  ms_opset->set_version(1);

  auto* opset_id = builder.model_.add_opset_import();
  opset_id->set_domain("");
  opset_id->set_version(21);
  builder.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  std::ofstream stream(path, std::ios::binary);
  ASSERT_TRUE(stream.good());
  ASSERT_TRUE(builder.model_.SerializeToOstream(&stream));
}
#endif  // QNN_HTP_NATIVE_BQ_AVAILABLE

}  // namespace

TEST_F(QnnHTPBackendTests, EPContextMultiSoc_WeightSharing_NativeBQ_HeterogeneousArchs) {
#ifndef QNN_HTP_NATIVE_BQ_AVAILABLE
  GTEST_SKIP() << "Native BQ requires QNN API version >= 2.40.";
#else
  ProviderOptions per_soc_options;
  per_soc_options["htp_arch"] = "73,81";
#ifdef _WIN32
  per_soc_options["soc_model"] = "60,88";
#else
  per_soc_options["soc_model"] = "43,87";
#endif

  const std::filesystem::path dump_dir = "multi-soc_weight-sharing_native-bq";
  std::filesystem::remove_all(dump_dir);
  ASSERT_TRUE(std::filesystem::create_directory(dump_dir));
  auto cleanup = gsl::finally([&dump_dir]() { std::filesystem::remove_all(dump_dir); });

  per_soc_options["dump_json_qnn_graph"] = "1";
  per_soc_options["json_qnn_graph_dir"] = dump_dir.string();

  const std::filesystem::path model_file = dump_dir / "model_bq.onnx";
  CreateNativeBqModel(model_file);

  WeightSharingWarningLog log;
  CompileModelWithWeightSharing({per_soc_options, per_soc_options}, model_file, {}, nullptr, false, &log);

  {
    std::lock_guard<std::mutex> lock(log.mutex);
    ASSERT_EQ(log.messages.size(), 2);
    for (size_t idx = 0; idx < log.messages.size(); ++idx) {
      EXPECT_THAT(log.messages[idx], testing::HasSubstr("model " + std::to_string(idx + 1) + " of 2"));
      EXPECT_THAT(log.messages[idx], testing::HasSubstr("MatMulNBits"));
      EXPECT_THAT(log.messages[idx], testing::HasSubstr("native BQ"));
      EXPECT_THAT(log.messages[idx], testing::HasSubstr("translation arch 73; affected target archs [81]"));
    }
  }

  AssertOpInQnnGraph(dump_dir, "Quantize", 2);
  AssertOpInQnnGraph(dump_dir, "Dequantize", 2);
#endif
}

#endif  // !defined(__aarch64__) && !defined(_M_ARM64) && !defined(_M_ARM64EC)) || defined(QNN_HTP_CROSS_DEVICE_PREPARE_AVAILABLE)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
