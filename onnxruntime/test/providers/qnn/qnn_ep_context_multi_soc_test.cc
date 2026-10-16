// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <cstddef>
#include <filesystem>
#include <string>
#include <unordered_map>

#include <gsl/gsl>
#include <gsl/util>
#include <gtest/gtest.h>
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_session_options_config_keys.h"

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/model_test_builder.h"
#include "test/util/include/api_asserts.h"
#include "test/util/include/asserts.h"

#define ORT_MODEL_FOLDER ORT_TSTR("testdata/")

// Defined in test_main.cc.
extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

#if (!defined(__aarch64__) && !defined(_M_ARM64)) || defined(QNN_HTP_CROSS_DEVICE_PREPARE_AVAILABLE)

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

void CompileModelWithWeightSharing(const ProviderOptions& per_soc_options1, const ProviderOptions& per_soc_options2) {
  const ORTCHAR_T* input_model_file1 = ORT_MODEL_FOLDER "nhwc_resize_sizes_opset18.quant.onnx";
  const ORTCHAR_T* input_model_file2 = ORT_TSTR("./input_model_file2.onnx");
  const ORTCHAR_T* output_model_file1 = ORT_TSTR("model_ctx1.onnx");
  const ORTCHAR_T* output_model_file2 = ORT_TSTR("model_ctx2.onnx");
  const ORTCHAR_T* output_bin_file = ORT_TSTR("model_ctx1_qnn.bin");

  // Compile the same model twice into the same QNN context would result in duplicate graph name.
  // As a result, copy the model to pretend they are different.
  ASSERT_TRUE(std::filesystem::copy_file(input_model_file1,
                                         input_model_file2,
                                         std::filesystem::copy_options::overwrite_existing));
  std::filesystem::remove(output_model_file1);
  std::filesystem::remove(output_model_file2);
  std::filesystem::remove(output_bin_file);
  auto cleanup = gsl::finally([&input_model_file2, &output_model_file1, &output_model_file2, &output_bin_file]() {
    std::filesystem::remove(input_model_file2);
    std::filesystem::remove(output_model_file1);
    std::filesystem::remove(output_model_file2);
    std::filesystem::remove(output_bin_file);
  });

  auto get_provider_options = [](const ProviderOptions& per_soc_options) {
    ProviderOptions provider_options(per_soc_options);
    provider_options["backend_type"] = "htp";
    provider_options["offload_graph_io_quantization"] = "0";  // Avoid IO QDQ CPU fallback.
    provider_options["num_graph_prepare_threads"] = "1";
#if defined(QNN_HTP_CROSS_DEVICE_PREPARE_AVAILABLE)
    provider_options["enable_htp_cross_device_prepare"] = "1";
#endif

    return provider_options;
  };

  {
    // Create the first session with "ep.share_ep_contexts" set.
    Ort::SessionOptions so1;
    so1.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so1.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "0");
    so1.AddConfigEntry(kOrtSessionOptionEpContextFilePath, std::filesystem::path(output_model_file1).string().c_str());
    so1.AddConfigEntry(kOrtSessionOptionShareEpContexts, "1");

    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so1, kQnnExecutionProvider, get_provider_options(per_soc_options1));

    ScopedOrtSession scoped1(std::move(registered_ep_device), Ort::Session(*ort_env, input_model_file1, so1));

    // Create the second session with both "ep.share_ep_contexts" and "ep.stop_share_ep_contexts" set.
    Ort::SessionOptions so2;
    so2.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so2.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "0");
    so2.AddConfigEntry(kOrtSessionOptionEpContextFilePath, std::filesystem::path(output_model_file2).string().c_str());
    so2.AddConfigEntry(kOrtSessionOptionShareEpContexts, "1");
    so2.AddConfigEntry(kOrtSessionOptionStopShareEpContexts, "1");

    // Manually append EP for different provider options.
    so2.AppendExecutionProvider_V2(*ort_env,
                                   {Ort::ConstEpDevice(scoped1.ep_device())},
                                   get_provider_options(per_soc_options2));

    Ort::Session session2(*ort_env, input_model_file2, so2);

    ASSERT_TRUE(std::filesystem::exists(output_model_file1));
    ASSERT_TRUE(std::filesystem::exists(output_model_file2));
    ASSERT_TRUE(std::filesystem::exists(output_bin_file));
    ASSERT_FALSE(std::filesystem::exists("model_ctx2_qnn.bin"));
  }

#if defined(QNN_HTP_CROSS_DEVICE_PREPARE_AVAILABLE)
  {
    ProviderOptions provider_options = {{"backend_type", "htp"}, {"disable_file_mapped_weights", "1"}};
    Ort::SessionOptions so;
    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

    so.AddConfigEntry(kOrtSessionOptionShareEpContexts, "1");
    ScopedOrtSession scoped1(std::move(registered_ep_device), Ort::Session(*ort_env, output_model_file1, so));

    so.AddConfigEntry(kOrtSessionOptionStopShareEpContexts, "1");
    Ort::Session session2(*ort_env, output_model_file2, so);
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

  CompileModelWithWeightSharing(per_soc_options, per_soc_options);
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

  CompileModelWithWeightSharing(per_soc_options1, per_soc_options2);
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
    CompileModelWithWeightSharing(per_soc_options1, per_soc_options2);
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
    CompileModelWithWeightSharing(per_soc_options1, per_soc_options2);
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
    CompileModelWithWeightSharing(per_soc_options1, per_soc_options2);
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
    CompileModelWithWeightSharing(per_soc_options1, per_soc_options2);
    FAIL() << "Expecting mismatched htp_arch/soc_model across sessions should fail.";
  } catch (const Ort::Exception& ex) {
    ASSERT_EQ(ex.GetOrtErrorCode(), ORT_EP_FAIL);
    ASSERT_THAT(
        ex.what(),
        testing::HasSubstr("Mismatched htp_arch/soc_model between current session and the previous one."));
  }
}

#endif  // !defined(__aarch64__) && !defined(_M_ARM64) || defined(QNN_HTP_CROSS_DEVICE_PREPARE_AVAILABLE)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
