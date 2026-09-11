// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <cstddef>
#include <filesystem>
#include <string>
#include <unordered_map>

#include "gsl/gsl"
#include "gtest/gtest.h"
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

#if !defined(__aarch64__) && !defined(_M_ARM64)

namespace {

void CompileModelWithPerSocOptions(const ProviderOptions& per_soc_options,
                                   bool is_embed_mode = true) {
  ProviderOptions provider_options(per_soc_options);
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";  // Avoid IO QDQ CPU fallback.
  provider_options["num_graph_prepare_threads"] = "1";

  const ORTCHAR_T* input_model_file = ORT_MODEL_FOLDER "nhwc_resize_sizes_opset18.quant.onnx";
  std::filesystem::path output_model_file("model_ctx.onnx");
  std::filesystem::path output_bin_file("model_ctx_qnn.bin");
  std::filesystem::remove(output_model_file);
  std::filesystem::remove(output_bin_file);

  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, is_embed_mode ? "1" : "0");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, output_model_file.string().c_str());

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, input_model_file, so));
  ASSERT_TRUE(std::filesystem::exists(output_model_file));
  if (!is_embed_mode) {
    ASSERT_TRUE(std::filesystem::exists(output_bin_file));
  }

  std::filesystem::remove(output_model_file);
  std::filesystem::remove(output_bin_file);
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

#endif  // !defined(__aarch64__) && !defined(_M_ARM64)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
