// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <string>

#include "test/providers/qnn/qnn_test_utils.h"

#include "core/graph/onnx_protobuf.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Builds a QDQ Reshape model with mismatched input/output quantization types.
// input -> Q(uint16) -> DQ(uint16) -> Reshape -> Q(uint8) -> DQ(uint8) -> output
static void BuildQDQReshapeDtypeMismatchModel(ModelTestBuilder& builder) {
  constexpr int64_t num_elements = 12;
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, num_elements);
  TestInputDef<float> input_def({1, 3, 2, 2}, false, input_data);
  MakeTestInput(builder, "input", input_def);

  float scale_u16 = 0.003f;
  uint16_t zp_u16 = 32768;
  std::string dq_out = AddQDQNodePair<uint16_t>(builder, "qdq_in", "input", scale_u16, zp_u16,
                                                true /* use_ms_domain */);

  builder.Make1DInitializer<int64_t>("shape", {1, num_elements});
  builder.AddNode("reshape_node", "Reshape", {dq_out, "shape"}, {"reshape_out"}, "");

  float scale_u8 = 0.08f;
  uint8_t zp_u8 = 128;
  AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, "qdq_out", "reshape_out", scale_u8, zp_u8);
}

// Test that QDQ Reshape with mismatched DQ input type (uint16) and Q output type (uint8)
// is NOT selected as a single QDQ node group. Before the dtype fix, GetNodeIODataType could
// silently return -1 on failure, causing the selector to incorrectly match types (-1 == -1).
// Verify by counting EP nodes: if incorrectly fused, count == 1; unfused should be > 1.
TEST_F(QnnHTPBackendTests, Reshape_QDQ_DtypeMismatch_NotSelected) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  constexpr int opset_version = 19;
  const std::unordered_map<std::string, int> domain_to_version = {{"", opset_version}, {kMSDomain, 1}};

  ModelTestBuilder helper;
  BuildQDQReshapeDtypeMismatchModel(helper);
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  std::string model_data;
  helper.model_.SerializeToString(&model_data);

  RegisteredEpDeviceUniquePtr registered_ep_device;
  const std::string registration_name = "QNNExecutionProvider";
  Ort::SessionOptions session_options;
  session_options.AddConfigEntry(kOrtSessionOptionsRecordEpGraphAssignmentInfo, "1");
  session_options.SetLogSeverityLevel(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR);

  TryEnableQNNSaver(provider_options);
  RegisterQnnEpLibrary(registered_ep_device, session_options, registration_name, provider_options);

  Ort::Session session(*GetOrtEnv(), model_data.data(), static_cast<int>(model_data.size()), session_options);

  size_t num_qnn_nodes = CountAssignedNodes(session, registration_name);
  // If QDQ incorrectly fused around Reshape, there would be only 1 node.
  // With correct type mismatch rejection, individual ops remain unfused (> 1 node).
  ASSERT_NE(num_qnn_nodes, static_cast<size_t>(1))
      << "QDQ group with mismatched types should NOT be fused into a single node";
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
