// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Helper function to build a MultiHeadAttention model with separate Q, K, V inputs
static GetTestModelFn BuildMultiHeadAttentionTestCase(const TestInputDef<float>& query_def,
                                                      const TestInputDef<float>& key_def,
                                                      const TestInputDef<float>& value_def,
                                                      const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs) {
  return [query_def, key_def, value_def, attrs](ModelTestBuilder& builder) {
    NodeArg* query = MakeTestInput(builder, query_def);
    NodeArg* key = MakeTestInput(builder, key_def);
    NodeArg* value = MakeTestInput(builder, value_def);

    NodeArg* output = builder.MakeOutput();
    Node& mha_node = builder.AddNode("MultiHeadAttention", {query, key, value}, {output}, kMSDomain);

    for (const auto& attr : attrs) {
      mha_node.AddAttributeProto(attr);
    }
  };
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Test MultiHeadAttention with basic separate Q, K, V inputs on CPU backend
TEST_F(QnnCPUBackendTests, MultiHeadAttention_SeparateQKV_Basic) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";

  const int64_t batch_size = 1;
  const int64_t sequence_length = 4;
  const int64_t hidden_size = 12;
  const int64_t num_heads = 3;

  std::vector<ONNX_NAMESPACE::AttributeProto> attrs = {
      utils::MakeAttribute("num_heads", num_heads)};

  RunQnnModelTest(BuildMultiHeadAttentionTestCase(
                      TestInputDef<float>({batch_size, sequence_length, hidden_size}, false,
                                          GetFloatDataInRange(-1.0f, 1.0f, batch_size * sequence_length * hidden_size)),
                      TestInputDef<float>({batch_size, sequence_length, hidden_size}, false,
                                          GetFloatDataInRange(-1.0f, 1.0f, batch_size * sequence_length * hidden_size)),
                      TestInputDef<float>({batch_size, sequence_length, hidden_size}, false,
                                          GetFloatDataInRange(-1.0f, 1.0f, batch_size * sequence_length * hidden_size)),
                      attrs),
                  provider_options,
                  13,  // opset
                  ExpectedEPNodeAssignment::All);
}

// Test MultiHeadAttention with different sequence lengths for Q and KV
TEST_F(QnnCPUBackendTests, MultiHeadAttention_SeparateQKV_DifferentSeqLen) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";

  const int64_t batch_size = 2;
  const int64_t q_sequence_length = 3;
  const int64_t kv_sequence_length = 5;
  const int64_t hidden_size = 16;
  const int64_t num_heads = 4;

  std::vector<ONNX_NAMESPACE::AttributeProto> attrs = {
      utils::MakeAttribute("num_heads", num_heads)};

  RunQnnModelTest(BuildMultiHeadAttentionTestCase(
                      TestInputDef<float>({batch_size, q_sequence_length, hidden_size}, false,
                                          GetFloatDataInRange(-1.0f, 1.0f, batch_size * q_sequence_length * hidden_size)),
                      TestInputDef<float>({batch_size, kv_sequence_length, hidden_size}, false,
                                          GetFloatDataInRange(-1.0f, 1.0f, batch_size * kv_sequence_length * hidden_size)),
                      TestInputDef<float>({batch_size, kv_sequence_length, hidden_size}, false,
                                          GetFloatDataInRange(-1.0f, 1.0f, batch_size * kv_sequence_length * hidden_size)),
                      attrs),
                  provider_options,
                  13,  // opset
                  ExpectedEPNodeAssignment::All);
}

// Test MultiHeadAttention with custom scale attribute
TEST_F(QnnCPUBackendTests, MultiHeadAttention_CustomScale) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";

  const int64_t batch_size = 1;
  const int64_t sequence_length = 4;
  const int64_t hidden_size = 12;
  const int64_t num_heads = 3;
  const float scale = 0.5f;

  std::vector<ONNX_NAMESPACE::AttributeProto> attrs = {
      utils::MakeAttribute("num_heads", num_heads),
      utils::MakeAttribute("scale", scale)};

  RunQnnModelTest(BuildMultiHeadAttentionTestCase(
                      TestInputDef<float>({batch_size, sequence_length, hidden_size}, false,
                                          GetFloatDataInRange(-1.0f, 1.0f, batch_size * sequence_length * hidden_size)),
                      TestInputDef<float>({batch_size, sequence_length, hidden_size}, false,
                                          GetFloatDataInRange(-1.0f, 1.0f, batch_size * sequence_length * hidden_size)),
                      TestInputDef<float>({batch_size, sequence_length, hidden_size}, false,
                                          GetFloatDataInRange(-1.0f, 1.0f, batch_size * sequence_length * hidden_size)),
                      attrs),
                  provider_options,
                  13,  // opset
                  ExpectedEPNodeAssignment::All);
}

// Test MultiHeadAttention with larger dimensions
TEST_F(QnnCPUBackendTests, MultiHeadAttention_LargeDimensions) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";

  const int64_t batch_size = 2;
  const int64_t sequence_length = 8;
  const int64_t hidden_size = 64;
  const int64_t num_heads = 8;

  std::vector<ONNX_NAMESPACE::AttributeProto> attrs = {
      utils::MakeAttribute("num_heads", num_heads)};

  RunQnnModelTest(BuildMultiHeadAttentionTestCase(
                      TestInputDef<float>({batch_size, sequence_length, hidden_size}, false,
                                          GetFloatDataInRange(-1.0f, 1.0f, batch_size * sequence_length * hidden_size)),
                      TestInputDef<float>({batch_size, sequence_length, hidden_size}, false,
                                          GetFloatDataInRange(-1.0f, 1.0f, batch_size * sequence_length * hidden_size)),
                      TestInputDef<float>({batch_size, sequence_length, hidden_size}, false,
                                          GetFloatDataInRange(-1.0f, 1.0f, batch_size * sequence_length * hidden_size)),
                      attrs),
                  provider_options,
                  13,  // opset
                  ExpectedEPNodeAssignment::All);
}

// Test MultiHeadAttention on HTP backend with FP16 precision
TEST_F(QnnHTPBackendTests, MultiHeadAttention_FP16) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["enable_htp_fp16_precision"] = "1";
#if defined(__linux__) && !defined(__aarch64__)
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif

  const int64_t batch_size = 1;
  const int64_t sequence_length = 4;
  const int64_t hidden_size = 12;
  const int64_t num_heads = 3;

  std::vector<ONNX_NAMESPACE::AttributeProto> attrs = {
      utils::MakeAttribute("num_heads", num_heads)};

  RunQnnModelTest(BuildMultiHeadAttentionTestCase(
                      TestInputDef<float>({batch_size, sequence_length, hidden_size}, false,
                                          GetFloatDataInRange(-1.0f, 1.0f, batch_size * sequence_length * hidden_size)),
                      TestInputDef<float>({batch_size, sequence_length, hidden_size}, false,
                                          GetFloatDataInRange(-1.0f, 1.0f, batch_size * sequence_length * hidden_size)),
                      TestInputDef<float>({batch_size, sequence_length, hidden_size}, false,
                                          GetFloatDataInRange(-1.0f, 1.0f, batch_size * sequence_length * hidden_size)),
                      attrs),
                  provider_options,
                  13,  // opset
                  ExpectedEPNodeAssignment::All,
                  1e-2f);  // Relax tolerance for FP16
}

// Test MultiHeadAttention with single head (essentially scaled dot-product attention)
TEST_F(QnnCPUBackendTests, MultiHeadAttention_SingleHead) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";

  const int64_t batch_size = 1;
  const int64_t sequence_length = 4;
  const int64_t hidden_size = 8;
  const int64_t num_heads = 1;

  std::vector<ONNX_NAMESPACE::AttributeProto> attrs = {
      utils::MakeAttribute("num_heads", num_heads)};

  RunQnnModelTest(BuildMultiHeadAttentionTestCase(
                      TestInputDef<float>({batch_size, sequence_length, hidden_size}, false,
                                          GetFloatDataInRange(-1.0f, 1.0f, batch_size * sequence_length * hidden_size)),
                      TestInputDef<float>({batch_size, sequence_length, hidden_size}, false,
                                          GetFloatDataInRange(-1.0f, 1.0f, batch_size * sequence_length * hidden_size)),
                      TestInputDef<float>({batch_size, sequence_length, hidden_size}, false,
                                          GetFloatDataInRange(-1.0f, 1.0f, batch_size * sequence_length * hidden_size)),
                      attrs),
                  provider_options,
                  13,  // opset
                  ExpectedEPNodeAssignment::All);
}

// Test MultiHeadAttention with batch size > 1
TEST_F(QnnCPUBackendTests, MultiHeadAttention_BatchSize4) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";

  const int64_t batch_size = 4;
  const int64_t sequence_length = 6;
  const int64_t hidden_size = 24;
  const int64_t num_heads = 4;

  std::vector<ONNX_NAMESPACE::AttributeProto> attrs = {
      utils::MakeAttribute("num_heads", num_heads)};

  RunQnnModelTest(BuildMultiHeadAttentionTestCase(
                      TestInputDef<float>({batch_size, sequence_length, hidden_size}, false,
                                          GetFloatDataInRange(-1.0f, 1.0f, batch_size * sequence_length * hidden_size)),
                      TestInputDef<float>({batch_size, sequence_length, hidden_size}, false,
                                          GetFloatDataInRange(-1.0f, 1.0f, batch_size * sequence_length * hidden_size)),
                      TestInputDef<float>({batch_size, sequence_length, hidden_size}, false,
                                          GetFloatDataInRange(-1.0f, 1.0f, batch_size * sequence_length * hidden_size)),
                      attrs),
                  provider_options,
                  13,  // opset
                  ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
