// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

#include "core/graph/onnx_protobuf.h"  // ONNX_NAMESPACE::AttributeProto
#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

constexpr float kDefaultRopeOpToleranceFp32 = 1e-4f;

namespace {

// Build an int64 AttributeProto without relying on onnxruntime::utils::MakeAttribute.
// This avoids link-time dependency on node_attr_utils.cc.
inline ONNX_NAMESPACE::AttributeProto MakeIntAttr(std::string name, int64_t value) {
  ONNX_NAMESPACE::AttributeProto a;
  a.set_name(std::move(name));
  a.set_type(ONNX_NAMESPACE::AttributeProto::INT);
  a.set_i(value);
  return a;
}

}  // namespace

template <typename DataType = float>
static void RunRopeOpTest(const TestInputDef<DataType>& input_def,
                          const TestInputDef<int64_t>& position_ids_def,
                          const TestInputDef<DataType>& cos_cache_def,
                          const TestInputDef<DataType>& sin_cache_def,
                          const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                          int opset_version,
                          ExpectedEPNodeAssignment expected_ep_assignment,
                          float fp32_abs_err = kDefaultRopeOpToleranceFp32) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  // Custom model builder for RotaryEmbedding with 4 inputs
  auto build_test_case = [&](ModelTestBuilder& builder) {
    // Create all 4 inputs with proper names
    MakeTestInput<DataType>(builder, "input", input_def);
    MakeTestInput<int64_t>(builder, "position_ids", position_ids_def);
    MakeTestInput<DataType>(builder, "cos_cache", cos_cache_def);
    MakeTestInput<DataType>(builder, "sin_cache", sin_cache_def);

    // Create output
    builder.MakeOutput("output");

    // Add RotaryEmbedding node with all 4 inputs
    builder.AddNode("",
                    "RotaryEmbedding",
                    {"input", "position_ids", "cos_cache", "sin_cache"},
                    {"output"},
                    kMSDomain,
                    attrs);
  };

  RunQnnModelTest(build_test_case,
                  provider_options,
                  opset_version,
                  expected_ep_assignment,
                  fp32_abs_err);
}

// Basic test to ensure build passes
TEST_F(QnnHTPBackendTests, RotaryEmbedding_Basic) {
  constexpr int64_t batch_size = 1;
  constexpr int64_t num_heads = 2;
  constexpr int64_t seq_len = 4;
  constexpr int64_t head_size = 8;
  constexpr int64_t rotary_dim = head_size;

  std::vector<float> input_data =
      GetFloatDataInRange(-1.0f, 1.0f, batch_size * num_heads * seq_len * head_size);
  std::vector<float> cos_cache =
      GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));
  std::vector<float> sin_cache =
      GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));
  std::vector<int64_t> position_ids = {0, 1, 2, 3};

  RunRopeOpTest<float>(
      TestInputDef<float>({batch_size, num_heads, seq_len, head_size}, false, input_data),
      TestInputDef<int64_t>({batch_size, seq_len}, false, position_ids),
      TestInputDef<float>({seq_len, rotary_dim / 2}, false, cos_cache),
      TestInputDef<float>({seq_len, rotary_dim / 2}, false, sin_cache),
      {MakeIntAttr("interleaved", 0),
       MakeIntAttr("num_heads", num_heads),
       MakeIntAttr("rotary_embedding_dim", rotary_dim)},
      1,
      ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
