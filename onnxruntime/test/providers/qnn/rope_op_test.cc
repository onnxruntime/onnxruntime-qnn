// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

constexpr float kDefaultRopeOpToleranceFp16 = 5e-3f;

template <typename DataType>
static void RunRopeOpTest(const TestInputDef<DataType>& input_def,
                          const TestInputDef<int64_t>& position_ids_def,
                          const TestInputDef<DataType>& cos_cache_def,
                          const TestInputDef<DataType>& sin_cache_def,
                          const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                          int opset_version,
                          ExpectedEPNodeAssignment expected_ep_assignment,
                          float abs_err) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  GetTestModelFn model_builder =
      [input_def, position_ids_def, cos_cache_def, sin_cache_def, attrs](ModelTestBuilder& builder) {
        MakeTestInput(builder, "input", input_def);
        MakeTestInput(builder, "position_ids", position_ids_def);
        MakeTestInput(builder, "cos_cache", cos_cache_def);
        MakeTestInput(builder, "sin_cache", sin_cache_def);

        builder.MakeOutput("output");

        builder.AddNode("RotaryEmbedding",
                        "RotaryEmbedding",
                        {"input", "position_ids", "cos_cache", "sin_cache"},
                        {"output"},
                        kMSDomain,
                        attrs);
      };

  RunQnnModelTest(model_builder,
                  provider_options,
                  opset_version,
                  expected_ep_assignment,
                  abs_err);
}

// Basic test with FP16 data type (QNN EP only supports FP16 for RotaryEmbedding)
TEST_F(QnnHTPBackendTests, RotaryEmbedding_Basic) {
#if defined(_WIN32)
  // Skip test on Windows if HTP FP16 is not supported (V68 and below)
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
#endif

  constexpr int64_t batch_size = 1;
  constexpr int64_t num_heads = 2;
  constexpr int64_t seq_len = 4;
  constexpr int64_t head_size = 8;
  constexpr int64_t rotary_dim = head_size;

  // Generate FP32 data and convert to FP16
  // Note: cos_cache and sin_cache dimensions are [max_seq_len, rotary_dim/2]
  // because RoPE uses pair-wise rotation
  auto input_f32 = GetFloatDataInRange(-1.0f, 1.0f, batch_size * num_heads * seq_len * head_size);
  auto cos_f32 = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));
  auto sin_f32 = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));

  // Convert to FP16 using the helper function
  TestInputDef<float> input_def_f32({batch_size, num_heads, seq_len, head_size}, false, input_f32);
  TestInputDef<float> cos_def_f32({seq_len, rotary_dim / 2}, false, cos_f32);
  TestInputDef<float> sin_def_f32({seq_len, rotary_dim / 2}, false, sin_f32);

  TestInputDef<Ort::Float16_t> input_def_fp16 = ConvertToFP16InputDef(input_def_f32);
  TestInputDef<Ort::Float16_t> cos_def_fp16 = ConvertToFP16InputDef(cos_def_f32);
  TestInputDef<Ort::Float16_t> sin_def_fp16 = ConvertToFP16InputDef(sin_def_f32);

  std::vector<int64_t> position_ids = {0, 1, 2, 3};

  RunRopeOpTest<Ort::Float16_t>(
      input_def_fp16,
      TestInputDef<int64_t>({batch_size, seq_len}, false, position_ids),
      cos_def_fp16,
      sin_def_fp16,
      {test::MakeAttribute("interleaved", int64_t{0}),
       test::MakeAttribute("num_heads", num_heads),
       test::MakeAttribute("rotary_embedding_dim", rotary_dim)},
      /*opset_version*/ 7,  // Use opset 7 to avoid legacy warning
      ExpectedEPNodeAssignment::All,
      kDefaultRopeOpToleranceFp16);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
