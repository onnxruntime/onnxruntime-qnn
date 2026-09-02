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

// 4-input form: [X, cos_cache, sin_cache, position_ids]
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
        MakeTestInput(builder, "cos_cache", cos_cache_def);
        MakeTestInput(builder, "sin_cache", sin_cache_def);
        MakeTestInput(builder, "position_ids", position_ids_def);

        builder.MakeOutput("output");

        builder.AddNode("RotaryEmbedding",
                        "RotaryEmbedding",
                        {"input", "cos_cache", "sin_cache", "position_ids"},
                        {"output"},
                        kOnnxDomain,
                        attrs);
      };

  RunQnnModelTest(model_builder,
                  provider_options,
                  opset_version,
                  EPVerificationParams{expected_ep_assignment, ElementwiseAbsoluteVerifier(abs_err)});
}

// 3-input form: [X, cos_cache, sin_cache] — no position_ids.
// cos/sin cache must be 3D [B, S, rotary_dim/2] in this mode.
template <typename DataType>
static void RunRopeOpTestNoPositionIds(const TestInputDef<DataType>& input_def,
                                       const TestInputDef<DataType>& cos_cache_def,
                                       const TestInputDef<DataType>& sin_cache_def,
                                       const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                       int opset_version,
                                       ExpectedEPNodeAssignment expected_ep_assignment,
                                       float abs_err) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  GetTestModelFn model_builder =
      [input_def, cos_cache_def, sin_cache_def, attrs](ModelTestBuilder& builder) {
        MakeTestInput(builder, "input", input_def);
        MakeTestInput(builder, "cos_cache", cos_cache_def);
        MakeTestInput(builder, "sin_cache", sin_cache_def);

        builder.MakeOutput("output");

        builder.AddNode("RotaryEmbedding",
                        "RotaryEmbedding",
                        {"input", "cos_cache", "sin_cache"},
                        {"output"},
                        kOnnxDomain,
                        attrs);
      };

  RunQnnModelTest(model_builder,
                  provider_options,
                  opset_version,
                  EPVerificationParams{expected_ep_assignment, ElementwiseAbsoluteVerifier(abs_err)});
}

// Basic 4D input with position_ids, interleaved=0, full rotation.
TEST_F(QnnHTPBackendTests, RotaryEmbedding_Basic) {
#if defined(_WIN32)
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
#endif

  constexpr int64_t batch_size = 1;
  constexpr int64_t num_heads = 2;
  constexpr int64_t seq_len = 4;
  constexpr int64_t head_size = 8;
  constexpr int64_t rotary_dim = head_size;

  auto input_f32 = GetFloatDataInRange(-1.0f, 1.0f, batch_size * num_heads * seq_len * head_size);
  auto cos_f32 = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));
  auto sin_f32 = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));

  TestInputDef<Ort::Float16_t> input_def_fp16 = ConvertToFP16InputDef(
      TestInputDef<float>({batch_size, num_heads, seq_len, head_size}, false, input_f32));
  TestInputDef<Ort::Float16_t> cos_def_fp16 = ConvertToFP16InputDef(
      TestInputDef<float>({seq_len, rotary_dim / 2}, false, cos_f32));
  TestInputDef<Ort::Float16_t> sin_def_fp16 = ConvertToFP16InputDef(
      TestInputDef<float>({seq_len, rotary_dim / 2}, false, sin_f32));

  RunRopeOpTest<Ort::Float16_t>(
      input_def_fp16,
      TestInputDef<int64_t>({batch_size, seq_len}, false, std::vector<int64_t>{0, 1, 2, 3}),
      cos_def_fp16,
      sin_def_fp16,
      {test::MakeAttribute("interleaved", int64_t{0}),
       test::MakeAttribute("num_heads", num_heads),
       test::MakeAttribute("rotary_embedding_dim", rotary_dim)},
      /*opset_version*/ 23,
      ExpectedEPNodeAssignment::All,
      kDefaultRopeOpToleranceFp16);
}

// Rank-3 input [B, S, NH*HS] — exercises reshape/transpose adapter.
TEST_F(QnnHTPBackendTests, RotaryEmbedding_Rank3Input) {
#if defined(_WIN32)
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
#endif

  constexpr int64_t batch_size = 1;
  constexpr int64_t num_heads = 2;
  constexpr int64_t seq_len = 4;
  constexpr int64_t head_size = 8;
  constexpr int64_t hidden_size = num_heads * head_size;
  constexpr int64_t rotary_dim = head_size;

  auto input_f32 = GetFloatDataInRange(-1.0f, 1.0f, batch_size * seq_len * hidden_size);
  auto cos_f32 = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));
  auto sin_f32 = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));

  TestInputDef<Ort::Float16_t> input_def_fp16 = ConvertToFP16InputDef(
      TestInputDef<float>({batch_size, seq_len, hidden_size}, false, input_f32));
  TestInputDef<Ort::Float16_t> cos_def_fp16 = ConvertToFP16InputDef(
      TestInputDef<float>({seq_len, rotary_dim / 2}, false, cos_f32));
  TestInputDef<Ort::Float16_t> sin_def_fp16 = ConvertToFP16InputDef(
      TestInputDef<float>({seq_len, rotary_dim / 2}, false, sin_f32));

  RunRopeOpTest<Ort::Float16_t>(
      input_def_fp16,
      TestInputDef<int64_t>({batch_size, seq_len}, false, std::vector<int64_t>{0, 1, 2, 3}),
      cos_def_fp16,
      sin_def_fp16,
      {test::MakeAttribute("interleaved", int64_t{0}),
       test::MakeAttribute("num_heads", num_heads),
       test::MakeAttribute("rotary_embedding_dim", rotary_dim)},
      /*opset_version*/ 23,
      ExpectedEPNodeAssignment::All,
      kDefaultRopeOpToleranceFp16);
}

// No position_ids (3-input form) — cos/sin cache is 3D [B, S, rotary_dim/2].
TEST_F(QnnHTPBackendTests, RotaryEmbedding_NoPositionIds) {
#if defined(_WIN32)
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
#endif

  constexpr int64_t batch_size = 1;
  constexpr int64_t num_heads = 2;
  constexpr int64_t seq_len = 4;
  constexpr int64_t head_size = 8;
  constexpr int64_t rotary_dim = head_size;

  auto input_f32 = GetFloatDataInRange(-1.0f, 1.0f, batch_size * num_heads * seq_len * head_size);
  // 3D cache: [B, S, rotary_dim/2] when position_ids not provided
  auto cos_f32 = GetFloatDataInRange(-1.0f, 1.0f, batch_size * seq_len * (rotary_dim / 2));
  auto sin_f32 = GetFloatDataInRange(-1.0f, 1.0f, batch_size * seq_len * (rotary_dim / 2));

  TestInputDef<Ort::Float16_t> input_def_fp16 = ConvertToFP16InputDef(
      TestInputDef<float>({batch_size, num_heads, seq_len, head_size}, false, input_f32));
  TestInputDef<Ort::Float16_t> cos_def_fp16 = ConvertToFP16InputDef(
      TestInputDef<float>({batch_size, seq_len, rotary_dim / 2}, false, cos_f32));
  TestInputDef<Ort::Float16_t> sin_def_fp16 = ConvertToFP16InputDef(
      TestInputDef<float>({batch_size, seq_len, rotary_dim / 2}, false, sin_f32));

  RunRopeOpTestNoPositionIds<Ort::Float16_t>(
      input_def_fp16,
      cos_def_fp16,
      sin_def_fp16,
      {test::MakeAttribute("interleaved", int64_t{0}),
       test::MakeAttribute("num_heads", num_heads),
       test::MakeAttribute("rotary_embedding_dim", rotary_dim)},
      /*opset_version*/ 23,
      ExpectedEPNodeAssignment::All,
      kDefaultRopeOpToleranceFp16);
}

// interleaved=1 — alternating even/odd index split.
TEST_F(QnnHTPBackendTests, RotaryEmbedding_Interleaved) {
#if defined(_WIN32)
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
#endif

  constexpr int64_t batch_size = 1;
  constexpr int64_t num_heads = 2;
  constexpr int64_t seq_len = 4;
  constexpr int64_t head_size = 8;
  constexpr int64_t rotary_dim = head_size;

  auto input_f32 = GetFloatDataInRange(-1.0f, 1.0f, batch_size * num_heads * seq_len * head_size);
  auto cos_f32 = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));
  auto sin_f32 = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));

  TestInputDef<Ort::Float16_t> input_def_fp16 = ConvertToFP16InputDef(
      TestInputDef<float>({batch_size, num_heads, seq_len, head_size}, false, input_f32));
  TestInputDef<Ort::Float16_t> cos_def_fp16 = ConvertToFP16InputDef(
      TestInputDef<float>({seq_len, rotary_dim / 2}, false, cos_f32));
  TestInputDef<Ort::Float16_t> sin_def_fp16 = ConvertToFP16InputDef(
      TestInputDef<float>({seq_len, rotary_dim / 2}, false, sin_f32));

  RunRopeOpTest<Ort::Float16_t>(
      input_def_fp16,
      TestInputDef<int64_t>({batch_size, seq_len}, false, std::vector<int64_t>{0, 1, 2, 3}),
      cos_def_fp16,
      sin_def_fp16,
      {test::MakeAttribute("interleaved", int64_t{1}),
       test::MakeAttribute("num_heads", num_heads),
       test::MakeAttribute("rotary_embedding_dim", rotary_dim)},
      /*opset_version*/ 23,
      ExpectedEPNodeAssignment::All,
      kDefaultRopeOpToleranceFp16);
}

// Partial rotation: rotary_dim < head_size. Only first rotary_dim elements are rotated.
TEST_F(QnnHTPBackendTests, RotaryEmbedding_PartialRotation) {
#if defined(_WIN32)
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
#endif

  constexpr int64_t batch_size = 1;
  constexpr int64_t num_heads = 2;
  constexpr int64_t seq_len = 4;
  constexpr int64_t head_size = 8;
  constexpr int64_t rotary_dim = 4;  // Only rotate first 4 of 8 dims

  auto input_f32 = GetFloatDataInRange(-1.0f, 1.0f, batch_size * num_heads * seq_len * head_size);
  auto cos_f32 = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));
  auto sin_f32 = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));

  TestInputDef<Ort::Float16_t> input_def_fp16 = ConvertToFP16InputDef(
      TestInputDef<float>({batch_size, num_heads, seq_len, head_size}, false, input_f32));
  TestInputDef<Ort::Float16_t> cos_def_fp16 = ConvertToFP16InputDef(
      TestInputDef<float>({seq_len, rotary_dim / 2}, false, cos_f32));
  TestInputDef<Ort::Float16_t> sin_def_fp16 = ConvertToFP16InputDef(
      TestInputDef<float>({seq_len, rotary_dim / 2}, false, sin_f32));

  RunRopeOpTest<Ort::Float16_t>(
      input_def_fp16,
      TestInputDef<int64_t>({batch_size, seq_len}, false, std::vector<int64_t>{0, 1, 2, 3}),
      cos_def_fp16,
      sin_def_fp16,
      {test::MakeAttribute("interleaved", int64_t{0}),
       test::MakeAttribute("num_heads", num_heads),
       test::MakeAttribute("rotary_embedding_dim", rotary_dim)},
      /*opset_version*/ 23,
      ExpectedEPNodeAssignment::All,
      kDefaultRopeOpToleranceFp16);
}

// Non-sequential position_ids with larger max_pos cache — tests random access into cos/sin cache.
TEST_F(QnnHTPBackendTests, RotaryEmbedding_RandomPositionIds) {
#if defined(_WIN32)
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
#endif

  constexpr int64_t batch_size = 1;
  constexpr int64_t num_heads = 2;
  constexpr int64_t seq_len = 4;
  constexpr int64_t head_size = 8;
  constexpr int64_t rotary_dim = head_size;
  constexpr int64_t max_pos = 16;  // Cache sized for 16 positions, only 4 used

  auto input_f32 = GetFloatDataInRange(-1.0f, 1.0f, batch_size * num_heads * seq_len * head_size);
  // Cache is [max_pos, rotary_dim/2] — larger than seq_len
  auto cos_f32 = GetFloatDataInRange(-1.0f, 1.0f, max_pos * (rotary_dim / 2));
  auto sin_f32 = GetFloatDataInRange(-1.0f, 1.0f, max_pos * (rotary_dim / 2));

  TestInputDef<Ort::Float16_t> input_def_fp16 = ConvertToFP16InputDef(
      TestInputDef<float>({batch_size, num_heads, seq_len, head_size}, false, input_f32));
  TestInputDef<Ort::Float16_t> cos_def_fp16 = ConvertToFP16InputDef(
      TestInputDef<float>({max_pos, rotary_dim / 2}, false, cos_f32));
  TestInputDef<Ort::Float16_t> sin_def_fp16 = ConvertToFP16InputDef(
      TestInputDef<float>({max_pos, rotary_dim / 2}, false, sin_f32));

  // Non-sequential position indices — random access into the cache
  RunRopeOpTest<Ort::Float16_t>(
      input_def_fp16,
      TestInputDef<int64_t>({batch_size, seq_len}, false, std::vector<int64_t>{7, 2, 14, 0}),
      cos_def_fp16,
      sin_def_fp16,
      {test::MakeAttribute("interleaved", int64_t{0}),
       test::MakeAttribute("num_heads", num_heads),
       test::MakeAttribute("rotary_embedding_dim", rotary_dim)},
      /*opset_version*/ 23,
      ExpectedEPNodeAssignment::All,
      kDefaultRopeOpToleranceFp16);
}

// FP32 variant — validates FP32 data type support advertised by IsOpSupported.
TEST_F(QnnHTPBackendTests, RotaryEmbedding_FP32) {
#if defined(_WIN32)
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
#endif

  constexpr int64_t batch_size = 1;
  constexpr int64_t num_heads = 2;
  constexpr int64_t seq_len = 4;
  constexpr int64_t head_size = 8;
  constexpr int64_t rotary_dim = head_size;

  auto input_f32 = GetFloatDataInRange(-1.0f, 1.0f, batch_size * num_heads * seq_len * head_size);
  auto cos_f32 = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));
  auto sin_f32 = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));

  RunRopeOpTest<float>(
      TestInputDef<float>({batch_size, num_heads, seq_len, head_size}, false, input_f32),
      TestInputDef<int64_t>({batch_size, seq_len}, false, std::vector<int64_t>{0, 1, 2, 3}),
      TestInputDef<float>({seq_len, rotary_dim / 2}, false, cos_f32),
      TestInputDef<float>({seq_len, rotary_dim / 2}, false, sin_f32),
      {test::MakeAttribute("interleaved", int64_t{0}),
       test::MakeAttribute("num_heads", num_heads),
       test::MakeAttribute("rotary_embedding_dim", rotary_dim)},
      /*opset_version*/ 23,
      ExpectedEPNodeAssignment::All,
      1e-3f);
}

// No rotary_embedding_dim attribute — exercises the 0→head_size resolution path.
TEST_F(QnnHTPBackendTests, RotaryEmbedding_DefaultRotaryDim) {
#if defined(_WIN32)
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
#endif

  constexpr int64_t batch_size = 1;
  constexpr int64_t num_heads = 2;
  constexpr int64_t seq_len = 4;
  constexpr int64_t head_size = 8;
  constexpr int64_t rotary_dim = head_size;  // full rotation

  auto input_f32 = GetFloatDataInRange(-1.0f, 1.0f, batch_size * num_heads * seq_len * head_size);
  auto cos_f32 = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));
  auto sin_f32 = GetFloatDataInRange(-1.0f, 1.0f, seq_len * (rotary_dim / 2));

  TestInputDef<Ort::Float16_t> input_def_fp16 = ConvertToFP16InputDef(
      TestInputDef<float>({batch_size, num_heads, seq_len, head_size}, false, input_f32));
  TestInputDef<Ort::Float16_t> cos_def_fp16 = ConvertToFP16InputDef(
      TestInputDef<float>({seq_len, rotary_dim / 2}, false, cos_f32));
  TestInputDef<Ort::Float16_t> sin_def_fp16 = ConvertToFP16InputDef(
      TestInputDef<float>({seq_len, rotary_dim / 2}, false, sin_f32));

  // Note: no rotary_embedding_dim attribute — defaults to 0, resolved to head_size
  RunRopeOpTest<Ort::Float16_t>(
      input_def_fp16,
      TestInputDef<int64_t>({batch_size, seq_len}, false, std::vector<int64_t>{0, 1, 2, 3}),
      cos_def_fp16,
      sin_def_fp16,
      {test::MakeAttribute("interleaved", int64_t{0}),
       test::MakeAttribute("num_heads", num_heads)},
      /*opset_version*/ 23,
      ExpectedEPNodeAssignment::All,
      kDefaultRopeOpToleranceFp16);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
