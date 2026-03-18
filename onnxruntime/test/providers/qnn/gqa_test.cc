// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <optional>

#include "test/providers/qnn/qnn_test_utils.h"
#include "core/graph/node_attr_utils.h"

#include "core/graph/onnx_protobuf.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

template <typename T, typename M>
static GetTestModelFn BuildGQATestCase(
    // Op Inputs
    const TestInputDef<T>& query_def,
    const std::optional<std::reference_wrapper<TestInputDef<T>>> key_def,
    const std::optional<std::reference_wrapper<TestInputDef<T>>> value_def,
    const std::optional<std::reference_wrapper<TestInputDef<T>>> past_key_def,
    const std::optional<std::reference_wrapper<TestInputDef<T>>> past_value_def,
    const TestInputDef<M>& seqlens_k_def,
    const TestInputDef<M>& total_sequence_length_def,
    const std::optional<std::reference_wrapper<TestInputDef<T>>> cos_cache_def,
    const std::optional<std::reference_wrapper<TestInputDef<T>>> sin_cache_def,
    const std::optional<std::reference_wrapper<TestInputDef<int64_t>>> position_ids_def,
    const std::optional<std::reference_wrapper<TestInputDef<T>>> attention_bias_def,
    const std::optional<std::reference_wrapper<TestInputDef<T>>> head_sink_def,
    // Op Attributes
    const std::optional<int32_t> do_rotary,
    const std::optional<std::string> k_quant_type,
    const std::optional<int32_t> kv_cache_bit_width,
    const int32_t kv_num_heads,
    const std::optional<int32_t> local_window_size,
    const int32_t num_heads,
    const std::optional<int32_t> qk_output,
    const std::optional<int32_t> rotary_interleaved,
    const std::optional<float> scale,
    const std::optional<int32_t> smooth_softmax,
    const std::optional<std::string> v_quant_type) {
  return [query_def, key_def, value_def, past_key_def, past_value_def, seqlens_k_def, total_sequence_length_def,
          cos_cache_def, sin_cache_def, position_ids_def, attention_bias_def, head_sink_def,
          do_rotary, k_quant_type, kv_cache_bit_width, kv_num_heads, local_window_size, num_heads, qk_output,
          rotary_interleaved, scale, smooth_softmax, v_quant_type](ModelTestBuilder& builder) {
    // helpers to make inputs
    auto add_input_T = [&](const char* name, const TestInputDef<T>& def) -> std::string {
      MakeTestInput(builder, name, def);
      return name;
    };
    auto add_input_M = [&](const char* name, const TestInputDef<M>& def) -> std::string {
      MakeTestInput(builder, name, def);
      return name;
    };
    auto add_input_I64 = [&](const char* name, const TestInputDef<int64_t>& def) -> std::string {
      MakeTestInput(builder, name, def);
      return name;
    };

    std::vector<std::string> input_names;

    input_names.push_back(add_input_T("query", query_def));
    input_names.push_back(key_def ? add_input_T("key", key_def->get()) : "");
    input_names.push_back(value_def ? add_input_T("value", value_def->get()) : "");
    input_names.push_back(past_key_def ? add_input_T("past_key", past_key_def->get()) : "");
    input_names.push_back(past_value_def ? add_input_T("past_value", past_value_def->get()) : "");
    input_names.push_back(add_input_M("seqlens_k", seqlens_k_def));
    input_names.push_back(add_input_M("total_sequence_length", total_sequence_length_def));
    input_names.push_back(cos_cache_def ? add_input_T("cos_cache", cos_cache_def->get()) : "");
    input_names.push_back(sin_cache_def ? add_input_T("sin_cache", sin_cache_def->get()) : "");
    input_names.push_back(position_ids_def ? add_input_I64("position_ids", position_ids_def->get()) : "");
    input_names.push_back(attention_bias_def ? add_input_T("attention_bias", attention_bias_def->get()) : "");
    input_names.push_back(head_sink_def ? add_input_T("head_sink", head_sink_def->get()) : "");

    std::vector<std::string> output_names;

    builder.MakeOutput("output");
    output_names.push_back("output");

    builder.MakeOutput("present_key");
    output_names.push_back("present_key");

    builder.MakeOutput("present_value");
    output_names.push_back("present_value");

    if (qk_output.has_value() && qk_output.value() != 0) {
      builder.MakeOutput("output_qk");
      output_names.push_back("output_qk");
    }

    std::vector<ONNX_NAMESPACE::AttributeProto> attrs;

    attrs.push_back(builder.MakeScalarAttribute("num_heads", static_cast<int64_t>(num_heads)));
    attrs.push_back(builder.MakeScalarAttribute("kv_num_heads", static_cast<int64_t>(kv_num_heads)));

    if (do_rotary.has_value())
      attrs.push_back(builder.MakeScalarAttribute("do_rotary", static_cast<int64_t>(do_rotary.value())));
    if (local_window_size.has_value())
      attrs.push_back(builder.MakeScalarAttribute("local_window_size", static_cast<int64_t>(local_window_size.value())));
    if (rotary_interleaved.has_value())
      attrs.push_back(builder.MakeScalarAttribute("rotary_interleaved", static_cast<int64_t>(rotary_interleaved.value())));
    if (scale.has_value())
      attrs.push_back(builder.MakeScalarAttribute("scale", scale.value()));
    if (smooth_softmax.has_value())
      attrs.push_back(builder.MakeScalarAttribute("smooth_softmax", static_cast<int64_t>(smooth_softmax.value())));
    if (qk_output.has_value())
      attrs.push_back(builder.MakeScalarAttribute("qk_output", static_cast<int64_t>(qk_output.value())));
    if (kv_cache_bit_width.has_value())
      attrs.push_back(builder.MakeScalarAttribute("kv_cache_bit_width", static_cast<int64_t>(kv_cache_bit_width.value())));
    if (k_quant_type.has_value())
      attrs.push_back(builder.MakeStringAttribute("k_quant_type", k_quant_type.value()));
    if (v_quant_type.has_value())
      attrs.push_back(builder.MakeStringAttribute("v_quant_type", v_quant_type.value()));

    builder.AddNode("GQA",
                    "GroupQueryAttention",
                    input_names,
                    output_names,
                    kMSDomain,
                    attrs);
  };
}

// Runs a model with a GQA operator through QNN EP. Checks the graph node assignment
// and that inference outputs for QNN EP and CPU EP match.
template <typename T, typename M>
static void RunGQATest(
    // Op Inputs
    const TestInputDef<T>& query_def,
    const std::optional<std::reference_wrapper<TestInputDef<T>>> key_def,
    const std::optional<std::reference_wrapper<TestInputDef<T>>> value_def,
    const std::optional<std::reference_wrapper<TestInputDef<T>>> past_key_def,
    const std::optional<std::reference_wrapper<TestInputDef<T>>> past_value_def,
    const TestInputDef<M>& seqlens_k_def,
    const TestInputDef<M>& total_sequence_length_def,
    const std::optional<std::reference_wrapper<TestInputDef<T>>> cos_cache_def,
    const std::optional<std::reference_wrapper<TestInputDef<T>>> sin_cache_def,
    const std::optional<std::reference_wrapper<TestInputDef<int64_t>>> position_ids_def,
    const std::optional<std::reference_wrapper<TestInputDef<T>>> attention_bias_def,
    const std::optional<std::reference_wrapper<TestInputDef<T>>> head_sink_def,
    // Op Attributes
    const std::optional<int32_t> do_rotary,
    const std::optional<std::string> k_quant_type,
    const std::optional<int32_t> kv_cache_bit_width,
    const int32_t kv_num_heads,
    const std::optional<int32_t> local_window_size,
    const int32_t num_heads,
    const std::optional<int32_t> qk_output,
    const std::optional<int32_t> rotary_interleaved,
    const std::optional<float> scale,
    const std::optional<int32_t> smooth_softmax,
    const std::optional<std::string> v_quant_type,
    // Test options
    ExpectedEPNodeAssignment expected_ep_assignment,
    const std::string& backend_name,
    int opset = 13,
    float tolerance = 1e-5f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_name;

  RunQnnModelTest(BuildGQATestCase<T, M>(query_def, key_def, value_def,
                                         past_key_def, past_value_def,
                                         seqlens_k_def, total_sequence_length_def,
                                         cos_cache_def, sin_cache_def,
                                         position_ids_def, attention_bias_def, head_sink_def,
                                         do_rotary, k_quant_type, kv_cache_bit_width, kv_num_heads,
                                         local_window_size, num_heads, qk_output, rotary_interleaved,
                                         scale, smooth_softmax, v_quant_type),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  tolerance);
}

#if defined(_M_ARM64)
//
// GPU tests:
//

TEST_F(QnnGPUBackendTests, GroupQueryAttention_Basic_FP32) {
  // Test parameters
  const int32_t batch_size = 1;
  const int32_t sequence_length = 1;
  const int32_t total_seq_len = 1024;
  const int32_t num_heads = 8;
  const int32_t kv_num_heads = 4;
  const int32_t head_size = 32;

  const float scale = 10.0f;

  // Derived sizes
  const int32_t packed_qkv_d = num_heads * head_size + 2 * kv_num_heads * head_size;

  // === Inputs ===
  auto query_def = TestInputDef<float>({batch_size, sequence_length, packed_qkv_d},
                                       false, -1.0f, 1.0f);
  const std::optional<std::reference_wrapper<TestInputDef<float>>> key_def = std::nullopt;
  const std::optional<std::reference_wrapper<TestInputDef<float>>> value_def = std::nullopt;

  static auto pk_max = TestInputDef<float>({batch_size, kv_num_heads, total_seq_len, head_size},
                                           false, -1.0f, 1.0f);
  static auto pv_max = TestInputDef<float>({batch_size, kv_num_heads, total_seq_len, head_size},
                                           false, -1.0f, 1.0f);
  std::optional<std::reference_wrapper<TestInputDef<float>>> past_key_def = std::ref(pk_max);
  std::optional<std::reference_wrapper<TestInputDef<float>>> past_value_def = std::ref(pv_max);

  std::vector<int32_t> seqlens_k_data(batch_size, total_seq_len - 1);
  auto seqlens_k_def = TestInputDef<int32_t>({batch_size}, true, seqlens_k_data);

  auto total_sequence_length_def = TestInputDef<int32_t>({}, true,
                                                         std::vector<int32_t>{total_seq_len});

  static auto cos_def = TestInputDef<float>({total_seq_len, head_size / 2},
                                            true, -1.0f, 1.0f);
  static auto sin_def = TestInputDef<float>({total_seq_len, head_size / 2},
                                            true, -1.0f, 1.0f);

  std::optional<std::reference_wrapper<TestInputDef<float>>> cos_cache_def = std::ref(cos_def);
  std::optional<std::reference_wrapper<TestInputDef<float>>> sin_cache_def = std::ref(sin_def);

  std::vector<int64_t> position_ids_data(batch_size * sequence_length);
  for (int32_t i = 0; i < batch_size * sequence_length; i++) {
    position_ids_data[i] = (i + 2) % total_seq_len;
  }
  static TestInputDef<int64_t> pos_def({batch_size, sequence_length}, true, position_ids_data);
  std::optional<std::reference_wrapper<TestInputDef<int64_t>>> position_ids_def = std::ref(pos_def);

  std::optional<std::reference_wrapper<TestInputDef<float>>> attention_bias_def = std::nullopt;
  std::optional<std::reference_wrapper<TestInputDef<float>>> head_sink_def = std::nullopt;

  // === Attributes ===
  const std::optional<int32_t> do_rotary_attr = 1;
  const std::optional<std::string> k_quant_type = std::nullopt;
  const std::optional<int32_t> kv_cache_bit_width = std::nullopt;
  const std::optional<int32_t> local_window_size_attr = std::nullopt;
  const std::optional<int32_t> qk_output_attr = std::nullopt;
  const std::optional<int32_t> rotary_interleaved_attr = std::nullopt;
  const std::optional<float> scale_attr = scale;
  const std::optional<int32_t> smooth_softmax_attr = std::nullopt;
  const std::optional<std::string> v_quant_type = std::nullopt;

  // === Run ===
  RunGQATest(
      query_def,
      key_def,
      value_def,
      past_key_def,
      past_value_def,
      seqlens_k_def,
      total_sequence_length_def,
      cos_cache_def,
      sin_cache_def,
      position_ids_def,
      attention_bias_def,
      head_sink_def,
      do_rotary_attr,
      k_quant_type,
      kv_cache_bit_width,
      kv_num_heads,
      local_window_size_attr,
      num_heads,
      qk_output_attr,
      rotary_interleaved_attr,
      scale_attr,
      smooth_softmax_attr,
      v_quant_type,
      ExpectedEPNodeAssignment::All,
      "gpu",
      13,
      1e-5f);
}

#endif  // defined(_M_ARM64) GPU tests

}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)

