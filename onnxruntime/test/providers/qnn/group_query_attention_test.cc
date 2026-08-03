// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/util/include/api_asserts.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

template <typename T>
static GetTestModelFn BuildGQATestCase(
    // Op Inputs
    const TestInputDef<T>& query_def,
    const std::optional<std::reference_wrapper<TestInputDef<T>>> key_def,
    const std::optional<std::reference_wrapper<TestInputDef<T>>> value_def,
    const std::optional<std::reference_wrapper<TestInputDef<T>>> past_key_def,
    const std::optional<std::reference_wrapper<TestInputDef<T>>> past_value_def,
    const TestInputDef<int32_t>& seqlens_k_def,
    const TestInputDef<int32_t>& total_sequence_length_def,
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
    // helper to make an input: creates the graph input and returns its name.
    auto add_input = [&](const char* name, const auto& def) -> std::string {
      MakeTestInput(builder, name, def);
      return name;
    };

    std::vector<std::string> input_names;

    input_names.push_back(add_input("query", query_def));
    input_names.push_back(key_def ? add_input("key", key_def->get()) : "");
    input_names.push_back(value_def ? add_input("value", value_def->get()) : "");
    input_names.push_back(past_key_def ? add_input("past_key", past_key_def->get()) : "");
    input_names.push_back(past_value_def ? add_input("past_value", past_value_def->get()) : "");
    input_names.push_back(add_input("seqlens_k", seqlens_k_def));
    input_names.push_back(add_input("total_sequence_length", total_sequence_length_def));
    input_names.push_back(cos_cache_def ? add_input("cos_cache", cos_cache_def->get()) : "");
    input_names.push_back(sin_cache_def ? add_input("sin_cache", sin_cache_def->get()) : "");
    input_names.push_back(position_ids_def ? add_input("position_ids", position_ids_def->get()) : "");
    input_names.push_back(attention_bias_def ? add_input("attention_bias", attention_bias_def->get()) : "");
    input_names.push_back(head_sink_def ? add_input("head_sink", head_sink_def->get()) : "");

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
template <typename T>
static void RunGQATest(
    // Op Inputs
    const TestInputDef<T>& query_def,
    const std::optional<std::reference_wrapper<TestInputDef<T>>> key_def,
    const std::optional<std::reference_wrapper<TestInputDef<T>>> value_def,
    const std::optional<std::reference_wrapper<TestInputDef<T>>> past_key_def,
    const std::optional<std::reference_wrapper<TestInputDef<T>>> past_value_def,
    const TestInputDef<int32_t>& seqlens_k_def,
    const TestInputDef<int32_t>& total_sequence_length_def,
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
    float fp32_abs_err = 1e-5f,
    bool use_shared_memory_allocator = false) {
  const GetTestModelFn build_test_case = BuildGQATestCase<T>(query_def, key_def, value_def,
                                                             past_key_def, past_value_def,
                                                             seqlens_k_def, total_sequence_length_def,
                                                             cos_cache_def, sin_cache_def,
                                                             position_ids_def, attention_bias_def, head_sink_def,
                                                             do_rotary, k_quant_type, kv_cache_bit_width, kv_num_heads,
                                                             local_window_size, num_heads, qk_output, rotary_interleaved,
                                                             scale, smooth_softmax, v_quant_type);
  // The GPU backend only supports GQA with past/present buffer sharing on gpu-accessible shared memory. So, the GQA UTs do not
  // use RunQnnModelTest and instead manually create/run the QNN inference session with the buffer sharing.
  ModelTestBuilder helper;
  build_test_case(helper);

  const std::unordered_map<std::string, int> domain_to_version = {{"", opset}, {kMSDomain, 1}};
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  std::string model_data;
  helper.model_.SerializeToString(&model_data);

  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_name;
  if (use_shared_memory_allocator) {
    // GPU and HTP use different shared-memory allocators. Both expose the same "QnnHtpShared"
    // host-accessible MemoryInfo, but they are enabled via different provider options.
    if (backend_name == "gpu") {
      provider_options["enable_dx12_shared_memory_allocator"] = "1";
    } else if (backend_name == "htp") {
      provider_options["enable_htp_shared_memory_allocator"] = "1";
    }
  }

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions qnn_so;
  qnn_so.AddConfigEntry(kOrtSessionOptionsRecordEpGraphAssignmentInfo, "1");
  RegisterQnnEpLibrary(registered_ep_device, qnn_so, kQnnExecutionProvider, provider_options);
  ScopedOrtSession scoped_qnn_session(
      std::move(registered_ep_device),
      Ort::Session(*GetOrtEnv(), model_data.data(), static_cast<int>(model_data.size()), qnn_so));
  Ort::Session& qnn_session = scoped_qnn_session.session();
  ASSERT_NO_FATAL_FAILURE(VerifyEPNodeAssignment(qnn_session, kQnnExecutionProvider, expected_ep_assignment));

  Ort::SessionOptions cpu_so;
  Ort::Session cpu_session(*GetOrtEnv(), model_data.data(), static_cast<int>(model_data.size()), cpu_so);

  std::vector<std::string> input_names = qnn_session.GetInputNames();
  std::vector<std::string> output_names = qnn_session.GetOutputNames();
  std::vector<const char*> input_names_cstr;
  std::vector<const char*> output_names_cstr;
  input_names_cstr.reserve(input_names.size());
  output_names_cstr.reserve(output_names.size());
  for (const auto& input_name : input_names) {
    input_names_cstr.push_back(input_name.c_str());
  }
  for (const auto& output_name : output_names) {
    output_names_cstr.push_back(output_name.c_str());
  }

  struct FeedCopy {
    Ort::MemoryAllocation allocation;
    Ort::Value value{nullptr};
  };

  Ort::MemoryInfo memory_info(nullptr);
  Ort::Allocator allocator(nullptr);
  if (use_shared_memory_allocator && (backend_name == "gpu" || backend_name == "htp")) {
    // Both GPU and HTP share past/present in-place on QNN host-accessible shared memory, exposed
    // via the "QnnHtpShared" allocator on the QNN session. HTP requires RPCMEM (libcdsprpc), which
    // is only available on-device, so this may be unavailable in host/emulator environments.
    try {
      memory_info = Ort::MemoryInfo("QnnHtpShared", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeCPU);
      allocator = Ort::Allocator(qnn_session, memory_info);
    } catch (const Ort::Exception&) {
      GTEST_SKIP() << "QNN shared memory allocator unavailable (driver / RPCMEM). Skipping test.";
    }
  } else {
    memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    allocator = Ort::Allocator(cpu_session, memory_info);
  }

  std::vector<FeedCopy> qnn_feeds;
  qnn_feeds.reserve(input_names.size());
  std::unordered_map<std::string, size_t> input_name_to_index;

  for (const auto& input_name : input_names) {
    const Ort::Value& source_value = helper.feeds_.at(input_name);
    const auto tensor_info = source_value.GetTensorTypeAndShapeInfo();
    const auto shape = tensor_info.GetShape();
    const size_t num_bytes = source_value.GetTensorSizeInBytes();
    const auto* source_data = reinterpret_cast<const std::byte*>(source_value.GetTensorRawData());

    FeedCopy feed_copy{allocator.GetAllocation(num_bytes)};
    ASSERT_NE(feed_copy.allocation.get(), nullptr);
    memcpy(feed_copy.allocation.get(), source_data, num_bytes);

    feed_copy.value = Ort::Value::CreateTensor(memory_info,
                                               feed_copy.allocation.get(),
                                               feed_copy.allocation.size(),
                                               shape.data(),
                                               shape.size(),
                                               tensor_info.GetElementType());

    input_name_to_index.emplace(input_name, qnn_feeds.size());
    qnn_feeds.push_back(std::move(feed_copy));
  }

  std::vector<const OrtValue*> qnn_input_values;
  qnn_input_values.reserve(qnn_feeds.size());
  for (const auto& qnn_feed : qnn_feeds) {
    qnn_input_values.push_back(qnn_feed.value);
  }

  std::vector<OrtValue*> qnn_output_values(output_names.size(), nullptr);
  const auto past_key_input = input_name_to_index.find("past_key");
  const auto past_value_input = input_name_to_index.find("past_value");
  for (size_t i = 0; i < output_names.size(); i++) {
    // Make present_key and present_value use the same buffer as past_key and past_value.
    if (output_names[i] == "present_key" && past_key_input != input_name_to_index.end()) {
      qnn_output_values[i] = qnn_feeds[past_key_input->second].value;
    } else if (output_names[i] == "present_value" && past_value_input != input_name_to_index.end()) {
      qnn_output_values[i] = qnn_feeds[past_value_input->second].value;
    }
  }

  Ort::RunOptions qnn_run_options;
  ASSERT_ORTSTATUS_OK(Ort::GetApi().Run(qnn_session,
                                        qnn_run_options,
                                        input_names_cstr.data(),
                                        qnn_input_values.data(),
                                        qnn_input_values.size(),
                                        output_names_cstr.data(),
                                        output_names_cstr.size(),
                                        qnn_output_values.data()));

  std::vector<Ort::Value> owned_qnn_outputs;
  owned_qnn_outputs.reserve(output_names.size());
  std::vector<const Ort::Value*> qnn_outputs;
  qnn_outputs.reserve(output_names.size());
  for (size_t i = 0; i < output_names.size(); i++) {
    if (output_names[i] == "present_key" && past_key_input != input_name_to_index.end()) {
      ASSERT_EQ(qnn_output_values[i], static_cast<OrtValue*>(qnn_feeds[past_key_input->second].value));
      qnn_outputs.push_back(&qnn_feeds[past_key_input->second].value);
    } else if (output_names[i] == "present_value" && past_value_input != input_name_to_index.end()) {
      ASSERT_EQ(qnn_output_values[i], static_cast<OrtValue*>(qnn_feeds[past_value_input->second].value));
      qnn_outputs.push_back(&qnn_feeds[past_value_input->second].value);
    } else {
      ASSERT_NE(qnn_output_values[i], nullptr);
      owned_qnn_outputs.emplace_back(qnn_output_values[i]);
      qnn_outputs.push_back(&owned_qnn_outputs.back());
    }
  }

  Ort::RunOptions cpu_run_options;
  std::vector<Ort::Value> cpu_outputs;
  // The CPU EP can do GQA without buffer sharing, so we can just use RunWithEP
  RunWithEP(cpu_session, cpu_run_options, helper.feeds_, cpu_outputs);

  // Check QNN outputs against CPU
  ASSERT_EQ(cpu_outputs.size(), output_names.size());
  ASSERT_EQ(qnn_outputs.size(), output_names.size());
  for (size_t i = 0; i < output_names.size(); i++) {
    VerifyOutput(output_names[i], cpu_outputs[i], *qnn_outputs[i], ElementwiseAbsoluteVerifier{fp32_abs_err});
  }
}

// BuildGQATestCase and RunGQATest above are backend-agnostic and stay unguarded so they can be
// shared by GPU tests (and reused after rebasing onto the GPU GQA PR). The HTP-specific driver and
// the QnnHTPBackendTests cases below are guarded: the HTP backend exists on ARM64 (Windows on
// Snapdragon / device) and on x86 Linux (HTP emulation), but not on x86 Windows hosts, which must
// not try to instantiate the QnnHTPBackendTests fixture.
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//
// These run on arm64 (Windows on Snapdragon / Linux arm64) with a QAIRT SDK >= 2.48 (QNN opset
// 2.12), which is the version that compiles the GQA op builder in (see the
// QNN_OPSET_VERSION_MAJOR/MINOR guard at the top of
// core/providers/qnn/builder/opbuilder/group_query_attention_op_builder.cc). The inference tests
// (RunHTPPackedGQATest / RunHTPUnpackedGQATest) execute GQA on device and compare against the CPU
// EP; they need QNN host-accessible shared memory (RPCMEM), so on hosts without it (e.g. x86 Linux
// HTP emulation) they GTEST_SKIP at the allocator step rather than fail. The op_affinity
// EP-assignment tests do not run inference and run anywhere the builder is compiled in.
//
// The one still-DISABLED_ case is GroupQueryAttention_PaddedCache_NoisePadding_FP32: it is
// designed to fail as written (see its EXPECTED-TO-FAIL note below) and must have its comparator
// fixed before it can be enabled.
//

// Compact driver for HTP GQA tests over a packed-QKV model with a full-capacity past KV cache.
// Only the knobs that matter for edge-case coverage are exposed; everything else mirrors the
// common LLM decode/prefill setup. The harness aliases present->past (in-place KV cache), so the
// present_* outputs use the max_sequence_length (== total_seq_len) BNSH layout that matches ONNX.
//
// scale is passed through verbatim: scale == 0.0f exercises the "use default 1/sqrt(head_size)"
// sentinel path that both the ORT CPU EP and the QNN EP must honor.
template <typename T>
static void RunHTPPackedGQATest(int32_t num_heads,
                                int32_t kv_num_heads,
                                int32_t head_size,
                                int32_t sequence_length,
                                int32_t total_seq_len,
                                float scale,
                                int32_t do_rotary,
                                float fp32_abs_err = 1e-2f,
                                int32_t max_seq_len = 0) {
  const int32_t batch_size = 1;
  const int32_t packed_qkv_d = num_heads * head_size + 2 * kv_num_heads * head_size;
  if (max_seq_len <= 0) {
    max_seq_len = total_seq_len;  // default: cache capacity == valid length (no padding region)
  }

  auto query_def = TestInputDef<T>({batch_size, sequence_length, packed_qkv_d},
                                   false, static_cast<T>(-1.0f), static_cast<T>(1.0f));
  const std::optional<std::reference_wrapper<TestInputDef<T>>> key_def = std::nullopt;
  const std::optional<std::reference_wrapper<TestInputDef<T>>> value_def = std::nullopt;

  // Past KV buffer has capacity max_seq_len, filled with random data (so when max_seq_len >
  // total_seq_len the padding region naturally holds noise values).
  TestInputDef<T> pk_max({batch_size, kv_num_heads, max_seq_len, head_size},
                         false, static_cast<T>(-1.0f), static_cast<T>(1.0f));
  TestInputDef<T> pv_max({batch_size, kv_num_heads, max_seq_len, head_size},
                         false, static_cast<T>(-1.0f), static_cast<T>(1.0f));
  std::optional<std::reference_wrapper<TestInputDef<T>>> past_key_def = std::ref(pk_max);
  std::optional<std::reference_wrapper<TestInputDef<T>>> past_value_def = std::ref(pv_max);

  std::vector<int32_t> seqlens_k_data(batch_size, total_seq_len - 1);
  auto seqlens_k_def = TestInputDef<int32_t>({batch_size}, true, seqlens_k_data);
  auto total_sequence_length_def = TestInputDef<int32_t>({}, true, std::vector<int32_t>{total_seq_len});

  // Rotary caches (only consumed when do_rotary != 0, but cheap to always build). Sized to cache
  // capacity (max_seq_len rows).
  TestInputDef<T> cos_def({max_seq_len, head_size / 2}, true, static_cast<T>(-1.0f), static_cast<T>(1.0f));
  TestInputDef<T> sin_def({max_seq_len, head_size / 2}, true, static_cast<T>(-1.0f), static_cast<T>(1.0f));
  std::optional<std::reference_wrapper<TestInputDef<T>>> cos_cache_def =
      do_rotary != 0 ? std::optional<std::reference_wrapper<TestInputDef<T>>>(std::ref(cos_def)) : std::nullopt;
  std::optional<std::reference_wrapper<TestInputDef<T>>> sin_cache_def =
      do_rotary != 0 ? std::optional<std::reference_wrapper<TestInputDef<T>>>(std::ref(sin_def)) : std::nullopt;

  std::optional<std::reference_wrapper<TestInputDef<int64_t>>> position_ids_def = std::nullopt;
  std::optional<std::reference_wrapper<TestInputDef<T>>> attention_bias_def = std::nullopt;
  std::optional<std::reference_wrapper<TestInputDef<T>>> head_sink_def = std::nullopt;

  RunGQATest<T>(
      query_def, key_def, value_def, past_key_def, past_value_def,
      seqlens_k_def, total_sequence_length_def,
      cos_cache_def, sin_cache_def, position_ids_def, attention_bias_def, head_sink_def,
      /*do_rotary*/ do_rotary,
      /*k_quant_type*/ std::nullopt,
      /*kv_cache_bit_width*/ std::nullopt,
      kv_num_heads,
      /*local_window_size*/ std::nullopt,
      num_heads,
      /*qk_output*/ std::nullopt,
      /*rotary_interleaved*/ std::nullopt,
      /*scale*/ scale,
      /*smooth_softmax*/ std::nullopt,
      /*v_quant_type*/ std::nullopt,
      ExpectedEPNodeAssignment::All,
      "htp",
      /*opset*/ 13,
      fp32_abs_err,
      /*use_shared_memory_allocator*/ true);
}

// Compact driver for HTP GQA tests over an unpacked (separate Q / K / V) model with a full-capacity
// past KV cache. Mirrors RunHTPPackedGQATest but feeds query, key and value as three separate
// inputs instead of a single packed-QKV tensor. With separate inputs the last dim of query is
// num_heads * head_size and the last dim of key / value is kv_num_heads * head_size (BSD layout).
// Everything else (past KV aliasing, rotary caches, scale==0 default) matches the packed driver.
template <typename T>
static void RunHTPUnpackedGQATest(int32_t num_heads,
                                  int32_t kv_num_heads,
                                  int32_t head_size,
                                  int32_t sequence_length,
                                  int32_t total_seq_len,
                                  float scale,
                                  int32_t do_rotary,
                                  float fp32_abs_err = 1e-2f) {
  const int32_t batch_size = 1;
  const int32_t q_hidden = num_heads * head_size;
  const int32_t kv_hidden = kv_num_heads * head_size;

  auto query_def = TestInputDef<T>({batch_size, sequence_length, q_hidden},
                                   false, static_cast<T>(-1.0f), static_cast<T>(1.0f));

  TestInputDef<T> k_cur({batch_size, sequence_length, kv_hidden},
                        false, static_cast<T>(-1.0f), static_cast<T>(1.0f));
  TestInputDef<T> v_cur({batch_size, sequence_length, kv_hidden},
                        false, static_cast<T>(-1.0f), static_cast<T>(1.0f));
  std::optional<std::reference_wrapper<TestInputDef<T>>> key_def = std::ref(k_cur);
  std::optional<std::reference_wrapper<TestInputDef<T>>> value_def = std::ref(v_cur);

  TestInputDef<T> pk_max({batch_size, kv_num_heads, total_seq_len, head_size},
                         false, static_cast<T>(-1.0f), static_cast<T>(1.0f));
  TestInputDef<T> pv_max({batch_size, kv_num_heads, total_seq_len, head_size},
                         false, static_cast<T>(-1.0f), static_cast<T>(1.0f));
  std::optional<std::reference_wrapper<TestInputDef<T>>> past_key_def = std::ref(pk_max);
  std::optional<std::reference_wrapper<TestInputDef<T>>> past_value_def = std::ref(pv_max);

  std::vector<int32_t> seqlens_k_data(batch_size, total_seq_len - 1);
  auto seqlens_k_def = TestInputDef<int32_t>({batch_size}, true, seqlens_k_data);
  auto total_sequence_length_def = TestInputDef<int32_t>({}, true, std::vector<int32_t>{total_seq_len});

  // Rotary caches (only consumed when do_rotary != 0, but cheap to always build).
  TestInputDef<T> cos_def({total_seq_len, head_size / 2}, true, static_cast<T>(-1.0f), static_cast<T>(1.0f));
  TestInputDef<T> sin_def({total_seq_len, head_size / 2}, true, static_cast<T>(-1.0f), static_cast<T>(1.0f));
  std::optional<std::reference_wrapper<TestInputDef<T>>> cos_cache_def =
      do_rotary != 0 ? std::optional<std::reference_wrapper<TestInputDef<T>>>(std::ref(cos_def)) : std::nullopt;
  std::optional<std::reference_wrapper<TestInputDef<T>>> sin_cache_def =
      do_rotary != 0 ? std::optional<std::reference_wrapper<TestInputDef<T>>>(std::ref(sin_def)) : std::nullopt;

  std::optional<std::reference_wrapper<TestInputDef<int64_t>>> position_ids_def = std::nullopt;
  std::optional<std::reference_wrapper<TestInputDef<T>>> attention_bias_def = std::nullopt;
  std::optional<std::reference_wrapper<TestInputDef<T>>> head_sink_def = std::nullopt;

  RunGQATest<T>(
      query_def, key_def, value_def, past_key_def, past_value_def,
      seqlens_k_def, total_sequence_length_def,
      cos_cache_def, sin_cache_def, position_ids_def, attention_bias_def, head_sink_def,
      /*do_rotary*/ do_rotary,
      /*k_quant_type*/ std::nullopt,
      /*kv_cache_bit_width*/ std::nullopt,
      kv_num_heads,
      /*local_window_size*/ std::nullopt,
      num_heads,
      /*qk_output*/ std::nullopt,
      /*rotary_interleaved*/ std::nullopt,
      /*scale*/ scale,
      /*smooth_softmax*/ std::nullopt,
      /*v_quant_type*/ std::nullopt,
      ExpectedEPNodeAssignment::All,
      "htp",
      /*opset*/ 13,
      fp32_abs_err,
      /*use_shared_memory_allocator*/ true);
}

// Basic GQA on the HTP backend (FP32 model, FP16 precision on device).
// Uses scale=0.0f, which both the ORT CPU EP and the QNN EP must interpret as the default
// scale (1/sqrt(head_size)). This guards against the scale==0 sentinel-handling bug.
TEST_F(QnnHTPBackendTests, GroupQueryAttention_Basic_FP32) {
  // num_heads=8, kv_num_heads=4, head_size=32, decode (seq=1), total=1024, scale=0 (default).
  RunHTPPackedGQATest<float>(8, 4, 32, 1, 1024, /*scale*/ 0.0f, /*do_rotary*/ 0);
}

// Explicit non-default scale. head_size=32 -> default would be 1/sqrt(32) ~= 0.1768; using a very
// different value (0.5) ensures the explicit scale actually flows through to the op.
TEST_F(QnnHTPBackendTests, GroupQueryAttention_ScaleExplicit_FP32) {
  RunHTPPackedGQATest<float>(8, 4, 32, 1, 1024, /*scale*/ 0.5f, /*do_rotary*/ 0);
}

// Degenerate grouping: num_heads == kv_num_heads is standard multi-head attention (no grouping).
TEST_F(QnnHTPBackendTests, GroupQueryAttention_MHA_NumHeadsEqKv_FP32) {
  RunHTPPackedGQATest<float>(8, 8, 32, 1, 1024, /*scale*/ 0.0f, /*do_rotary*/ 0);
}

// Extreme grouping: kv_num_heads == 1 is multi-query attention (all query heads share one KV head).
TEST_F(QnnHTPBackendTests, GroupQueryAttention_MQA_KvOne_FP32) {
  RunHTPPackedGQATest<float>(8, 1, 32, 1, 1024, /*scale*/ 0.0f, /*do_rotary*/ 0);
}

// Prefill: sequence_length > 1 (process a whole prompt chunk at once).
TEST_F(QnnHTPBackendTests, GroupQueryAttention_Prefill_FP32) {
  RunHTPPackedGQATest<float>(8, 4, 32, /*seq*/ 64, /*total*/ 1024, /*scale*/ 0.0f, /*do_rotary*/ 0);
}

// Llama-3-like geometry: num_heads=32, kv_num_heads=8, head_size=64.
TEST_F(QnnHTPBackendTests, GroupQueryAttention_Llama3_AR1_FP32) {
  RunHTPPackedGQATest<float>(32, 8, 64, 1, 1024, /*scale*/ 0.0f, /*do_rotary*/ 0);
}

// Rotary embeddings enabled (do_rotary=1) with cos/sin caches.
TEST_F(QnnHTPBackendTests, GroupQueryAttention_Rotary_FP32) {
  RunHTPPackedGQATest<float>(8, 4, 32, 1, 1024, /*scale*/ 0.0f, /*do_rotary*/ 1);
}

// FP16 query/cache path.
TEST_F(QnnHTPBackendTests, GroupQueryAttention_Basic_FP16) {
  RunHTPPackedGQATest<Ort::Float16_t>(8, 4, 32, 1, 1024, /*scale*/ 0.0f, /*do_rotary*/ 0);
}

// === PhiVNext-pattern tests ===
// These mirror the GQA configuration in the customer PhiVNext 7B model: fp16, do_rotary=1, and the
// real head geometry (num_heads=32, kv_num_heads=8, head_size=128). The model splits inference into
// an AR64 prompt-processing (prefill) graph and an AR1 token-generation (decode) graph; both feed
// GQA fp16 query/KV with in-place (buffer-shared) KV cache. scale=0 exercises the default path.

// PhiVNext decode (AR1): sequence_length=1.
TEST_F(QnnHTPBackendTests, GroupQueryAttention_PhiVNext_Decode_FP16) {
  RunHTPPackedGQATest<Ort::Float16_t>(32, 8, 128, /*seq*/ 1, /*total*/ 1024,
                                      /*scale*/ 0.0f, /*do_rotary*/ 1);
}

// PhiVNext prefill (AR64): sequence_length=64.
TEST_F(QnnHTPBackendTests, GroupQueryAttention_PhiVNext_Prefill_FP16) {
  RunHTPPackedGQATest<Ort::Float16_t>(32, 8, 128, /*seq*/ 64, /*total*/ 1024,
                                      /*scale*/ 0.0f, /*do_rotary*/ 1);
}

// Padded KV cache with noise in the padding region: max_seq_len (128) > total_seq_len (16).
// Past KV is random data so the padding region [16:128) has noise values. With buffer sharing,
// QNN present aliases past -> padding = noise. CPU EP golden runs without sharing and memsets
// present padding to 0. Full tensor compare -> expect to fail on the padding region. This
// demonstrates the known padding mismatch between QNN HTP (sharing) and CPU EP (non-sharing).
//
// EXPECTED TO FAIL AS WRITTEN: RunHTPPackedGQATest compares the full present_key/present_value
// tensors, so this case mismatches on the [16:128) padding region by design. This is the one test
// in this file kept DISABLED_ (all other HTP GQA tests are enabled); before enabling it, the
// comparator must be changed to compare only the valid [0:total_seq_len) region (and, ideally,
// EXPECT the padding region to differ) so the test turns into a real tripwire that goes green for
// the right reason instead of a landmine.
TEST_F(QnnHTPBackendTests, DISABLED_GroupQueryAttention_PaddedCache_NoisePadding_FP32) {
  RunHTPPackedGQATest<float>(8, 4, 32, /*seq*/ 1, /*total*/ 16, /*scale*/ 0.0f, /*do_rotary*/ 0,
                             /*fp32_abs_err*/ 1e-2f, /*max_seq_len*/ 128);
}

// Unpacked (separate Q / K / V) tests. These mirror the packed coverage above but exercise the
// three-separate-inputs path that many models use instead of a single packed-QKV tensor.
//

// Basic unpacked GQA (FP32 model, FP16 on device), decode geometry, scale=0 default.
TEST_F(QnnHTPBackendTests, GroupQueryAttention_Unpacked_Basic_FP32) {
  // num_heads=8, kv_num_heads=4, head_size=32, decode (seq=1), total=1024, scale=0 (default).
  RunHTPUnpackedGQATest<float>(8, 4, 32, 1, 1024, /*scale*/ 0.0f, /*do_rotary*/ 0);
}

// Explicit non-default scale on the unpacked path.
TEST_F(QnnHTPBackendTests, GroupQueryAttention_Unpacked_ScaleExplicit_FP32) {
  RunHTPUnpackedGQATest<float>(8, 4, 32, 1, 1024, /*scale*/ 0.5f, /*do_rotary*/ 0);
}

// Extreme grouping: kv_num_heads == 1 (multi-query attention), unpacked inputs.
TEST_F(QnnHTPBackendTests, GroupQueryAttention_Unpacked_MQA_KvOne_FP32) {
  RunHTPUnpackedGQATest<float>(8, 1, 32, 1, 1024, /*scale*/ 0.0f, /*do_rotary*/ 0);
}

// Prefill: sequence_length > 1, unpacked inputs.
TEST_F(QnnHTPBackendTests, GroupQueryAttention_Unpacked_Prefill_FP32) {
  RunHTPUnpackedGQATest<float>(8, 4, 32, /*seq*/ 64, /*total*/ 1024, /*scale*/ 0.0f, /*do_rotary*/ 0);
}

// Rotary embeddings enabled, unpacked inputs.
TEST_F(QnnHTPBackendTests, GroupQueryAttention_Unpacked_Rotary_FP32) {
  RunHTPUnpackedGQATest<float>(8, 4, 32, 1, 1024, /*scale*/ 0.0f, /*do_rotary*/ 1);
}

// FP16 query/cache path, unpacked inputs.
TEST_F(QnnHTPBackendTests, GroupQueryAttention_Unpacked_Basic_FP16) {
  RunHTPUnpackedGQATest<Ort::Float16_t>(8, 4, 32, 1, 1024, /*scale*/ 0.0f, /*do_rotary*/ 0);
}

// === op_affinity EP-assignment gate tests ===
// These check ONLY the QNN EP's partitioning / session-creation decision for the op_affinity gate
// (see core/providers/qnn/builder/opbuilder/group_query_attention_op_builder.cc IsOpSupported and
// core/providers/qnn/qnn_op_affinity_map.h). They deliberately do NOT run inference: unlike
// RunGQATest/RunHTPPackedGQATest above (which couple EP-assignment verification with a buffer-shared
// device Run against a CPU reference), these only need to observe whether GQA ends up on QNN EP or
// whether session creation itself fails -- so they build a minimal GQA model directly with
// BuildGQATestCase and drive session creation the same way RunGQATest does (see the ProviderOptions /
// RegisterQnnEpLibrary / ScopedOrtSession / VerifyEPNodeAssignment usage at lines ~188-208 above),
// with one addition: an "op_affinity" provider option pointing at a temp JSON config file.

// Writes `contents` to a uniquely-named temp file (tagged so parallel tests don't collide) and
// returns its path. Caller deletes it. Mirrors WriteTempConfig in
// test/providers/qnn/unit/qnn_op_affinity_map_test.cc.
static std::filesystem::path WriteOpAffinityConfig(const std::string& contents, const std::string& tag) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / ("gqa_op_affinity_" + tag + ".json");
  std::ofstream ofs(path);
  ofs << contents;
  ofs.close();
  return path;
}

// Builds a minimal packed-QKV GQA model (decode geometry: num_heads=8, kv_num_heads=4, head_size=32,
// sequence_length=1, total_seq_len=1024, scale=0 default, do_rotary=0 -- the same shape used by
// GroupQueryAttention_Basic_FP32's RunHTPPackedGQATest<float>(8, 4, 32, 1, 1024, 0.0f, 0)
// call above), then creates a QNN session with the given backend and (optional) op_affinity config
// path set as provider options. Does NOT run inference.
//
// On success, *expected_ep_assignment is verified via VerifyEPNodeAssignment (mirrors the pattern at
// lines ~200-208 above). On failure, the Ort::Exception is caught and reported via *session_failed
// (mirrors the try/catch pattern in qnn_basic_test.cc's TestDisableCPUFallback_BackendNotFound,
// around lines 70-78 of that file).
static void RunGQAOpAffinityAssignmentCheck(const std::string& backend_name,
                                            const std::optional<std::string>& op_affinity_path,
                                            ExpectedEPNodeAssignment expected_ep_assignment,
                                            bool* session_failed) {
  const int32_t num_heads = 8;
  const int32_t kv_num_heads = 4;
  const int32_t head_size = 32;
  const int32_t sequence_length = 1;
  const int32_t total_seq_len = 1024;
  const int32_t batch_size = 1;
  const int32_t packed_qkv_d = num_heads * head_size + 2 * kv_num_heads * head_size;

  auto query_def = TestInputDef<float>({batch_size, sequence_length, packed_qkv_d},
                                       false, -1.0f, 1.0f);
  const std::optional<std::reference_wrapper<TestInputDef<float>>> key_def = std::nullopt;
  const std::optional<std::reference_wrapper<TestInputDef<float>>> value_def = std::nullopt;

  TestInputDef<float> pk_max({batch_size, kv_num_heads, total_seq_len, head_size}, false, -1.0f, 1.0f);
  TestInputDef<float> pv_max({batch_size, kv_num_heads, total_seq_len, head_size}, false, -1.0f, 1.0f);
  std::optional<std::reference_wrapper<TestInputDef<float>>> past_key_def = std::ref(pk_max);
  std::optional<std::reference_wrapper<TestInputDef<float>>> past_value_def = std::ref(pv_max);

  std::vector<int32_t> seqlens_k_data(batch_size, total_seq_len - 1);
  auto seqlens_k_def = TestInputDef<int32_t>({batch_size}, true, seqlens_k_data);
  auto total_sequence_length_def = TestInputDef<int32_t>({}, true, std::vector<int32_t>{total_seq_len});

  const std::optional<std::reference_wrapper<TestInputDef<float>>> cos_cache_def = std::nullopt;
  const std::optional<std::reference_wrapper<TestInputDef<float>>> sin_cache_def = std::nullopt;
  const std::optional<std::reference_wrapper<TestInputDef<int64_t>>> position_ids_def = std::nullopt;
  const std::optional<std::reference_wrapper<TestInputDef<float>>> attention_bias_def = std::nullopt;
  const std::optional<std::reference_wrapper<TestInputDef<float>>> head_sink_def = std::nullopt;

  const GetTestModelFn build_test_case = BuildGQATestCase<float>(
      query_def, key_def, value_def, past_key_def, past_value_def,
      seqlens_k_def, total_sequence_length_def,
      cos_cache_def, sin_cache_def, position_ids_def, attention_bias_def, head_sink_def,
      /*do_rotary*/ 0,
      /*k_quant_type*/ std::nullopt,
      /*kv_cache_bit_width*/ std::nullopt,
      kv_num_heads,
      /*local_window_size*/ std::nullopt,
      num_heads,
      /*qk_output*/ std::nullopt,
      /*rotary_interleaved*/ std::nullopt,
      /*scale*/ 0.0f,
      /*smooth_softmax*/ std::nullopt,
      /*v_quant_type*/ std::nullopt);

  ModelTestBuilder helper;
  build_test_case(helper);

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  std::string model_data;
  helper.model_.SerializeToString(&model_data);

  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_name;
  if (op_affinity_path.has_value()) {
    provider_options["op_affinity"] = *op_affinity_path;
  }

  *session_failed = false;
  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions qnn_so;
  qnn_so.AddConfigEntry(kOrtSessionOptionsRecordEpGraphAssignmentInfo, "1");
  RegisterQnnEpLibrary(registered_ep_device, qnn_so, kQnnExecutionProvider, provider_options);
  try {
    ScopedOrtSession scoped_qnn_session(
        std::move(registered_ep_device),
        Ort::Session(*GetOrtEnv(), model_data.data(), static_cast<int>(model_data.size()), qnn_so));
    Ort::Session& qnn_session = scoped_qnn_session.session();
    ASSERT_NO_FATAL_FAILURE(VerifyEPNodeAssignment(qnn_session, kQnnExecutionProvider, expected_ep_assignment));
  } catch (const Ort::Exception&) {
    *session_failed = true;
  }
}

// No op_affinity config: HTP is opt-in by default (see OpAffinityMap::Evaluate's per-backend
// default), so GQA is NOT assigned to QNN.
TEST_F(QnnHTPBackendTests, GroupQueryAttention_OpAffinity_HtpNoConfig_NotAssigned) {
  bool session_failed = false;
  RunGQAOpAffinityAssignmentCheck("htp", std::nullopt, ExpectedEPNodeAssignment::None, &session_failed);
  ASSERT_FALSE(session_failed);
}

// op_affinity pins GroupQueryAttention to HTP, session runs HTP -> assigned to QNN.
TEST_F(QnnHTPBackendTests, GroupQueryAttention_OpAffinity_HtpPinHtp_Assigned) {
  const auto path = WriteOpAffinityConfig(R"({ "op_type": { "GroupQueryAttention": "HTP" } })", "htp_pin_htp");
  bool session_failed = false;
  RunGQAOpAffinityAssignmentCheck("htp", path.string(), ExpectedEPNodeAssignment::All, &session_failed);
  ASSERT_FALSE(session_failed);
  std::filesystem::remove(path);
}

// op_affinity pins GroupQueryAttention to GPU, but the session runs HTP -> pin can never be
// honored, so session creation must fail (ValidateForSessionBackend reports an error).
TEST_F(QnnHTPBackendTests, GroupQueryAttention_OpAffinity_HtpPinGpu_SessionFails) {
  const auto path = WriteOpAffinityConfig(R"({ "op_type": { "GroupQueryAttention": "GPU" } })", "htp_pin_gpu");
  bool session_failed = false;
  RunGQAOpAffinityAssignmentCheck("htp", path.string(), ExpectedEPNodeAssignment::None, &session_failed);
  ASSERT_TRUE(session_failed);
  std::filesystem::remove(path);
}

// op_affinity pins GroupQueryAttention to CPU: a legitimate silent-off intent, so GQA is NOT
// assigned to QNN but session creation still succeeds (falls back to CPU EP for that node).
TEST_F(QnnHTPBackendTests, GroupQueryAttention_OpAffinity_PinCpu_NotAssigned) {
  const auto path = WriteOpAffinityConfig(R"({ "op_type": { "GroupQueryAttention": "CPU" } })", "pin_cpu");
  bool session_failed = false;
  RunGQAOpAffinityAssignmentCheck("htp", path.string(), ExpectedEPNodeAssignment::None, &session_failed);
  ASSERT_FALSE(session_failed);
  std::filesystem::remove(path);
}

// op_affinity points at a config file that does not exist -> FromConfigFile throws
// std::runtime_error at EP construction time, so session creation must fail.
TEST_F(QnnHTPBackendTests, GroupQueryAttention_OpAffinity_MissingConfigFile_SessionFails) {
  const std::filesystem::path missing =
      std::filesystem::temp_directory_path() / "gqa_op_affinity_does_not_exist_12345.json";
  bool session_failed = false;
  RunGQAOpAffinityAssignmentCheck("htp", missing.string(), ExpectedEPNodeAssignment::None, &session_failed);
  ASSERT_TRUE(session_failed);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
