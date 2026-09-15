// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <atomic>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/util/include/api_asserts.h"
#include "gtest/gtest.h"

// QNN_GROUP_QUERY_ATTENTION_AVAILABLE is available from QNN API 2.37 (QAIRT 2.48).
// QNN_HTP_GROUP_QUERY_ATTENTION_AVAILABLE is available from QNN API 2.38 (QAIRT 2.49).
// Defined here (duplicated from core/providers/qnn/builder/qnn_def.h) so this test file does not
// need to include that EP-private header.
#if QNN_API_VERSION_MAJOR > 2 || \
    (QNN_API_VERSION_MAJOR == 2 && QNN_API_VERSION_MINOR >= 37)
#define QNN_GROUP_QUERY_ATTENTION_AVAILABLE
#endif

#if QNN_API_VERSION_MAJOR > 2 || \
    (QNN_API_VERSION_MAJOR == 2 && QNN_API_VERSION_MINOR >= 38)
#define QNN_HTP_GROUP_QUERY_ATTENTION_AVAILABLE
#endif

namespace onnxruntime {
namespace test {

#ifdef QNN_GROUP_QUERY_ATTENTION_AVAILABLE

#if (defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)) && defined(QNN_HTP_GROUP_QUERY_ATTENTION_AVAILABLE)

template <typename T, typename M>
struct GqaTestConfig {
  // Required GQA inputs.
  TestInputDef<T> query_def;
  TestInputDef<M> seqlens_k_def;
  TestInputDef<M> total_sequence_length_def;

  // Optional GQA inputs.
  std::optional<TestInputDef<T>> key_def;
  std::optional<TestInputDef<T>> value_def;
  std::optional<TestInputDef<T>> past_key_def;
  std::optional<TestInputDef<T>> past_value_def;
  std::optional<TestInputDef<T>> cos_cache_def;
  std::optional<TestInputDef<T>> sin_cache_def;
  std::optional<TestInputDef<int64_t>> position_ids_def;
  std::optional<TestInputDef<T>> attention_bias_def;
  std::optional<TestInputDef<T>> head_sink_def;

  // GQA attributes.
  std::optional<int32_t> do_rotary;
  std::optional<std::string> k_quant_type;
  std::optional<int32_t> kv_cache_bit_width;
  int32_t kv_num_heads{};
  std::optional<int32_t> local_window_size;
  int32_t num_heads{};
  std::optional<int32_t> qk_output;
  std::optional<int32_t> rotary_interleaved;
  std::optional<float> scale;
  std::optional<int32_t> smooth_softmax;
  std::optional<std::string> v_quant_type;

  // Test execution settings.
  ExpectedEPNodeAssignment expected_ep_assignment{ExpectedEPNodeAssignment::All};
  std::string backend_name;
  int opset{13};
  TensorVerifier tensor_verifier{ElementwiseAbsoluteVerifier{1e-5f}};
  bool use_shared_memory_allocator{false};
  std::optional<std::string> op_affinity_path;
};

template <typename T, typename M>
static GetTestModelFn BuildGQATestCase(const GqaTestConfig<T, M>& config) {
  return [config](ModelTestBuilder& builder) {
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

    input_names.push_back(add_input_T("query", config.query_def));
    input_names.push_back(config.key_def ? add_input_T("key", *config.key_def) : "");
    input_names.push_back(config.value_def ? add_input_T("value", *config.value_def) : "");
    input_names.push_back(config.past_key_def ? add_input_T("past_key", *config.past_key_def) : "");
    input_names.push_back(config.past_value_def ? add_input_T("past_value", *config.past_value_def) : "");
    input_names.push_back(add_input_M("seqlens_k", config.seqlens_k_def));
    input_names.push_back(add_input_M("total_sequence_length", config.total_sequence_length_def));
    input_names.push_back(config.cos_cache_def ? add_input_T("cos_cache", *config.cos_cache_def) : "");
    input_names.push_back(config.sin_cache_def ? add_input_T("sin_cache", *config.sin_cache_def) : "");
    input_names.push_back(config.position_ids_def ? add_input_I64("position_ids", *config.position_ids_def) : "");
    input_names.push_back(config.attention_bias_def ? add_input_T("attention_bias", *config.attention_bias_def) : "");
    input_names.push_back(config.head_sink_def ? add_input_T("head_sink", *config.head_sink_def) : "");

    std::vector<std::string> output_names;

    builder.MakeOutput("output");
    output_names.push_back("output");

    builder.MakeOutput("present_key");
    output_names.push_back("present_key");

    builder.MakeOutput("present_value");
    output_names.push_back("present_value");

    if (config.qk_output.has_value() && config.qk_output.value() != 0) {
      builder.MakeOutput("output_qk");
      output_names.push_back("output_qk");
    }

    std::vector<ONNX_NAMESPACE::AttributeProto> attrs;

    attrs.push_back(builder.MakeScalarAttribute("num_heads", static_cast<int64_t>(config.num_heads)));
    attrs.push_back(builder.MakeScalarAttribute("kv_num_heads", static_cast<int64_t>(config.kv_num_heads)));

    if (config.do_rotary.has_value())
      attrs.push_back(builder.MakeScalarAttribute("do_rotary", static_cast<int64_t>(config.do_rotary.value())));
    if (config.local_window_size.has_value())
      attrs.push_back(builder.MakeScalarAttribute("local_window_size", static_cast<int64_t>(config.local_window_size.value())));
    if (config.rotary_interleaved.has_value())
      attrs.push_back(builder.MakeScalarAttribute("rotary_interleaved", static_cast<int64_t>(config.rotary_interleaved.value())));
    if (config.scale.has_value())
      attrs.push_back(builder.MakeScalarAttribute("scale", config.scale.value()));
    if (config.smooth_softmax.has_value())
      attrs.push_back(builder.MakeScalarAttribute("smooth_softmax", static_cast<int64_t>(config.smooth_softmax.value())));
    if (config.qk_output.has_value())
      attrs.push_back(builder.MakeScalarAttribute("qk_output", static_cast<int64_t>(config.qk_output.value())));
    if (config.kv_cache_bit_width.has_value())
      attrs.push_back(builder.MakeScalarAttribute("kv_cache_bit_width", static_cast<int64_t>(config.kv_cache_bit_width.value())));
    if (config.k_quant_type.has_value())
      attrs.push_back(builder.MakeStringAttribute("k_quant_type", config.k_quant_type.value()));
    if (config.v_quant_type.has_value())
      attrs.push_back(builder.MakeStringAttribute("v_quant_type", config.v_quant_type.value()));

    builder.AddNode("GQA",
                    "GroupQueryAttention",
                    input_names,
                    output_names,
                    kMSDomain,
                    attrs);
  };
}

// === Shared model builder + backend-agnostic driver (used by HTP now, GPU after rebase) ===

// Holds a shared-memory allocation plus the Ort::Value view over it, so GQA feeds
// (and in-place present/past KV buffers) live on the QNN host-accessible allocator.
struct GQAFeedCopy {
  Ort::MemoryAllocation allocation;
  Ort::Value value{nullptr};
};

// Copies each graph input from source_feeds into a tensor backed by `allocator` /
// `memory_info`, returning the feed copies (index-aligned with input_names) and a
// name->index map. Verbatim extraction of the per-input copy loop from RunGQATest.
static void MakeSharedMemoryFeeds(const std::vector<std::string>& input_names,
                                  const std::unordered_map<std::string, Ort::Value>& source_feeds,
                                  Ort::Allocator& allocator,
                                  const Ort::MemoryInfo& memory_info,
                                  std::vector<GQAFeedCopy>& qnn_feeds,
                                  std::unordered_map<std::string, size_t>& input_name_to_index) {
  qnn_feeds.reserve(input_names.size());
  for (const auto& input_name : input_names) {
    const Ort::Value& source_value = source_feeds.at(input_name);
    const auto tensor_info = source_value.GetTensorTypeAndShapeInfo();
    const auto shape = tensor_info.GetShape();
    const size_t num_bytes = source_value.GetTensorSizeInBytes();
    const auto* source_data = reinterpret_cast<const std::byte*>(source_value.GetTensorRawData());

    GQAFeedCopy feed_copy{allocator.GetAllocation(num_bytes)};
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
}

// Aliases present_key/present_value outputs onto the past_key/past_value shared-memory
// buffers (in-place KV cache), runs the QNN session, then collects output views. Outputs
// that are not aliased are owned by owned_qnn_outputs. Verbatim extraction from RunGQATest.
static void AliasPresentToPastAndRun(Ort::Session& qnn_session,
                                     const std::vector<std::string>& output_names,
                                     const std::vector<const char*>& input_names_cstr,
                                     const std::vector<const char*>& output_names_cstr,
                                     std::vector<GQAFeedCopy>& qnn_feeds,
                                     const std::unordered_map<std::string, size_t>& input_name_to_index,
                                     std::vector<Ort::Value>& owned_qnn_outputs,
                                     std::vector<const Ort::Value*>& qnn_outputs) {
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

  owned_qnn_outputs.reserve(output_names.size());
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
}

// Runs the model on the CPU EP and compares each QNN output against it. Verbatim
// extraction of RunGQATest's final compare block.
static void CompareQnnVsCpuOutputs(Ort::Session& cpu_session,
                                   const std::unordered_map<std::string, Ort::Value>& cpu_feeds,
                                   const std::vector<std::string>& output_names,
                                   const std::vector<const Ort::Value*>& qnn_outputs,
                                   const TensorVerifier& tensor_verifier) {
  Ort::RunOptions cpu_run_options;
  std::vector<Ort::Value> cpu_outputs;
  // The CPU EP can do GQA without buffer sharing, so we can just use RunWithEP
  RunWithEP(cpu_session, cpu_run_options, cpu_feeds, cpu_outputs);

  // Check QNN outputs against CPU
  ASSERT_EQ(cpu_outputs.size(), output_names.size());
  ASSERT_EQ(qnn_outputs.size(), output_names.size());
  for (size_t i = 0; i < output_names.size(); i++) {
    VerifyOutput(output_names[i], cpu_outputs[i], *qnn_outputs[i], tensor_verifier);
  }
}

// Runs a model with a GQA operator through QNN EP. Checks the graph node assignment
// and that inference outputs for QNN EP and CPU EP match.
template <typename T, typename M>
static void RunGQATest(const GqaTestConfig<T, M>& config) {
  const GetTestModelFn build_test_case = BuildGQATestCase(config);
  // The GPU backend only supports GQA with past/present buffer sharing on gpu-accessible shared memory. So, the GQA UTs do not
  // use RunQnnModelTest and instead manually create/run the QNN inference session with the buffer sharing.
  ModelTestBuilder helper;
  build_test_case(helper);

  const std::unordered_map<std::string, int> domain_to_version = {{"", config.opset}, {kMSDomain, 1}};
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  std::string model_data;
  helper.model_.SerializeToString(&model_data);

  ProviderOptions provider_options;
  provider_options["backend_type"] = config.backend_name;

  // On HTP the EP seeds a default CPU pin for GQA (HTP is opt-in); callers targeting HTP pass a
  // config pinning GQA -> HTP, else the op is filtered off QNN and the assignment check below fails.
  if (config.op_affinity_path.has_value()) {
    provider_options["op_affinity"] = *config.op_affinity_path;
  }
  if (config.use_shared_memory_allocator) {
    // GPU and HTP use different shared-memory allocators. Both expose the same "QnnHtpShared"
    // host-accessible MemoryInfo, but they are enabled via different provider options.
    if (config.backend_name == "gpu") {
      provider_options["enable_dx12_shared_memory_allocator"] = "1";
    } else if (config.backend_name == "htp") {
      provider_options["enable_htp_shared_memory_allocator"] = "1";
    }
  }
#if defined(__linux__) && !defined(__aarch64__)
  if (config.backend_name == "htp") {
    provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
  }
#endif

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions qnn_so;
  qnn_so.AddConfigEntry(kOrtSessionOptionsRecordEpGraphAssignmentInfo, "1");
  RegisterQnnEpLibrary(registered_ep_device, qnn_so, kQnnExecutionProvider, provider_options);
  ScopedOrtSession scoped_qnn_session(
      std::move(registered_ep_device),
      Ort::Session(*GetOrtEnv(), model_data.data(), static_cast<int>(model_data.size()), qnn_so));
  Ort::Session& qnn_session = scoped_qnn_session.session();
  ASSERT_NO_FATAL_FAILURE(VerifyEPNodeAssignment(qnn_session, kQnnExecutionProvider, config.expected_ep_assignment));

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

  Ort::MemoryInfo memory_info(nullptr);
  Ort::Allocator allocator(nullptr);
  if (config.use_shared_memory_allocator && (config.backend_name == "gpu" || config.backend_name == "htp")) {
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

  std::vector<GQAFeedCopy> qnn_feeds;
  std::unordered_map<std::string, size_t> input_name_to_index;
  ASSERT_NO_FATAL_FAILURE(MakeSharedMemoryFeeds(input_names, helper.feeds_, allocator, memory_info,
                                                qnn_feeds, input_name_to_index));

  std::vector<Ort::Value> owned_qnn_outputs;
  std::vector<const Ort::Value*> qnn_outputs;
  ASSERT_NO_FATAL_FAILURE(AliasPresentToPastAndRun(qnn_session, output_names, input_names_cstr,
                                                   output_names_cstr, qnn_feeds, input_name_to_index,
                                                   owned_qnn_outputs, qnn_outputs));

  ASSERT_NO_FATAL_FAILURE(CompareQnnVsCpuOutputs(cpu_session, helper.feeds_, output_names,
                                                 qnn_outputs, config.tensor_verifier));
}

class ScopedGqaOpAffinityConfig {
 public:
  explicit ScopedGqaOpAffinityConfig(const std::string& contents, const std::string& tag = CurrentTestTag())
      : path_(MakeUniquePath(tag)) {
    std::ofstream ofs(path_);
    ofs << contents;
    if (!ofs) {
      ADD_FAILURE() << "ScopedGqaOpAffinityConfig: failed to write " << path_;
    }
  }

  ~ScopedGqaOpAffinityConfig() {
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }

  ScopedGqaOpAffinityConfig(const ScopedGqaOpAffinityConfig&) = delete;
  ScopedGqaOpAffinityConfig& operator=(const ScopedGqaOpAffinityConfig&) = delete;

  const std::filesystem::path& path() const { return path_; }

  static std::filesystem::path MakeUniquePath(const std::string& tag) {
    static std::atomic<uint64_t> next_id{0};
    std::string sanitized_tag;
    sanitized_tag.reserve(tag.size());
    for (const char ch : tag) {
      sanitized_tag.push_back(std::isalnum(static_cast<unsigned char>(ch)) ? ch : '_');
    }
    return std::filesystem::temp_directory_path() /
           ("gqa_op_affinity_" + sanitized_tag + "_" + std::to_string(next_id.fetch_add(1)) + ".json");
  }

 private:
  static std::string CurrentTestTag() {
    const ::testing::TestInfo* info = ::testing::UnitTest::GetInstance()->current_test_info();
    return info == nullptr ? "gqa" : std::string(info->test_suite_name()) + "_" + info->name();
  }

  std::filesystem::path path_;
};

// Compact driver for HTP GQA tests over a packed-QKV model with a full-capacity past KV cache.
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
  // GQA op validation fails on HTP (QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE) on V68 and below.
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  const int32_t batch_size = 1;
  const int32_t packed_qkv_d = num_heads * head_size + 2 * kv_num_heads * head_size;
  if (max_seq_len <= 0) {
    max_seq_len = total_seq_len;  // default: cache capacity == valid length (no padding region)
  }
  GqaTestConfig<T, int32_t> config{
      TestInputDef<T>({batch_size, sequence_length, packed_qkv_d}, false, static_cast<T>(-1.0f), static_cast<T>(1.0f)),
      TestInputDef<int32_t>({batch_size}, true, std::vector<int32_t>(batch_size, total_seq_len - 1)),
      TestInputDef<int32_t>({}, true, std::vector<int32_t>{total_seq_len})};

  // Past KV buffer has capacity max_seq_len, filled with random data (so when max_seq_len >
  // total_seq_len the padding region naturally holds noise values).
  config.past_key_def.emplace(std::vector<int64_t>{batch_size, kv_num_heads, max_seq_len, head_size},
                              false, static_cast<T>(-1.0f), static_cast<T>(1.0f));
  config.past_value_def.emplace(std::vector<int64_t>{batch_size, kv_num_heads, max_seq_len, head_size},
                                false, static_cast<T>(-1.0f), static_cast<T>(1.0f));

  // Rotary caches (only consumed when do_rotary != 0, but cheap to always build). Sized to cache
  // capacity (max_seq_len rows).
  if (do_rotary != 0) {
    config.cos_cache_def.emplace(std::vector<int64_t>{max_seq_len, head_size / 2}, true,
                                 static_cast<T>(-1.0f), static_cast<T>(1.0f));
    config.sin_cache_def.emplace(std::vector<int64_t>{max_seq_len, head_size / 2}, true,
                                 static_cast<T>(-1.0f), static_cast<T>(1.0f));
  }

  config.do_rotary = do_rotary;
  config.kv_num_heads = kv_num_heads;
  config.num_heads = num_heads;
  config.scale = scale;
  config.expected_ep_assignment = ExpectedEPNodeAssignment::All;
  config.backend_name = "htp";
  config.tensor_verifier = ElementwiseAbsoluteVerifier{fp32_abs_err};
#if !defined(__linux__)
  config.use_shared_memory_allocator = true;
#endif  // !defined(__linux__)

  const ScopedGqaOpAffinityConfig affinity_config(R"({ "op_type": { "GroupQueryAttention": "HTP" } })");
  config.op_affinity_path = affinity_config.path().string();
  RunGQATest(config);
}

// Compact driver for HTP GQA tests over an unpacked (separate Q / K / V) model with a full-capacity
// past KV cache.
template <typename T>
static void RunHTPUnpackedGQATest(int32_t num_heads,
                                  int32_t kv_num_heads,
                                  int32_t head_size,
                                  int32_t sequence_length,
                                  int32_t total_seq_len,
                                  float scale,
                                  int32_t do_rotary,
                                  float fp32_abs_err = 1e-2f) {
  // GQA op validation fails on HTP (QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE) on V68 and below.
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  const int32_t batch_size = 1;
  const int32_t q_hidden = num_heads * head_size;
  const int32_t kv_hidden = kv_num_heads * head_size;

  GqaTestConfig<T, int32_t> config{
      TestInputDef<T>({batch_size, sequence_length, q_hidden}, false, static_cast<T>(-1.0f), static_cast<T>(1.0f)),
      TestInputDef<int32_t>({batch_size}, true, std::vector<int32_t>(batch_size, total_seq_len - 1)),
      TestInputDef<int32_t>({}, true, std::vector<int32_t>{total_seq_len})};
  config.key_def.emplace(std::vector<int64_t>{batch_size, sequence_length, kv_hidden},
                         false, static_cast<T>(-1.0f), static_cast<T>(1.0f));
  config.value_def.emplace(std::vector<int64_t>{batch_size, sequence_length, kv_hidden},
                           false, static_cast<T>(-1.0f), static_cast<T>(1.0f));
  config.past_key_def.emplace(std::vector<int64_t>{batch_size, kv_num_heads, total_seq_len, head_size},
                              false, static_cast<T>(-1.0f), static_cast<T>(1.0f));
  config.past_value_def.emplace(std::vector<int64_t>{batch_size, kv_num_heads, total_seq_len, head_size},
                                false, static_cast<T>(-1.0f), static_cast<T>(1.0f));
  if (do_rotary != 0) {
    config.cos_cache_def.emplace(std::vector<int64_t>{total_seq_len, head_size / 2}, true,
                                 static_cast<T>(-1.0f), static_cast<T>(1.0f));
    config.sin_cache_def.emplace(std::vector<int64_t>{total_seq_len, head_size / 2}, true,
                                 static_cast<T>(-1.0f), static_cast<T>(1.0f));
  }

  config.do_rotary = do_rotary;
  config.kv_num_heads = kv_num_heads;
  config.num_heads = num_heads;
  config.scale = scale;
  config.expected_ep_assignment = ExpectedEPNodeAssignment::All;
  config.backend_name = "htp";
  config.tensor_verifier = ElementwiseAbsoluteVerifier{fp32_abs_err};
#if !defined(__linux__)
  config.use_shared_memory_allocator = true;
#endif  // !defined(__linux__)

  const ScopedGqaOpAffinityConfig affinity_config(R"({ "op_type": { "GroupQueryAttention": "HTP" } })");
  config.op_affinity_path = affinity_config.path().string();
  RunGQATest(config);
}

// === HTP inference tests (QNN vs CPU) ===

TEST_F(QnnHTPBackendTests, GroupQueryAttention_Basic_FP32) {
  // num_heads=8, kv_num_heads=4, head_size=32, decode (seq=1), total=1024, scale=0 (default).
  RunHTPPackedGQATest<float>(8, 4, 32, 1, 1024, /*scale*/ 0.0f, /*do_rotary*/ 0);
}

TEST_F(QnnHTPBackendTests, GroupQueryAttention_ScaleExplicit_FP32) {
  RunHTPPackedGQATest<float>(8, 4, 32, 1, 1024, /*scale*/ 0.5f, /*do_rotary*/ 0);
}

TEST_F(QnnHTPBackendTests, GroupQueryAttention_MHA_NumHeadsEqKv_FP32) {
  RunHTPPackedGQATest<float>(8, 8, 32, 1, 1024, /*scale*/ 0.0f, /*do_rotary*/ 0);
}

TEST_F(QnnHTPBackendTests, GroupQueryAttention_MQA_KvOne_FP32) {
  RunHTPPackedGQATest<float>(8, 1, 32, 1, 1024, /*scale*/ 0.0f, /*do_rotary*/ 0);
}

TEST_F(QnnHTPBackendTests, GroupQueryAttention_Prefill_FP32) {
  RunHTPPackedGQATest<float>(8, 4, 32, /*seq*/ 64, /*total*/ 1024, /*scale*/ 0.0f, /*do_rotary*/ 0);
}

TEST_F(QnnHTPBackendTests, GroupQueryAttention_Llama3_AR1_FP32) {
  RunHTPPackedGQATest<float>(32, 8, 64, 1, 1024, /*scale*/ 0.0f, /*do_rotary*/ 0);
}

TEST_F(QnnHTPBackendTests, GroupQueryAttention_Rotary_FP32) {
  RunHTPPackedGQATest<float>(8, 4, 32, 1, 1024, /*scale*/ 0.0f, /*do_rotary*/ 1);
}

TEST_F(QnnHTPBackendTests, GroupQueryAttention_Basic_FP16) {
  RunHTPPackedGQATest<Ort::Float16_t>(8, 4, 32, 1, 1024, /*scale*/ 0.0f, /*do_rotary*/ 0);
}

TEST_F(QnnHTPBackendTests, GroupQueryAttention_Unpacked_Basic_FP32) {
  // num_heads=8, kv_num_heads=4, head_size=32, decode (seq=1), total=1024, scale=0 (default).
  RunHTPUnpackedGQATest<float>(8, 4, 32, 1, 1024, /*scale*/ 0.0f, /*do_rotary*/ 0);
}

TEST_F(QnnHTPBackendTests, GroupQueryAttention_Unpacked_ScaleExplicit_FP32) {
  RunHTPUnpackedGQATest<float>(8, 4, 32, 1, 1024, /*scale*/ 0.5f, /*do_rotary*/ 0);
}

TEST_F(QnnHTPBackendTests, GroupQueryAttention_Unpacked_MQA_KvOne_FP32) {
  RunHTPUnpackedGQATest<float>(8, 1, 32, 1, 1024, /*scale*/ 0.0f, /*do_rotary*/ 0);
}

TEST_F(QnnHTPBackendTests, GroupQueryAttention_Unpacked_Prefill_FP32) {
  RunHTPUnpackedGQATest<float>(8, 4, 32, /*seq*/ 64, /*total*/ 1024, /*scale*/ 0.0f, /*do_rotary*/ 0);
}

TEST_F(QnnHTPBackendTests, GroupQueryAttention_Unpacked_Rotary_FP32) {
  RunHTPUnpackedGQATest<float>(8, 4, 32, 1, 1024, /*scale*/ 0.0f, /*do_rotary*/ 1);
}

TEST_F(QnnHTPBackendTests, GroupQueryAttention_Unpacked_Basic_FP16) {
  RunHTPUnpackedGQATest<Ort::Float16_t>(8, 4, 32, 1, 1024, /*scale*/ 0.0f, /*do_rotary*/ 0);
}

// === op_affinity EP-assignment gate tests ===
// These check ONLY the QNN EP's partitioning / session-creation decision for the op_affinity gate
static void RunGQAOpAffinityAssignmentCheck(const std::string& backend_name,
                                            const std::optional<std::string>& op_affinity_path,
                                            ExpectedEPNodeAssignment expected_ep_assignment,
                                            bool* session_failed,
                                            std::string* session_error) {
  const int32_t num_heads = 8;
  const int32_t kv_num_heads = 4;
  const int32_t head_size = 32;
  const int32_t sequence_length = 1;
  const int32_t total_seq_len = 128;
  const int32_t batch_size = 1;
  const int32_t packed_qkv_d = num_heads * head_size + 2 * kv_num_heads * head_size;

  GqaTestConfig<float, int32_t> config{
      TestInputDef<float>({batch_size, sequence_length, packed_qkv_d}, false, -1.0f, 1.0f),
      TestInputDef<int32_t>({batch_size}, true, std::vector<int32_t>(batch_size, total_seq_len - 1)),
      TestInputDef<int32_t>({}, true, std::vector<int32_t>{total_seq_len})};
  config.past_key_def.emplace(std::vector<int64_t>{batch_size, kv_num_heads, total_seq_len, head_size},
                              false, -1.0f, 1.0f);
  config.past_value_def.emplace(std::vector<int64_t>{batch_size, kv_num_heads, total_seq_len, head_size},
                                false, -1.0f, 1.0f);
  config.do_rotary = 0;
  config.kv_num_heads = kv_num_heads;
  config.num_heads = num_heads;
  config.scale = 0.0f;

  const GetTestModelFn build_test_case = BuildGQATestCase(config);
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
  session_error->clear();
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
  } catch (const Ort::Exception& e) {
    *session_failed = true;
    *session_error = e.what();
  }
}

TEST_F(QnnHTPBackendTests, GroupQueryAttention_OpAffinity_HtpNoConfig_NotAssigned) {
  bool session_failed = false;
  std::string session_error;
  RunGQAOpAffinityAssignmentCheck("htp", std::nullopt, ExpectedEPNodeAssignment::None, &session_failed, &session_error);
  ASSERT_FALSE(session_failed) << session_error;
}

TEST_F(QnnHTPBackendTests, GroupQueryAttention_OpAffinity_HtpPinHtp_Assigned) {
  // GQA op validation fails on HTP (QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE) on V68 and below,
  // so pinning it to HTP can't actually get the node assigned there.
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  const ScopedGqaOpAffinityConfig config(R"({ "op_type": { "GroupQueryAttention": "HTP" } })");
  bool session_failed = false;
  std::string session_error;
  RunGQAOpAffinityAssignmentCheck("htp", config.path().string(), ExpectedEPNodeAssignment::All, &session_failed, &session_error);
  ASSERT_FALSE(session_failed) << session_error;
}

TEST_F(QnnHTPBackendTests, GroupQueryAttention_OpAffinity_HtpPinGpu_SessionFails) {
  const ScopedGqaOpAffinityConfig config(R"({ "op_type": { "GroupQueryAttention": "GPU" } })");
  bool session_failed = false;
  std::string session_error;
  RunGQAOpAffinityAssignmentCheck("htp", config.path().string(), ExpectedEPNodeAssignment::None, &session_failed, &session_error);
  ASSERT_TRUE(session_failed);
}

TEST_F(QnnHTPBackendTests, GroupQueryAttention_OpAffinity_PinCpu_NotAssigned) {
  const ScopedGqaOpAffinityConfig config(R"({ "op_type": { "GroupQueryAttention": "CPU" } })");
  bool session_failed = false;
  std::string session_error;
  RunGQAOpAffinityAssignmentCheck("htp", config.path().string(), ExpectedEPNodeAssignment::None, &session_failed, &session_error);
  ASSERT_FALSE(session_failed) << session_error;
}

TEST_F(QnnHTPBackendTests, GroupQueryAttention_OpAffinity_MissingConfigFile_SessionFails) {
  const std::filesystem::path missing = ScopedGqaOpAffinityConfig::MakeUniquePath("missing");
  std::error_code ec;
  std::filesystem::remove(missing, ec);
  ASSERT_FALSE(std::filesystem::exists(missing));

  bool session_failed = false;
  std::string session_error;
  RunGQAOpAffinityAssignmentCheck("htp", missing.string(), ExpectedEPNodeAssignment::None, &session_failed, &session_error);
  ASSERT_TRUE(session_failed);
}

#endif  // (defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)) && defined(QNN_HTP_GROUP_QUERY_ATTENTION_AVAILABLE)

#if defined(_M_ARM64)
//
// GPU tests:
//

struct GpuPackedGqaCase {
  int32_t num_heads;
  int32_t kv_num_heads;
  int32_t head_size;
  int32_t sequence_length;
  int32_t total_seq_len;
  float scale;
  bool use_2d_seqlens{false};
  bool use_shared_memory_allocator{false};
};

template <typename T>
static void RunGpuPackedGQATest(const GpuPackedGqaCase& test_case) {
  const int32_t batch_size = 1;
  const int32_t packed_qkv_d = test_case.num_heads * test_case.head_size +
                               2 * test_case.kv_num_heads * test_case.head_size;
  const std::vector<int64_t> seqlens_k_shape =
      test_case.use_2d_seqlens ? std::vector<int64_t>{batch_size, 1} : std::vector<int64_t>{batch_size};

  GqaTestConfig<T, int32_t> config{
      TestInputDef<T>({batch_size, test_case.sequence_length, packed_qkv_d},
                      false, static_cast<T>(-1.0f), static_cast<T>(1.0f)),
      TestInputDef<int32_t>(seqlens_k_shape, true,
                            std::vector<int32_t>(batch_size, test_case.total_seq_len - 1)),
      TestInputDef<int32_t>({}, true, std::vector<int32_t>{test_case.total_seq_len})};
  config.past_key_def.emplace(std::vector<int64_t>{batch_size, test_case.kv_num_heads,
                                                   test_case.total_seq_len, test_case.head_size},
                              false, static_cast<T>(-1.0f), static_cast<T>(1.0f));
  config.past_value_def.emplace(std::vector<int64_t>{batch_size, test_case.kv_num_heads,
                                                     test_case.total_seq_len, test_case.head_size},
                                false, static_cast<T>(-1.0f), static_cast<T>(1.0f));
  config.cos_cache_def.emplace(std::vector<int64_t>{test_case.total_seq_len, test_case.head_size / 2},
                               true, static_cast<T>(-1.0f), static_cast<T>(1.0f));
  config.sin_cache_def.emplace(std::vector<int64_t>{test_case.total_seq_len, test_case.head_size / 2},
                               true, static_cast<T>(-1.0f), static_cast<T>(1.0f));
  config.do_rotary = 1;
  config.kv_num_heads = test_case.kv_num_heads;
  config.num_heads = test_case.num_heads;
  config.scale = test_case.scale;
  config.expected_ep_assignment = ExpectedEPNodeAssignment::All;
  config.backend_name = "gpu";
  config.opset = 13;
  config.tensor_verifier = CosineSimilarityVerifier{0.99f};
  config.use_shared_memory_allocator = test_case.use_shared_memory_allocator;
  RunGQATest(config);
}

TEST_F(QnnGPUBackendTests, GroupQueryAttention_Basic_FP32) {
  RunGpuPackedGQATest<float>({8, 4, 32, 1, 1024, 10.0f});
}

TEST_F(QnnGPUBackendTests, GroupQueryAttention_Basic_FP16) {
  RunGpuPackedGQATest<Ort::Float16_t>({8, 4, 32, 1, 1024, 10.0f});
}

TEST_F(QnnGPUBackendTests, GroupQueryAttention_2D_SeqlensK) {
  RunGpuPackedGQATest<float>({8, 4, 32, 1, 1024, 10.0f, /*use_2d_seqlens*/ true});
}

TEST_F(QnnGPUBackendTests, GroupQueryAttention_Llama3_1_AR1_FP32) {
  RunGpuPackedGQATest<float>({32, 8, 64, 1, 1024, 0.125f});
}

TEST_F(QnnGPUBackendTests, GroupQueryAttention_Llama3_1_AR1_FP16) {
  RunGpuPackedGQATest<Ort::Float16_t>({32, 8, 64, 1, 1024, 0.125f});
}

TEST_F(QnnGPUBackendTests, GroupQueryAttention_Llama3_1_AR64_FP32) {
  RunGpuPackedGQATest<float>({32, 8, 64, 64, 1024, 0.125f});
}

TEST_F(QnnGPUBackendTests, GroupQueryAttention_Llama3_1_AR64_FP16) {
  RunGpuPackedGQATest<Ort::Float16_t>({32, 8, 64, 64, 1024, 0.125f});
}

#if defined(_WIN32)

TEST_F(QnnGPUBackendTests, GroupQueryAttention_Basic_SharedMemoryAllocator_FP32) {
  RunGpuPackedGQATest<float>({8, 4, 32, 1, 1024, 10.0f, false, /*shared*/ true});
}

TEST_F(QnnGPUBackendTests, GroupQueryAttention_Basic_SharedMemoryAllocator_FP16) {
  RunGpuPackedGQATest<Ort::Float16_t>({8, 4, 32, 1, 1024, 10.0f, false, /*shared*/ true});
}

TEST_F(QnnGPUBackendTests, GroupQueryAttention_Llama3_1_AR1_SharedMemoryAllocator_FP32) {
  RunGpuPackedGQATest<float>({32, 8, 64, 1, 1024, 0.125f, false, /*shared*/ true});
}

TEST_F(QnnGPUBackendTests, GroupQueryAttention_Llama3_1_AR1_SharedMemoryAllocator_FP16) {
  RunGpuPackedGQATest<Ort::Float16_t>({32, 8, 64, 1, 1024, 0.125f, false, /*shared*/ true});
}

TEST_F(QnnGPUBackendTests, GroupQueryAttention_Llama3_1_AR64_SharedMemoryAllocator_FP32) {
  RunGpuPackedGQATest<float>({32, 8, 64, 64, 1024, 0.125f, false, /*shared*/ true});
}

TEST_F(QnnGPUBackendTests, GroupQueryAttention_Llama3_1_AR64_SharedMemoryAllocator_FP16) {
  RunGpuPackedGQATest<Ort::Float16_t>({32, 8, 64, 64, 1024, 0.125f, false, /*shared*/ true});
}

#endif  // defined(_WIN32)
#endif  // defined(_M_ARM64) GPU tests
#endif  // QNN_GROUP_QUERY_ATTENTION_AVAILABLE
}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
