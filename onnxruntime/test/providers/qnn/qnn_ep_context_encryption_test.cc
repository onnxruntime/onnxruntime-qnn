// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// EPContext binary encryption tests (ORT API v28+). Split out of
// qnn_ep_context_test.cc so this feature's test surface has its own file.

#include <cstdlib>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include "onnxruntime_cxx_api.h"

#if ORT_API_VERSION >= 28
#include "onnxruntime_experimental_c_api.h"
#include "onnxruntime_experimental_cxx_api.h"
#endif

#include "onnxruntime_ep_device_ep_metadata_keys.h"
#include "onnxruntime_session_options_config_keys.h"

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "HTP/QnnHtpCommon.h"

// in test_main.cc
extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

#if ORT_API_VERSION >= 28

// Shared XOR encrypt/decrypt state + OrtWriteNamedBufferFunc/OrtReadNamedBufferFunc callbacks
// for the round-trip tests below. `cipher_path` must outlive the callbacks (tests hold it as a
// string literal or a member of a longer-lived object).
struct XorCipherIo {
  uint8_t key = 0;
  const char* cipher_path = nullptr;
  std::ofstream out;  // write side only
  size_t total = 0;   // write side: bytes written; read side: bytes returned by last call
  int call_count = 0;
  std::string filename_seen;

  void OpenWrite(const char* path) {
    cipher_path = path;
    out.open(path, std::ios::binary);
  }

  static OrtStatus* WriteCb(void* state, const char* file_name, const void* buffer, size_t n) {
    auto* s = static_cast<XorCipherIo*>(state);
    s->filename_seen = file_name;
    const auto* src = static_cast<const uint8_t*>(buffer);
    std::vector<uint8_t> enc(n);
    for (size_t i = 0; i < n; ++i) enc[i] = src[i] ^ s->key;
    s->out.write(reinterpret_cast<const char*>(enc.data()), n);
    s->total += n;
    return nullptr;
  }

  static OrtStatus* ReadCb(void* state, const char* file_name, OrtAllocator* allocator,
                           void** buffer, size_t* data_size) {
    auto* s = static_cast<XorCipherIo*>(state);
    s->call_count++;
    s->filename_seen = file_name;
    std::ifstream in(s->cipher_path, std::ios::binary | std::ios::ate);
    if (!in.good()) return Ort::GetApi().CreateStatus(ORT_FAIL, "cipher file missing");
    auto sz = static_cast<size_t>(in.tellg());
    if (sz == 0) return Ort::GetApi().CreateStatus(ORT_FAIL, "cipher file empty");
    in.seekg(0);
    void* mem = allocator->Alloc(allocator, sz);
    if (mem == nullptr) return Ort::GetApi().CreateStatus(ORT_FAIL, "alloc failed");
    in.read(static_cast<char*>(mem), sz);
    if (!in) {
      allocator->Free(allocator, mem);
      return Ort::GetApi().CreateStatus(ORT_FAIL, "cipher read failed");
    }
    auto* p = static_cast<uint8_t*>(mem);
    for (size_t i = 0; i < sz; ++i) p[i] ^= s->key;
    *buffer = mem;
    *data_size = sz;
    s->total = sz;
    return nullptr;
  }
};

// Minimal QDQ + non-QDQ graph builder, shared by the tests below. Duplicated
// (rather than shared via qnn_test_utils.h) from qnn_ep_context_test.cc's
// identical helper — kept file-local on purpose since neither file needs the
// other's test-only graph shapes.
//
// input1 -------------> FusedGemm -> Add -> Q -> DQ -> output
// add1_ini_input2 -/               /
//        input2 -> Q -> DQ -> FusedGemm -> Q -> DQ -> output
static GetTestModelFn BuildGraphWithQAndNonQ(bool single_ep_node = true) {
  return [single_ep_node](ModelTestBuilder& builder) {
    // Create non-quantized FusedGemm node1
    std::vector<float> data(200 * 200, 1.0f);
    MakeTestInput(builder, "input1", TestInputDef<float>({200, 200}, false, data));
    MakeTestInput(builder, "add1_ini_input2", TestInputDef<float>({200, 200}, true, data));
    std::vector<ONNX_NAMESPACE::AttributeProto> fusedgemm_attrs;
    fusedgemm_attrs.push_back(builder.MakeStringAttribute("activation", "Relu"));
    builder.AddNode("FusedGemm_node0",
                    "FusedGemm",
                    {"input1", "add1_ini_input2"},
                    {"add1_out"},
                    kMSDomain,
                    fusedgemm_attrs);

    // Create quantized Add node2
    std::vector<float> add_data(12, 1.0f);
    gsl::span<float> data_range = gsl::make_span(add_data);
    QuantParams<uint8_t> q_parameter = GetDataQuantParams<uint8_t>(data_range);
    std::string add2_input1_qdq =
        AddQDQNodePair<uint8_t>(builder, "add2_in1_qdq", "add1_out", q_parameter.scale, q_parameter.zero_point);

    MakeTestInput(builder, "add2_input2", TestInputDef<float>({200, 200}, true, data));
    std::string add2_input2_qdq =
        AddQDQNodePair<uint8_t>(builder, "add2_in2_qdq", "add2_input2", q_parameter.scale, q_parameter.zero_point);

    builder.AddNode("Add_node0",
                    "Add",
                    {add2_input1_qdq, add2_input2_qdq},
                    {"add2_out"});

    if (single_ep_node) {
      // add2_out -> Q -> DQ -> output
      auto final_out = AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, "qdq_out", "add2_out", q_parameter.scale, q_parameter.zero_point);
    } else {
      std::string add3_input1_qdq =
          AddQDQNodePair<uint8_t>(builder, "add3_in1_qdq", "add2_out", q_parameter.scale, q_parameter.zero_point);

      MakeTestInput(builder, "add3_ini_input2", TestInputDef<float>({200, 200}, true, data));

      std::vector<ONNX_NAMESPACE::AttributeProto> fusedgemm_attrs2;
      fusedgemm_attrs2.push_back(builder.MakeStringAttribute("activation", "Relu"));
      builder.AddNode("FusedGemm_node1",
                      "FusedGemm",
                      {add3_input1_qdq, "add3_ini_input2"},
                      {"add3_out"},
                      kMSDomain,
                      fusedgemm_attrs2);

      // Create quantized Add node4
      std::string add4_input1_qdq =
          AddQDQNodePair<uint8_t>(builder, "add4_in1_qdq", "add3_out", q_parameter.scale, q_parameter.zero_point);

      MakeTestInput(builder, "add4_input2", TestInputDef<float>({200, 200}, true, data));
      std::string add4_input2_qdq =
          AddQDQNodePair<uint8_t>(builder, "add4_in2_qdq", "add4_input2", q_parameter.scale, q_parameter.zero_point);

      builder.AddNode("Add_node1",
                      "Add",
                      {add4_input1_qdq, add4_input2_qdq},
                      {"add4_out"});

      auto final_out = AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, "qdq_out", "add4_out", q_parameter.scale, q_parameter.zero_point);
    }
  };
}

struct TestModel {
  std::unique_ptr<ModelTestBuilder> builder;

  std::string Serialize() const {
    std::string model_data;
    builder->model_.SerializeToString(&model_data);
    return model_data;
  }

  bool Save(const ORTCHAR_T* path) const {
    std::ofstream ofs(path, std::ios::binary);
    return builder->model_.SerializeToOstream(&ofs);
  }
};

// Create a test model from a function that programmatically builds a graph.
// Note: We intentionally avoid onnxruntime::Model + Graph::Resolve() here. These tests only need a valid ONNX
// ModelProto for feeding into Ort::Session / Ort::Compile_extractor.
static void CreateTestModel(test::GetTestModelFn graph_builder,
                            int onnx_opset_version,
                            OrtLoggingLevel log_severity,
                            TestModel& test_model) {
  QNN_TEST_UNUSED_PARAMETER(log_severity);
  const std::unordered_map<std::string, int> domain_to_version = {{"", onnx_opset_version}, {kMSDomain, 1}};

  test_model.builder = std::make_unique<ModelTestBuilder>();
  graph_builder(*test_model.builder);

  // Populate opset imports (similar to RunQnnModelTest).
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{test_model.builder->model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }

  // Keep IR version consistent with other QNN ABI tests.
  test_model.builder->model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
}

// EmbedFalse with no callback registered — verify legacy disk-write path is undisturbed.
TEST_F(QnnHTPBackendTests, Encryption_NoCallback_EmbedFalse_FallsBackToDisk) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  const bool kV = []() {
    const char* e = std::getenv("QNN_EP_ENCRYPTION_VERBOSE");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  }();
  auto log = [&kV](const std::string& msg) {
    if (kV) std::cerr << "[Encryption_NoCallback_EmbedFalse]   " << msg << '\n';
  };

  TestModel test_model;
  CreateTestModel(BuildGraphWithQAndNonQ(false), 21,
                  OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, test_model);
  std::string model_data = test_model.Serialize();

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  provider_options["num_graph_prepare_threads"] = "1";
#endif
  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions session_options;
  RegisterQnnEpLibrary(registered_ep_device, session_options,
                       kQnnExecutionProvider, provider_options);

  constexpr const char* kQnnBinPath = "./noCallbackTest_qnn.bin";
  std::error_code pre_ec;
  std::filesystem::remove(kQnnBinPath, pre_ec);
  std::filesystem::remove("./noCallbackTest.onnx", pre_ec);

  Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
  compile_options.SetInputModelFromBuffer(model_data.data(), model_data.size());
  compile_options.SetEpContextEmbedMode(false);
  // SetOutputModelPath required when input is from buffer with no write callback —
  // ORT needs an output target for _ctx.onnx even in embed=false mode.
  compile_options.SetOutputModelPath(ORT_TSTR("./noCallbackTest.onnx"));
  compile_options.SetEpContextBinaryInformation(ORT_TSTR("./"), ORT_TSTR("noCallbackTest.onnx"));
  compile_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

  auto compile_status = Ort::CompileModel(*ort_env, compile_options);
  ASSERT_TRUE(compile_status.IsOK()) << compile_status.GetErrorMessage();

  std::error_code ec;
  ASSERT_TRUE(std::filesystem::exists(kQnnBinPath, ec))
      << "_qnn.bin missing — backward-compat path broken";
  auto sz = std::filesystem::file_size(kQnnBinPath, ec);
  log("disk: " + std::string(kQnnBinPath) + " size=" + std::to_string(sz) + " bytes");
  EXPECT_GT(sz, 0u) << "_qnn.bin is empty";

  // Spot-check first byte: QNN raw context binary header starts 0x00, NOT XOR'd.
  std::ifstream probe(kQnnBinPath, std::ios::binary);
  uint8_t b0 = 0xFF;
  probe.read(reinterpret_cast<char*>(&b0), 1);
  log("byte[0] = 0x" + [&]() {
    std::ostringstream o;
    o << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(b0);
    return o.str();
  }());
  EXPECT_EQ(b0, 0x00u) << "byte[0] != 0x00 — disk write path may be intercepted unexpectedly";

  std::filesystem::remove(kQnnBinPath, ec);
  std::filesystem::remove("./noCallbackTest.onnx", ec);
}

// End-to-end round-trip: compile → write_cb encrypts, read_cb decrypts, output matches unencrypted golden.
TEST_F(QnnHTPBackendTests, Encryption_NewReadWriteCallback_RoundTrip) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  const bool kV = []() {
    const char* e = std::getenv("QNN_EP_ENCRYPTION_VERBOSE");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  }();
  auto sec = [&kV](const char* title) {
    if (kV) std::cerr << "[Encryption_NewReadWriteCallback_RoundTrip] == " << title << " ==\n";
  };
  auto log = [&kV](const std::string& msg) {
    if (kV) std::cerr << "[Encryption_NewReadWriteCallback_RoundTrip]   " << msg << '\n';
  };

  constexpr uint8_t kKey = 0x5A;
  constexpr const char* kCipherPath = "./roundTrip_qnn_cipher.bin";
  constexpr const char* kPlaintextQnnBin = "./roundTrip_qnn.bin";
  constexpr const char* kCompiledModelPath = "./roundTrip.onnx";

  // Clean slate so Phase A's "no plaintext fallback exists" check isn't confounded.
  std::error_code ec;
  std::filesystem::remove(kCipherPath, ec);
  std::filesystem::remove(kPlaintextQnnBin, ec);
  std::filesystem::remove(kCompiledModelPath, ec);

  TestModel test_model;
  CreateTestModel(BuildGraphWithQAndNonQ(false), 21,
                  OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, test_model);
  std::string model_data = test_model.Serialize();

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  std::vector<int64_t> input_shape{200, 200};
  std::vector<float> input_data(200 * 200, 0.0f);
  Ort::MemoryInfo mem_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto run_and_capture = [&](Ort::Session& session) -> std::vector<float> {
    Ort::AllocatorWithDefaultOptions allocator;
    auto in_name = session.GetInputNameAllocated(0, allocator);
    auto out_name = session.GetOutputNameAllocated(0, allocator);
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor(mem_info, input_data.data(),
                                              input_data.size(),
                                              input_shape.data(),
                                              input_shape.size()));
    const char* in_names[] = {in_name.get()};
    const char* out_names[] = {out_name.get()};
    auto outputs = session.Run(Ort::RunOptions{}, in_names, inputs.data(), 1,
                               out_names, 1);
    auto count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    auto* data = outputs[0].GetTensorMutableData<float>();
    return std::vector<float>(data, data + count);
  };

  sec("Phase 0: baseline inference on uncompiled model (golden)");
  std::vector<float> golden;
  {
    RegisteredEpDeviceUniquePtr registered_ep_device;
    Ort::SessionOptions session_options;
    RegisterQnnEpLibrary(registered_ep_device, session_options,
                         kQnnExecutionProvider, provider_options);
    Ort::Session session(*ort_env, model_data.data(), model_data.size(),
                         session_options);
    golden = run_and_capture(session);
  }
  ASSERT_GT(golden.size(), 0u) << "phase 0 baseline produced empty output";
  log("phase 0 golden output size: " + std::to_string(golden.size()));

  sec("Phase A: compile with new write callback");

  XorCipherIo ws;
  ws.key = kKey;
  ws.OpenWrite(kCipherPath);
  ASSERT_TRUE(ws.out.is_open());

  {
    RegisteredEpDeviceUniquePtr registered_ep_device;
    Ort::SessionOptions session_options;
    RegisterQnnEpLibrary(registered_ep_device, session_options,
                         kQnnExecutionProvider, provider_options);

    try {
      Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
      compile_options.SetInputModelFromBuffer(model_data.data(), model_data.size());
      compile_options.SetEpContextEmbedMode(false);
      // Phase B opens this exact path; pin the location so the read side has a target.
      compile_options.SetOutputModelPath(ORT_TSTR("./roundTrip.onnx"));
      compile_options.SetEpContextBinaryInformation(ORT_TSTR("./"), ORT_TSTR("roundTrip.onnx"));
      compile_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

      auto* set_fn = Ort::Experimental::Get_OrtCompileApi_ModelCompilationOptions_SetEpContextDataWriteFunc_SinceV28_Fn(
          &Ort::GetApi());
      ASSERT_NE(set_fn, nullptr) << "SetEpContextDataWriteFunc_SinceV28 resolver not bound";
      auto* st = set_fn(compile_options, XorCipherIo::WriteCb, &ws);
      if (st != nullptr) Ort::GetApi().ReleaseStatus(st);

      auto cs = Ort::CompileModel(*ort_env, compile_options);
      ASSERT_TRUE(cs.IsOK()) << "compile failed: " << cs.GetErrorMessage();
      log("compile: OK");
    } catch (const std::exception& e) {
      ws.out.close();
      FAIL() << "phase A exception: " << e.what();
    }
  }
  ws.out.close();

  ASSERT_GT(ws.total, 0u) << "write callback was never invoked — encryption path not exercised";
  ASSERT_TRUE(std::filesystem::exists(kCipherPath))
      << "App-encrypted artifact " << kCipherPath << " was not produced";
  ASSERT_GT(std::filesystem::file_size(kCipherPath), 0u);
  // Critical: callback must REPLACE the EP's disk write, not run alongside.
  // If a plaintext _qnn.bin landed on disk at the same time the callback fired,
  // the encryption hook is misbehaving (additive instead of replacement).
  ASSERT_FALSE(std::filesystem::exists(kPlaintextQnnBin))
      << "EP also produced a plaintext " << kPlaintextQnnBin
      << "; the write callback was meant to replace, not augment, the disk write";
  log("phase A wrote " + std::to_string(ws.total) + " bytes (encrypted) to " + kCipherPath);

  sec("Phase B: open session with new read callback");

  XorCipherIo rs;
  rs.key = kKey;
  rs.cipher_path = kCipherPath;

  std::vector<float> phase_b_output;
  {
    RegisteredEpDeviceUniquePtr registered_ep_device;
    Ort::SessionOptions session_options;
    RegisterQnnEpLibrary(registered_ep_device, session_options,
                         kQnnExecutionProvider, provider_options);

    try {
      auto* set_fn = Ort::Experimental::Get_OrtApi_SessionOptions_SetEpContextDataReadFunc_SinceV28_Fn(
          &Ort::GetApi());
      ASSERT_NE(set_fn, nullptr) << "SetEpContextDataReadFunc_SinceV28 resolver not bound";
      auto* st = set_fn(session_options, XorCipherIo::ReadCb, &rs);
      if (st != nullptr) {
        Ort::GetApi().ReleaseStatus(st);
        FAIL() << "SessionOptions_SetEpContextDataReadFunc returned non-OK status";
      }

      Ort::Session session(*ort_env, ORT_TSTR("./roundTrip.onnx"), session_options);
      log("session created OK");
      phase_b_output = run_and_capture(session);
    } catch (const std::exception& e) {
      FAIL() << "phase B exception: " << e.what();
    }
  }

  ASSERT_GT(rs.call_count, 0) << "read callback was never invoked — decryption path not exercised";
  ASSERT_GT(rs.total, 0u) << "read callback returned zero bytes";
  log("phase B read callback fired " + std::to_string(rs.call_count) +
      " time(s); returned " + std::to_string(rs.total) +
      " bytes; "
      "filename: " +
      (rs.filename_seen.empty() ? "<never seen>" : rs.filename_seen));

  // Mismatch here = encrypt/decrypt round-trip corrupted the context binary.
  ASSERT_EQ(phase_b_output.size(), golden.size())
      << "phase B output size differs from phase 0 golden (model IO drift)";
  for (size_t i = 0; i < phase_b_output.size(); ++i) {
    EXPECT_NEAR(phase_b_output[i], golden[i], 1e-3f)
        << "phase B output differs from phase 0 golden at index " << i
        << " (golden=" << golden[i] << ", phase B=" << phase_b_output[i] << ")";
  }
  log("phase B output matches phase 0 golden");

  std::filesystem::remove(kCipherPath, ec);
  std::filesystem::remove(kCompiledModelPath, ec);
  // Callback must REPLACE (not accompany) the disk write — remove any regression stragglers.
  std::filesystem::remove(kPlaintextQnnBin, ec);
}

// Baseline: htp_share_resource_optimization=1 WITHOUT encryption; hangs here → pre-existing QAIRT issue, unrelated to this PR.
TEST_F(QnnHTPBackendTests, Encryption_VtcmSharing_Baseline_NoCallback) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  {
    TestModel test_model;
    CreateTestModel(BuildGraphWithQAndNonQ(false), 21,
                    OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, test_model);
    std::string model_data = test_model.Serialize();

    ProviderOptions provider_options;
    provider_options["backend_type"] = "htp";
    provider_options["offload_graph_io_quantization"] = "0";
#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
    provider_options["num_graph_prepare_threads"] = "1";
#endif
    RegisteredEpDeviceUniquePtr registered_ep_device;
    Ort::SessionOptions session_options;
    RegisterQnnEpLibrary(registered_ep_device, session_options,
                         kQnnExecutionProvider, provider_options);

    try {
      Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
      compile_options.SetInputModelFromBuffer(model_data.data(), model_data.size());
      compile_options.SetEpContextEmbedMode(false);
      compile_options.SetOutputModelPath(ORT_TSTR("./vtcm_baseline.onnx"));
      compile_options.SetEpContextBinaryInformation(ORT_TSTR("./"), ORT_TSTR("vtcm_baseline.onnx"));
      compile_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
      auto cs = Ort::CompileModel(*ort_env, compile_options);
      ASSERT_TRUE(cs.IsOK()) << "Compile failed: " << cs.GetErrorMessage();
    } catch (const std::exception& e) {
      FAIL() << "Phase A exception: " << e.what();
    }
  }

  {
    ProviderOptions provider_options;
    provider_options["backend_type"] = "htp";
    provider_options["offload_graph_io_quantization"] = "0";
#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
    provider_options["num_graph_prepare_threads"] = "1";
#endif
    RegisteredEpDeviceUniquePtr registered_ep_device;
    Ort::SessionOptions session_options;
    RegisterQnnEpLibrary(registered_ep_device, session_options,
                         kQnnExecutionProvider, provider_options);

    session_options.AddConfigEntry("ep.qnnexecutionprovider.htp_share_resource_optimization", "1");

    try {
      Ort::Session session(*ort_env, ORT_TSTR("./vtcm_baseline.onnx"), session_options);

      Ort::AllocatorWithDefaultOptions allocator;
      auto in_name = session.GetInputNameAllocated(0, allocator);
      auto out_name = session.GetOutputNameAllocated(0, allocator);
      std::vector<int64_t> shape{200, 200};
      std::vector<float> input(200 * 200, 0.0f);
      Ort::MemoryInfo mem_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
      std::vector<Ort::Value> inputs;
      inputs.push_back(Ort::Value::CreateTensor(mem_info, input.data(), input.size(),
                                                shape.data(), shape.size()));
      const char* in_names[] = {in_name.get()};
      const char* out_names[] = {out_name.get()};
      auto outputs = session.Run(Ort::RunOptions{}, in_names, inputs.data(), 1, out_names, 1);
      ASSERT_EQ(outputs.size(), 1u);
    } catch (const std::exception& e) {
      FAIL() << "Phase B exception: " << e.what();
    }
  }

  std::error_code ec;
  std::filesystem::remove(ORT_TSTR("./vtcm_baseline.onnx"), ec);
  std::filesystem::remove(ORT_TSTR("./vtcm_baseline_qnn.bin"), ec);
}

// 2-session shared-context E2E. Session 1 heap-leaked (workaround for a pre-existing
// QAIRT 2.45 teardown hang, see Encryption_VtcmSharing_Baseline_NoCallback).
TEST_F(QnnHTPBackendTests, Encryption_VtcmSharing_MultiSession_EndToEnd) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  constexpr uint8_t kKey = 0x5A;
  constexpr const char* kCipherPath = "./vtcm_multi_qnn_cipher.bin";

  {
    TestModel test_model;
    CreateTestModel(BuildGraphWithQAndNonQ(false), 21,
                    OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, test_model);
    std::string model_data = test_model.Serialize();

    ProviderOptions provider_options;
    provider_options["backend_type"] = "htp";
    provider_options["offload_graph_io_quantization"] = "0";
#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
    provider_options["num_graph_prepare_threads"] = "1";
#endif
    RegisteredEpDeviceUniquePtr registered_ep_device;
    Ort::SessionOptions session_options;
    RegisterQnnEpLibrary(registered_ep_device, session_options,
                         kQnnExecutionProvider, provider_options);

    XorCipherIo ws;
    ws.key = kKey;
    ws.OpenWrite(kCipherPath);
    ASSERT_TRUE(ws.out.is_open());

    try {
      Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
      compile_options.SetInputModelFromBuffer(model_data.data(), model_data.size());
      compile_options.SetEpContextEmbedMode(false);
      compile_options.SetOutputModelPath(ORT_TSTR("./vtcm_multi.onnx"));
      compile_options.SetEpContextBinaryInformation(ORT_TSTR("./"), ORT_TSTR("vtcm_multi.onnx"));
      compile_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
      if (auto* set_fn = Ort::Experimental::Get_OrtCompileApi_ModelCompilationOptions_SetEpContextDataWriteFunc_SinceV28_Fn(
              &Ort::GetApi())) {
        auto* st = set_fn(compile_options, XorCipherIo::WriteCb, &ws);
        if (st != nullptr) Ort::GetApi().ReleaseStatus(st);
      }
      auto cs = Ort::CompileModel(*ort_env, compile_options);
      ASSERT_TRUE(cs.IsOK()) << "Compile failed: " << cs.GetErrorMessage();
    } catch (const std::exception& e) {
      FAIL() << "Phase A exception: " << e.what();
    }
    ws.out.close();
    ASSERT_TRUE(std::filesystem::exists(kCipherPath));
  }

  auto make_provider_options = []() -> ProviderOptions {
    ProviderOptions opts;
    opts["backend_type"] = "htp";
    opts["offload_graph_io_quantization"] = "0";
#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
    opts["num_graph_prepare_threads"] = "1";
#endif
    return opts;
  };

  // Session 1 resources at test scope so SharedContext survives into Session 2.
  XorCipherIo rs;
  rs.key = kKey;
  rs.cipher_path = kCipherPath;

  auto s1_provider_options = make_provider_options();
  RegisteredEpDeviceUniquePtr s1_registered_ep_device;
  Ort::SessionOptions s1_session_options;
  RegisterQnnEpLibrary(s1_registered_ep_device, s1_session_options,
                       kQnnExecutionProvider, s1_provider_options);
  s1_session_options.AddConfigEntry("ep.qnnexecutionprovider.htp_share_resource_optimization", "1");
  {
    if (auto* set_fn = Ort::Experimental::Get_OrtApi_SessionOptions_SetEpContextDataReadFunc_SinceV28_Fn(
            &Ort::GetApi())) {
      auto* st = set_fn(s1_session_options, XorCipherIo::ReadCb, &rs);
      if (st != nullptr) Ort::GetApi().ReleaseStatus(st);
    }
  }

  try {
    // Heap-allocate to avoid single-session VTCM async teardown hang (QAIRT 2.45 issue).
    // The leak is bounded to this test on HTP device (V69+); the ASan/LSan CI job added by
    // PR #419 runs on Linux x86_64 where this test is gtest-skipped, so no CI signal is lost.
    auto* s1 = new Ort::Session(*ort_env, ORT_TSTR("./vtcm_multi.onnx"), s1_session_options);
    (void)s1;
  } catch (const std::exception& e) {
    FAIL() << "Session 1 exception: " << e.what();
  }
  ASSERT_GE(rs.call_count, 1) << "read callback did not fire during Session 1 load";

  // Session 2 reuses the cached context; only the Phase 1 read callback fires (not the multi-binary hook).
  {
    int s2_call_count_start = rs.call_count;

    auto provider_options = make_provider_options();
    Ort::SessionOptions session_options;
    session_options.AppendExecutionProvider_V2(
        *ort_env,
        {Ort::ConstEpDevice(s1_registered_ep_device.get())},
        provider_options);
    session_options.AddConfigEntry("ep.qnnexecutionprovider.htp_share_resource_optimization", "1");
    {
      if (auto* set_fn = Ort::Experimental::Get_OrtApi_SessionOptions_SetEpContextDataReadFunc_SinceV28_Fn(
              &Ort::GetApi())) {
        auto* st = set_fn(session_options, XorCipherIo::ReadCb, &rs);
        if (st != nullptr) Ort::GetApi().ReleaseStatus(st);
      }
    }

    try {
      Ort::Session session(*ort_env, ORT_TSTR("./vtcm_multi.onnx"), session_options);

      EXPECT_GT(rs.call_count, s2_call_count_start)
          << "Phase 1 callback should fire for Session 2 CompileContextModel";

      Ort::AllocatorWithDefaultOptions allocator;
      auto in_name = session.GetInputNameAllocated(0, allocator);
      auto out_name = session.GetOutputNameAllocated(0, allocator);
      std::vector<int64_t> shape{200, 200};
      std::vector<float> input(200 * 200, 0.0f);
      Ort::MemoryInfo mem_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
      std::vector<Ort::Value> inputs;
      inputs.push_back(Ort::Value::CreateTensor(mem_info, input.data(), input.size(),
                                                shape.data(), shape.size()));
      const char* in_names[] = {in_name.get()};
      const char* out_names[] = {out_name.get()};
      auto outputs = session.Run(Ort::RunOptions{}, in_names, inputs.data(), 1, out_names, 1);
      ASSERT_EQ(outputs.size(), 1u)
          << "Session 2 inference failed — read callback may have produced invalid decrypted binary";
    } catch (const std::exception& e) {
      FAIL() << "Session 2 exception: " << e.what();
    }
  }

  std::error_code ec;
  std::filesystem::remove(kCipherPath, ec);
  std::filesystem::remove(ORT_TSTR("./vtcm_multi.onnx"), ec);
}

// A read callback returning ORT_FAIL must surface as an exception, not a crash or silent success.
TEST_F(QnnHTPBackendTests, Encryption_ReadCallback_ReturnsError_SessionCtorSurfacesFailure) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  constexpr uint8_t kKey = 0x5A;
  constexpr const char* kCipherPath = "./read_cb_err_cipher.bin";
  static constexpr const char* kErrorSentinel = "read_cb_injected_failure_sentinel_ort_fail_42";

  // Phase A produces a valid cipher so Phase B's failure is unambiguously from the injected error.
  {
    TestModel test_model;
    CreateTestModel(BuildGraphWithQAndNonQ(false), 21,
                    OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, test_model);
    std::string model_data = test_model.Serialize();

    ProviderOptions provider_options;
    provider_options["backend_type"] = "htp";
    provider_options["offload_graph_io_quantization"] = "0";
#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
    provider_options["num_graph_prepare_threads"] = "1";
#endif
    RegisteredEpDeviceUniquePtr registered_ep_device;
    Ort::SessionOptions session_options;
    RegisterQnnEpLibrary(registered_ep_device, session_options,
                         kQnnExecutionProvider, provider_options);

    XorCipherIo ws;
    ws.key = kKey;
    ws.OpenWrite(kCipherPath);
    ASSERT_TRUE(ws.out.is_open());

    Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
    compile_options.SetInputModelFromBuffer(model_data.data(), model_data.size());
    compile_options.SetEpContextEmbedMode(false);
    compile_options.SetOutputModelPath(ORT_TSTR("./read_cb_err.onnx"));
    compile_options.SetEpContextBinaryInformation(ORT_TSTR("./"), ORT_TSTR("read_cb_err.onnx"));
    compile_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

    auto* set_fn = Ort::Experimental::Get_OrtCompileApi_ModelCompilationOptions_SetEpContextDataWriteFunc_SinceV28_Fn(
        &Ort::GetApi());
    ASSERT_NE(set_fn, nullptr);
    auto* st = set_fn(compile_options, XorCipherIo::WriteCb, &ws);
    if (st != nullptr) Ort::GetApi().ReleaseStatus(st);

    auto cs = Ort::CompileModel(*ort_env, compile_options);
    ASSERT_TRUE(cs.IsOK()) << "Phase A compile failed: " << cs.GetErrorMessage();
    ws.out.close();
    ASSERT_TRUE(std::filesystem::exists(kCipherPath));
  }

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  provider_options["num_graph_prepare_threads"] = "1";
#endif
  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions session_options;
  RegisterQnnEpLibrary(registered_ep_device, session_options,
                       kQnnExecutionProvider, provider_options);

  int call_count = 0;
  auto err_read_cb = [](void* state, const char* /*file_name*/, OrtAllocator* /*allocator*/,
                        void** /*buffer*/, size_t* /*data_size*/) -> OrtStatus* {
    auto* count = static_cast<int*>(state);
    ++(*count);
    return Ort::GetApi().CreateStatus(ORT_FAIL, kErrorSentinel);
  };

  auto* set_fn = Ort::Experimental::Get_OrtApi_SessionOptions_SetEpContextDataReadFunc_SinceV28_Fn(
      &Ort::GetApi());
  ASSERT_NE(set_fn, nullptr);
  auto* st = set_fn(session_options, err_read_cb, &call_count);
  if (st != nullptr) {
    Ort::GetApi().ReleaseStatus(st);
    FAIL() << "SetEpContextDataReadFunc returned non-OK setup status";
  }

  bool threw = false;
  std::string message;
  try {
    Ort::Session session(*ort_env, ORT_TSTR("./read_cb_err.onnx"), session_options);
    (void)session;
  } catch (const Ort::Exception& e) {
    threw = true;
    message = e.what();
  } catch (const std::exception& e) {
    threw = true;
    message = e.what();
  }

  EXPECT_TRUE(threw) << "Session ctor did not throw when read callback returned ORT_FAIL";
  EXPECT_GT(call_count, 0) << "read callback was never invoked";
  // ORT may prefix the surfaced message with additional context, but the marker should survive.
  EXPECT_NE(message.find(kErrorSentinel), std::string::npos)
      << "sentinel error text was not surfaced to caller (got: " << message << ")";

  std::error_code ec;
  std::filesystem::remove(kCipherPath, ec);
  std::filesystem::remove(ORT_TSTR("./read_cb_err.onnx"), ec);
}

// Encryption and weight sharing are orthogonal: share_ep_contexts controls whether the
// QnnBackendManager (and its context binary) is shared across sessions, while the
// read/write callbacks only intercept the bytes on their way to/from disk. This test
// proves the two coexist — a single session that turns on share_ep_contexts AND registers
// an encrypting write callback still compiles to an encrypted binary (no plaintext on disk),
// and the decrypting read callback loads it back and runs. It does NOT exercise the
// multi-model merged-binary flow (that path compiles via Ort::Session create, which cannot
// carry a write callback in ORT 1.28); it isolates the interaction of the two feature flags.
TEST_F(QnnHTPBackendTests, Encryption_WithShareEpContexts_RoundTrip) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
#if (defined(__aarch64__) || defined(_M_ARM64)) && \
    !(QNN_API_VERSION_MAJOR > 2 || (QNN_API_VERSION_MAJOR == 2 && QNN_API_VERSION_MINOR >= 34))
  GTEST_SKIP() << "HTP weight sharing on ARM64 requires QNN API version >= 2.34.";
#elif defined(__ANDROID__)
  GTEST_SKIP() << "Weight sharing on Android devices is disabled";
#endif

  const bool kV = []() {
    const char* e = std::getenv("QNN_EP_ENCRYPTION_VERBOSE");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  }();
  auto log = [&kV](const std::string& msg) {
    if (kV) std::cerr << "[Encryption_WithShareEpContexts]   " << msg << '\n';
  };

  constexpr uint8_t kKey = 0x5A;
  constexpr const char* kCipherPath = "./shareEpCtx_qnn_cipher.bin";
  constexpr const char* kPlaintextQnnBin = "./shareEpCtx_qnn.bin";
  constexpr const char* kCompiledModelPath = "./shareEpCtx.onnx";

  std::error_code ec;
  std::filesystem::remove(kCipherPath, ec);
  std::filesystem::remove(kPlaintextQnnBin, ec);
  std::filesystem::remove(kCompiledModelPath, ec);

  TestModel test_model;
  CreateTestModel(BuildGraphWithQAndNonQ(false), 21,
                  OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, test_model);
  std::string model_data = test_model.Serialize();

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  std::vector<int64_t> input_shape{200, 200};
  std::vector<float> input_data(200 * 200, 0.0f);
  Ort::MemoryInfo mem_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto run_and_capture = [&](Ort::Session& session) -> std::vector<float> {
    Ort::AllocatorWithDefaultOptions allocator;
    auto in_name = session.GetInputNameAllocated(0, allocator);
    auto out_name = session.GetOutputNameAllocated(0, allocator);
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor(mem_info, input_data.data(),
                                              input_data.size(),
                                              input_shape.data(),
                                              input_shape.size()));
    const char* in_names[] = {in_name.get()};
    const char* out_names[] = {out_name.get()};
    auto outputs = session.Run(Ort::RunOptions{}, in_names, inputs.data(), 1,
                               out_names, 1);
    auto count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    auto* data = outputs[0].GetTensorMutableData<float>();
    return std::vector<float>(data, data + count);
  };

  // Golden: baseline inference on the uncompiled model (no share_ep_contexts, no callbacks).
  std::vector<float> golden;
  {
    RegisteredEpDeviceUniquePtr registered_ep_device;
    Ort::SessionOptions session_options;
    RegisterQnnEpLibrary(registered_ep_device, session_options,
                         kQnnExecutionProvider, provider_options);
    Ort::Session session(*ort_env, model_data.data(), model_data.size(),
                         session_options);
    golden = run_and_capture(session);
  }
  ASSERT_GT(golden.size(), 0u) << "baseline produced empty output";

  // Phase A: compile with share_ep_contexts=1 AND an encrypting write callback.
  XorCipherIo ws;
  ws.key = kKey;
  ws.OpenWrite(kCipherPath);
  ASSERT_TRUE(ws.out.is_open());

  {
    RegisteredEpDeviceUniquePtr registered_ep_device;
    Ort::SessionOptions session_options;
    RegisterQnnEpLibrary(registered_ep_device, session_options,
                         kQnnExecutionProvider, provider_options);
    // Single-model weight sharing: enable the flag and immediately stop it so this one
    // compile is both the first and last sharing session (a single merged context).
    session_options.AddConfigEntry(kOrtSessionOptionShareEpContexts, "1");
    session_options.AddConfigEntry(kOrtSessionOptionStopShareEpContexts, "1");

    try {
      Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
      compile_options.SetInputModelFromBuffer(model_data.data(), model_data.size());
      compile_options.SetEpContextEmbedMode(false);
      compile_options.SetOutputModelPath(ORT_TSTR("./shareEpCtx.onnx"));
      compile_options.SetEpContextBinaryInformation(ORT_TSTR("./"), ORT_TSTR("shareEpCtx.onnx"));
      compile_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

      auto* set_fn = Ort::Experimental::Get_OrtCompileApi_ModelCompilationOptions_SetEpContextDataWriteFunc_SinceV28_Fn(
          &Ort::GetApi());
      ASSERT_NE(set_fn, nullptr) << "SetEpContextDataWriteFunc_SinceV28 resolver not bound";
      auto* st = set_fn(compile_options, XorCipherIo::WriteCb, &ws);
      if (st != nullptr) Ort::GetApi().ReleaseStatus(st);

      auto cs = Ort::CompileModel(*ort_env, compile_options);
      ASSERT_TRUE(cs.IsOK()) << "compile failed: " << cs.GetErrorMessage();
    } catch (const std::exception& e) {
      ws.out.close();
      FAIL() << "phase A exception: " << e.what();
    }
  }
  ws.out.close();

  // The write callback must still fire with share_ep_contexts on, and it must still
  // REPLACE the disk write — no plaintext binary alongside the ciphertext.
  ASSERT_GT(ws.total, 0u) << "write callback did not fire with share_ep_contexts enabled";
  ASSERT_TRUE(std::filesystem::exists(kCipherPath));
  ASSERT_GT(std::filesystem::file_size(kCipherPath), 0u);
  ASSERT_FALSE(std::filesystem::exists(kPlaintextQnnBin))
      << "plaintext " << kPlaintextQnnBin << " leaked to disk alongside the encrypted binary";
  log("phase A wrote " + std::to_string(ws.total) + " encrypted bytes with share_ep_contexts on");

  // Phase B: load the compiled model with the decrypting read callback, share_ep_contexts still on.
  XorCipherIo rs;
  rs.key = kKey;
  rs.cipher_path = kCipherPath;

  std::vector<float> phase_b_output;
  {
    RegisteredEpDeviceUniquePtr registered_ep_device;
    Ort::SessionOptions session_options;
    RegisterQnnEpLibrary(registered_ep_device, session_options,
                         kQnnExecutionProvider, provider_options);
    session_options.AddConfigEntry(kOrtSessionOptionShareEpContexts, "1");

    try {
      auto* set_fn = Ort::Experimental::Get_OrtApi_SessionOptions_SetEpContextDataReadFunc_SinceV28_Fn(
          &Ort::GetApi());
      ASSERT_NE(set_fn, nullptr) << "SetEpContextDataReadFunc_SinceV28 resolver not bound";
      auto* st = set_fn(session_options, XorCipherIo::ReadCb, &rs);
      if (st != nullptr) {
        Ort::GetApi().ReleaseStatus(st);
        FAIL() << "SessionOptions_SetEpContextDataReadFunc returned non-OK";
      }

      Ort::Session session(*ort_env, ORT_TSTR("./shareEpCtx.onnx"), session_options);
      phase_b_output = run_and_capture(session);
    } catch (const std::exception& e) {
      FAIL() << "phase B exception: " << e.what();
    }
  }

  ASSERT_GT(rs.call_count, 0) << "read callback did not fire with share_ep_contexts enabled";
  ASSERT_EQ(phase_b_output.size(), golden.size())
      << "output size differs from golden (share_ep_contexts + encryption)";
  for (size_t i = 0; i < phase_b_output.size(); ++i) {
    EXPECT_NEAR(phase_b_output[i], golden[i], 1e-3f)
        << "output differs from golden at index " << i
        << " (golden=" << golden[i] << ", got=" << phase_b_output[i] << ")";
  }
  log("phase B decrypted + ran with share_ep_contexts on; output matches golden");

  std::filesystem::remove(kCipherPath, ec);
  std::filesystem::remove(kCompiledModelPath, ec);
  std::filesystem::remove(kPlaintextQnnBin, ec);
}

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))

// Minimal float Add model — used only to exercise the GPU backend, which does
// not support QDQ (quantized) ops. Duplicated from qnn_ep_context_test.cc's
// identical helper for the same file-locality reason as BuildGraphWithQAndNonQ.
static void CreateFloatModel(const std::string& model_file_name) {
  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}};

  ModelTestBuilder helper;
  std::vector<float> data = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
  MakeTestInput(helper, "add_in1", TestInputDef<float>({2, 3}, false, data));
  MakeTestInput(helper, "add_in2", TestInputDef<float>({2, 3}, true, data));
  helper.AddNode("Add_node", "Add", {"add_in1", "add_in2"}, {"add_out"});
  helper.MakeOutput<float>("add_out", std::vector<int64_t>{2, 3});

  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  const std::wstring model_file_name_w(model_file_name.begin(), model_file_name.end());
  std::ofstream model_ofs(model_file_name_w, std::ios::binary);

  ASSERT_TRUE(model_ofs.good());
  ASSERT_TRUE(helper.model_.SerializeToOstream(&model_ofs));
  model_ofs.close();
}

// Encryption round-trip on the QNN GPU backend. Mirrors the HTP RoundTrip test
// but with backend_type=gpu — proves the encryption hooks are backend-agnostic.
TEST_F(QnnGPUBackendTests, Encryption_NewReadWriteCallback_RoundTrip_Gpu) {
  constexpr uint8_t kKey = 0xA5;
  constexpr const char* kInputModelPath = "./roundTripGpu_input.onnx";
  constexpr const char* kCipherPath = "./roundTripGpu_qnn_cipher.bin";
  constexpr const char* kPlaintextQnnBin = "./roundTripGpu_qnn.bin";
  constexpr const char* kCompiledModelPath = "./roundTripGpu.onnx";

  std::error_code ec;
  std::filesystem::remove(kInputModelPath, ec);
  std::filesystem::remove(kCipherPath, ec);
  std::filesystem::remove(kPlaintextQnnBin, ec);
  std::filesystem::remove(kCompiledModelPath, ec);

  CreateFloatModel(kInputModelPath);
  ASSERT_TRUE(std::filesystem::exists(kInputModelPath));

  ProviderOptions provider_options;
  provider_options["backend_type"] = "gpu";

  // Phase 0: golden inference on uncompiled model.
  std::vector<int64_t> input_shape{2, 3};
  std::vector<float> input_data(6, 0.0f);
  Ort::MemoryInfo mem_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto run_and_capture = [&](Ort::Session& session) -> std::vector<float> {
    Ort::AllocatorWithDefaultOptions allocator;
    auto in_name = session.GetInputNameAllocated(0, allocator);
    auto out_name = session.GetOutputNameAllocated(0, allocator);
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor(mem_info, input_data.data(),
                                              input_data.size(),
                                              input_shape.data(),
                                              input_shape.size()));
    const char* in_names[] = {in_name.get()};
    const char* out_names[] = {out_name.get()};
    auto outputs = session.Run(Ort::RunOptions{}, in_names, inputs.data(), 1,
                               out_names, 1);
    auto count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    auto* data = outputs[0].GetTensorMutableData<float>();
    return std::vector<float>(data, data + count);
  };

  std::vector<float> golden;
  {
    RegisteredEpDeviceUniquePtr registered_ep_device;
    Ort::SessionOptions session_options;
    RegisterQnnEpLibrary(registered_ep_device, session_options,
                         kQnnExecutionProvider, provider_options);
    Ort::Session session(*ort_env, ORT_TSTR("./roundTripGpu_input.onnx"), session_options);
    golden = run_and_capture(session);
  }
  ASSERT_GT(golden.size(), 0u) << "phase 0 baseline produced empty output";

  // Phase A: compile with encrypting write callback.
  XorCipherIo ws;
  ws.key = kKey;
  ws.OpenWrite(kCipherPath);
  ASSERT_TRUE(ws.out.is_open());

  {
    RegisteredEpDeviceUniquePtr registered_ep_device;
    Ort::SessionOptions session_options;
    RegisterQnnEpLibrary(registered_ep_device, session_options,
                         kQnnExecutionProvider, provider_options);
    try {
      Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
      compile_options.SetInputModelPath(ORT_TSTR("./roundTripGpu_input.onnx"));
      compile_options.SetEpContextEmbedMode(false);
      compile_options.SetOutputModelPath(ORT_TSTR("./roundTripGpu.onnx"));
      compile_options.SetEpContextBinaryInformation(ORT_TSTR("./"), ORT_TSTR("roundTripGpu.onnx"));
      compile_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

      auto* set_fn = Ort::Experimental::Get_OrtCompileApi_ModelCompilationOptions_SetEpContextDataWriteFunc_SinceV28_Fn(
          &Ort::GetApi());
      ASSERT_NE(set_fn, nullptr) << "SetEpContextDataWriteFunc_SinceV28 resolver not bound";
      auto* st = set_fn(compile_options, XorCipherIo::WriteCb, &ws);
      if (st != nullptr) Ort::GetApi().ReleaseStatus(st);

      auto cs = Ort::CompileModel(*ort_env, compile_options);
      ASSERT_TRUE(cs.IsOK()) << "GPU compile failed: " << cs.GetErrorMessage();
    } catch (const std::exception& e) {
      ws.out.close();
      FAIL() << "phase A (GPU) exception: " << e.what();
    }
  }
  ws.out.close();

  ASSERT_GT(ws.total, 0u) << "write callback was never invoked on GPU backend";
  ASSERT_TRUE(std::filesystem::exists(kCipherPath));
  ASSERT_GT(std::filesystem::file_size(kCipherPath), 0u);
  // Callback must REPLACE the disk write, not augment it — same contract as HTP.
  ASSERT_FALSE(std::filesystem::exists(kPlaintextQnnBin))
      << "GPU backend also produced plaintext " << kPlaintextQnnBin;

  // Phase B: load with decrypting read callback and confirm output matches golden.
  XorCipherIo rs;
  rs.key = kKey;
  rs.cipher_path = kCipherPath;

  std::vector<float> phase_b_output;
  {
    RegisteredEpDeviceUniquePtr registered_ep_device;
    Ort::SessionOptions session_options;
    RegisterQnnEpLibrary(registered_ep_device, session_options,
                         kQnnExecutionProvider, provider_options);
    try {
      auto* set_fn = Ort::Experimental::Get_OrtApi_SessionOptions_SetEpContextDataReadFunc_SinceV28_Fn(
          &Ort::GetApi());
      ASSERT_NE(set_fn, nullptr) << "SetEpContextDataReadFunc_SinceV28 resolver not bound";
      auto* st = set_fn(session_options, XorCipherIo::ReadCb, &rs);
      if (st != nullptr) {
        Ort::GetApi().ReleaseStatus(st);
        FAIL() << "SessionOptions_SetEpContextDataReadFunc returned non-OK";
      }
      Ort::Session session(*ort_env, ORT_TSTR("./roundTripGpu.onnx"), session_options);
      phase_b_output = run_and_capture(session);
    } catch (const std::exception& e) {
      FAIL() << "phase B (GPU) exception: " << e.what();
    }
  }

  ASSERT_GT(rs.call_count, 0) << "read callback was never invoked on GPU backend";
  ASSERT_GT(rs.total, 0u) << "read callback returned zero bytes";
  ASSERT_EQ(phase_b_output.size(), golden.size())
      << "phase B (GPU) output size differs from golden";
  for (size_t i = 0; i < phase_b_output.size(); ++i) {
    EXPECT_NEAR(phase_b_output[i], golden[i], 1e-3f)
        << "phase B (GPU) output differs from golden at index " << i;
  }

  std::filesystem::remove(kInputModelPath, ec);
  std::filesystem::remove(kCipherPath, ec);
  std::filesystem::remove(kCompiledModelPath, ec);
  std::filesystem::remove(kPlaintextQnnBin, ec);
}

#endif  // defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))

#endif  // ORT_API_VERSION >= 28

}  // namespace test
}  // namespace onnxruntime
