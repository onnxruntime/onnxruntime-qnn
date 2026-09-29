// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// run_app — "Run" stage of the compiled-model-encryption example.
//
// Loads a compiled EPContext model (from prepare_app) and registers a read
// callback (ORT 1.28 EPContext API) that decrypts the cipher file and hands
// the plaintext back to the QNN EP, which runs inference. ARM64 only.
//
// This is an EXAMPLE: ReadCb below just XORs the bytes, to keep the callback
// mechanism obvious. Replace ReadCb with your own decryption (matching
// whatever WriteCb in prepare_app.cc used) to use this for real.
//
// Usage:
//   run_app <ctx_model.onnx> <cipher.bin> [xor_key_hex] [input.raw answer_run.raw]
//
//   ctx_model.onnx     the EPContext wrapper model prepare_app wrote
//   cipher.bin         the encrypted _qnn.bin prepare_app produced
//   xor_key_hex        optional 1-byte key in hex (default 5a); must match prepare
//   input.raw          optional float32 input (same one passed to prepare_app)
//   answer_run.raw     where to write the decrypted-model output (float32)
//
// This app only verifies the decrypt+load+run path; it does not check the
// output against prepare_app's answer. Use compare_answers.py for that.
//
// Exit 0 on success.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "enc_common.h"

#if ORT_API_VERSION >= 28

namespace {

struct ReadState {
  uint8_t key = encapp::kDefaultKey;
  std::string cipher_path;
  int call_count = 0;
  size_t bytes_returned = 0;
};

// OrtReadNamedBufferFunc: (state, file_name, allocator, out_buf, out_size).
// Example cipher: read the ciphertext, XOR-decrypt into an allocator-owned
// buffer, and return it. Replace this body with your own decryption.
OrtStatus* ReadCb(void* state, const char* file_name, OrtAllocator* allocator,
                  void** buffer, size_t* data_size) noexcept {
  auto* s = static_cast<ReadState*>(state);
  s->call_count++;
  const OrtApi& c_api = Ort::GetApi();

  std::ifstream in(s->cipher_path, std::ios::binary | std::ios::ate);
  if (!in.good()) return c_api.CreateStatus(ORT_FAIL, "run_app: cipher file missing");
  const auto sz = static_cast<size_t>(in.tellg());
  if (sz == 0) return c_api.CreateStatus(ORT_FAIL, "run_app: cipher file empty");
  in.seekg(0);

  void* mem = allocator->Alloc(allocator, sz);
  if (mem == nullptr) return c_api.CreateStatus(ORT_FAIL, "run_app: allocator failed");
  in.read(static_cast<char*>(mem), static_cast<std::streamsize>(sz));
  if (!in) {
    allocator->Free(allocator, mem);
    return c_api.CreateStatus(ORT_FAIL, "run_app: cipher read failed");
  }

  encapp::XorInPlace(static_cast<uint8_t*>(mem), sz, s->key);
  *buffer = mem;
  *data_size = sz;
  s->bytes_returned = sz;
  std::fprintf(stderr, "[run] read callback fired for \"%s\": decrypted %zu bytes\n",
               file_name ? file_name : "<null>", sz);
  return nullptr;
}

uint8_t ParseKey(const char* arg) {
  if (arg == nullptr) return encapp::kDefaultKey;
  const char* p = arg;
  if (p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) p += 2;
  return static_cast<uint8_t>(std::strtoul(p, nullptr, 16) & 0xFF);
}

// Byte size of one element of the given tensor type (numeric types only).
size_t ElemSize(ONNXTensorElementDataType t) {
  switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return 8;
    default:
      return 0;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s <ctx_model.onnx> <cipher.bin> [xor_key_hex] [input.raw answer_run.raw]\n",
                 argv[0]);
    return 2;
  }
  const std::string ctx_model = argv[1];
  const char* cipher_path = argv[2];
  const uint8_t key = ParseKey(argc >= 4 ? argv[3] : nullptr);
  // Optional real-input / answer-output pair. Both or neither.
  const char* input_raw = (argc >= 6) ? argv[4] : nullptr;
  const char* answer_raw = (argc >= 6) ? argv[5] : nullptr;
  if ((argc == 5) || (argc > 6)) {
    std::fprintf(stderr,
                 "[error] input.raw and answer_run.raw must be given together (got %d arg(s)).\n",
                 argc - 1);
    return 2;
  }

  // Fail fast if the loaded runtime doesn't support ORT_API_VERSION — must run
  // before Ort::GetApi(), which would otherwise dereference null on <1.28.
  if (!encapp::EnsureRuntimeApiAvailable()) return 1;

  const OrtApi& c_api = Ort::GetApi();

  try {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "run_app");

    // Everything touching the QNN EP plugin library (SessionOptions, Session,
    // and every Ort::Value from it) must be destroyed before
    // env.UnregisterExecutionProviderLibrary() unloads it below — hence this
    // nested scope, which closes before that call.
    int rs_call_count = 0;
    size_t rs_bytes_returned = 0;
    size_t num_outputs_returned = 0;
    {
      Ort::SessionOptions session_options;
      std::unordered_map<std::string, std::string> ep_options;
      ep_options["backend_type"] = "htp";
      const bool has_htp = encapp::RegisterAndAppendQnnEp(env, session_options, ep_options);

      // Fail fast if there's no real HTP device — run_app only works on ARM64.
      if (!has_htp) {
        std::fprintf(stderr,
                     "[error] no HTP (NPU) device available — run_app requires real QNN HTP "
                     "hardware and only runs on ARM64. This host reports a compile-only CPU "
                     "device (expected on x86_64); use prepare_app there instead.\n");
        env.UnregisterExecutionProviderLibrary(encapp::QnnEpName());
        return 1;
      }

      ReadState rs;
      rs.key = key;
      rs.cipher_path = cipher_path;

      auto* set_fn =
          Ort::Experimental::Get_OrtApi_SessionOptions_SetEpContextDataReadFunc_SinceV28_Fn(&c_api);
      if (set_fn == nullptr) {
        std::fprintf(stderr,
                     "[error] SetEpContextDataReadFunc_SinceV28 not available — "
                     "ORT runtime is older than 1.28.\n");
        env.UnregisterExecutionProviderLibrary(encapp::QnnEpName());
        return 1;
      }
      if (auto* st = set_fn(session_options, ReadCb, &rs)) {
        c_api.ReleaseStatus(st);
        std::fprintf(stderr, "[error] SetEpContextDataReadFunc returned non-OK\n");
        env.UnregisterExecutionProviderLibrary(encapp::QnnEpName());
        return 1;
      }

#if defined(_WIN32)
      std::wstring ctx_w(ctx_model.begin(), ctx_model.end());
      Ort::Session session(env, ctx_w.c_str(), session_options);
#else
      Ort::Session session(env, ctx_model.c_str(), session_options);
#endif

      if (rs.call_count == 0) {
        std::fprintf(stderr,
                     "[error] read callback never fired — the model may not have an "
                     "external _qnn.bin, or file mapping bypassed the callback.\n");
        env.UnregisterExecutionProviderLibrary(encapp::QnnEpName());
        return 1;
      }

      // With input.raw/answer_run.raw: run on that input and dump the output
      // for a separate tool to compare. Without: run on zeroed inputs (only
      // checks decrypt+load+run works).
      if (input_raw != nullptr) {
        std::vector<float> actual;
        if (!encapp::RunWithFloatInput(session, input_raw, actual)) {
          env.UnregisterExecutionProviderLibrary(encapp::QnnEpName());
          return 1;
        }
        if (!encapp::WriteFloatRaw(answer_raw, actual)) {
          env.UnregisterExecutionProviderLibrary(encapp::QnnEpName());
          return 1;
        }

        rs_call_count = rs.call_count;
        rs_bytes_returned = rs.bytes_returned;
        num_outputs_returned = 1;

        std::fprintf(stdout,
                     "[run] OK: read callback fired %d time(s), decrypted %zu bytes "
                     "(key=0x%02x)\n",
                     rs_call_count, rs_bytes_returned, key);

        const size_t show = actual.size() < 5 ? actual.size() : 5;
        std::fprintf(stdout, "[run] first %zu output value(s):", show);
        for (size_t i = 0; i < show; ++i) std::fprintf(stdout, " %g", actual[i]);
        std::fprintf(stdout, "\n");
        std::fprintf(stdout, "[run] answer written: %zu float value(s) -> \"%s\"\n",
                     actual.size(), answer_raw);
      } else {
        // No answer to write — just prove the decrypted context binary loads
        // and runs.
        Ort::AllocatorWithDefaultOptions allocator;
        const size_t num_inputs = session.GetInputCount();
        const size_t num_outputs = session.GetOutputCount();

        std::vector<std::string> input_name_storage;
        std::vector<Ort::Value> input_values;
        for (size_t i = 0; i < num_inputs; ++i) {
          Ort::AllocatedStringPtr name = session.GetInputNameAllocated(i, allocator);
          input_name_storage.emplace_back(name.get());

          Ort::TypeInfo ti = session.GetInputTypeInfo(i);
          auto tsi = ti.GetTensorTypeAndShapeInfo();
          std::vector<int64_t> shape = tsi.GetShape();
          size_t count = 1;
          for (auto& d : shape) {
            if (d < 0) d = 1;  // pin dynamic dims to 1
            count *= static_cast<size_t>(d);
          }
          const auto elem_type = tsi.GetElementType();

          Ort::Value v = Ort::Value::CreateTensor(allocator, shape.data(), shape.size(), elem_type);
          void* raw = v.GetTensorMutableRawData();
          const size_t total_bytes = count * ElemSize(elem_type);
          if (raw != nullptr && total_bytes > 0) std::memset(raw, 0, total_bytes);
          input_values.emplace_back(std::move(v));
        }
        std::vector<const char*> input_names;
        for (const auto& n : input_name_storage) input_names.push_back(n.c_str());

        std::vector<std::string> output_name_storage;
        for (size_t i = 0; i < num_outputs; ++i) {
          Ort::AllocatedStringPtr name = session.GetOutputNameAllocated(i, allocator);
          output_name_storage.emplace_back(name.get());
        }
        std::vector<const char*> output_names;
        for (const auto& n : output_name_storage) output_names.push_back(n.c_str());

        std::vector<Ort::Value> outputs =
            session.Run(Ort::RunOptions{nullptr},
                        input_names.data(), input_values.data(), input_values.size(),
                        output_names.data(), output_names.size());

        rs_call_count = rs.call_count;
        rs_bytes_returned = rs.bytes_returned;
        num_outputs_returned = outputs.size();

        std::fprintf(stdout,
                     "[run] OK: read callback fired %d time(s), decrypted %zu bytes; "
                     "inference produced %zu output tensor(s) (key=0x%02x)\n",
                     rs_call_count, rs_bytes_returned, num_outputs_returned, key);
        std::fprintf(stdout,
                     "[run] NOTE: ran with zeroed inputs — no answer file written. Pass "
                     "input.raw + answer_run.raw to dump output for comparison.\n");
      }
      // session, outputs, input_values, session_options are destroyed here,
      // while the QNN EP library is still loaded.
    }

    env.UnregisterExecutionProviderLibrary(encapp::QnnEpName());
    return 0;
  } catch (const Ort::Exception& e) {
    std::fprintf(stderr, "[error] Ort exception: %s\n", e.what());
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[error] exception: %s\n", e.what());
    return 1;
  }
}

#else  // ORT_API_VERSION < 28

int main(int, char**) {
  std::fprintf(stderr,
               "[error] run_app requires ORT_API_VERSION >= 28 (ONNX Runtime 1.28+); "
               "this build was compiled against API version %d.\n",
               static_cast<int>(ORT_API_VERSION));
  return 1;
}

#endif  // ORT_API_VERSION >= 28
