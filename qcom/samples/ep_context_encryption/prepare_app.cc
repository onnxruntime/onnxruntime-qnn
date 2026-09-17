// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// prepare_app — "Prepare" stage of the compiled-model-encryption example.
//
// Compiles an ONNX model with the QNN EP (embed_mode=0) and registers a write
// callback (ORT 1.28 EPContext API) that encrypts the external _qnn.bin
// instead of letting ORT write it as plaintext.
//
// This is an EXAMPLE: WriteCb below just XORs the bytes, to keep the callback
// mechanism obvious. Replace WriteCb with your own encryption to use this for
// real — everything else (how ORT hands you the bytes) stays the same.
//
// Usage:
//   prepare_app <input_model.onnx> <output_ctx.onnx> <output_cipher.bin> [xor_key_hex]
//               [input.raw answer_prepare.raw] [htp_arch]
//
//   input_model.onnx     a model QNN EP can offload to HTP (e.g. a QDQ model)
//   output_ctx.onnx      the compiled EPContext wrapper model ORT writes
//   output_cipher.bin    where the write callback stores the encrypted _qnn.bin
//   xor_key_hex          optional 1-byte key in hex (default 5a)
//   input.raw            optional float32 input; when given, the plaintext model
//                        runs once on it and writes answer_prepare.raw
//   answer_prepare.raw   where to write that plaintext-model output (float32)
//   htp_arch             optional target HTP arch (81/73) for cross-platform
//                        compile on a host with no real HTP device
//
// Whether the decrypted-model output (from run_app) matches answer_prepare.raw
// is not checked by this app — see compare_answers.py.
//
// Exit code 0 on success; non-zero with a message on failure.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#include "enc_common.h"

#if ORT_API_VERSION >= 28

namespace {

// State handed to the write callback via the opaque void* state pointer.
struct WriteState {
  uint8_t key = encapp::kDefaultKey;
  std::ofstream out;
  size_t total = 0;
  bool io_error = false;
};

// OrtWriteNamedBufferFunc: (state, file_name, buffer, size) -> OrtStatus*.
// Example cipher: XOR the EPContext bytes and append to the cipher file.
// Replace this body with your own encryption. Returning non-null fails the
// compile (fail-closed).
OrtStatus* WriteCb(void* state, const char* file_name,
                   const void* buffer, size_t n) noexcept {
  auto* s = static_cast<WriteState*>(state);
  const auto* src = static_cast<const uint8_t*>(buffer);
  std::vector<uint8_t> enc(n);
  for (size_t i = 0; i < n; ++i) enc[i] = src[i] ^ s->key;
  s->out.write(reinterpret_cast<const char*>(enc.data()),
               static_cast<std::streamsize>(n));
  if (!s->out) {
    s->io_error = true;
    return Ort::GetApi().CreateStatus(ORT_FAIL, "prepare_app: failed writing cipher file");
  }
  s->total += n;
  std::fprintf(stderr, "[prepare] write callback fired for \"%s\": %zu bytes\n",
               file_name ? file_name : "<null>", n);
  return nullptr;
}

// Parse a 1-byte hex key like "5a" / "0x5A".
uint8_t ParseKey(const char* arg) {
  if (arg == nullptr) return encapp::kDefaultKey;
  const char* p = arg;
  if (p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) p += 2;
  return static_cast<uint8_t>(std::strtoul(p, nullptr, 16) & 0xFF);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: %s <input_model.onnx> <output_ctx.onnx> <output_cipher.bin> [xor_key_hex]"
                 " [input.raw answer_prepare.raw] [htp_arch]\n",
                 argv[0]);
    return 2;
  }
  const char* input_model = argv[1];
  const std::string output_ctx = argv[2];
  const char* output_cipher = argv[3];
  const uint8_t key = ParseKey(argc >= 5 ? argv[4] : nullptr);
  // Optional real-input / answer-output pair. Both or neither.
  const char* input_raw = (argc >= 7) ? argv[5] : nullptr;
  const char* answer_raw = (argc >= 7) ? argv[6] : nullptr;
  if ((argc == 6) || (argc > 8)) {
    std::fprintf(stderr,
                 "[error] input.raw and answer_prepare.raw must be given together (got %d arg(s)).\n",
                 argc - 1);
    return 2;
  }
  // Optional target HTP arch for cross-platform compile (see usage comment above).
  const char* htp_arch = (argc == 8) ? argv[7] : nullptr;

  // Fail fast if the loaded runtime doesn't support ORT_API_VERSION — must run
  // before Ort::GetApi(), which would otherwise dereference null on <1.28.
  if (!encapp::EnsureRuntimeApiAvailable()) return 1;

  const OrtApi& c_api = Ort::GetApi();

  try {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "prepare_app");
    Ort::SessionOptions session_options;

    std::unordered_map<std::string, std::string> ep_options;
    ep_options["backend_type"] = "htp";
    // Target arch when this host has no real HTP device to auto-detect from.
    if (htp_arch != nullptr) {
      ep_options["htp_arch"] = htp_arch;
    }
    encapp::RegisterAndAppendQnnEp(env, session_options, ep_options);

    // Optional: run the plaintext model once and save its output. Whether
    // this matches run_app's decrypted-path output is checked separately by
    // compare_answers.py, not by this app.
    if (input_raw != nullptr) {
      std::fprintf(stderr, "[prepare] running plaintext model on \"%s\"\n", input_raw);
      Ort::SessionOptions plain_options;
      encapp::RegisterAndAppendQnnEp(env, plain_options, ep_options);
#if defined(_WIN32)
      std::wstring model_w(input_model, input_model + std::strlen(input_model));
      Ort::Session plain_session(env, model_w.c_str(), plain_options);
#else
      Ort::Session plain_session(env, input_model, plain_options);
#endif
      std::vector<float> answer;
      if (!encapp::RunWithFloatInput(plain_session, input_raw, answer)) {
        env.UnregisterExecutionProviderLibrary(encapp::QnnEpName());
        return 1;
      }
      if (!encapp::WriteFloatRaw(answer_raw, answer)) {
        env.UnregisterExecutionProviderLibrary(encapp::QnnEpName());
        return 1;
      }
      std::fprintf(stdout, "[prepare] answer written: %zu float value(s) -> \"%s\"\n",
                   answer.size(), answer_raw);
    }

    WriteState ws;
    ws.key = key;
    ws.out.open(output_cipher, std::ios::binary);
    if (!ws.out.is_open()) {
      std::fprintf(stderr, "[error] cannot open cipher output file: %s\n", output_cipher);
      env.UnregisterExecutionProviderLibrary(encapp::QnnEpName());
      return 1;
    }

    Ort::ModelCompilationOptions compile_options(env, session_options);

    // embed_mode=0 → external _qnn.bin, which the write callback intercepts.
#if defined(_WIN32)
    std::wstring input_model_w(input_model, input_model + std::strlen(input_model));
    std::wstring output_ctx_w(output_ctx.begin(), output_ctx.end());
    const ORTCHAR_T* input_model_p = input_model_w.c_str();
    const ORTCHAR_T* output_ctx_p = output_ctx_w.c_str();
#else
    const ORTCHAR_T* input_model_p = input_model;
    const ORTCHAR_T* output_ctx_p = output_ctx.c_str();
#endif

    compile_options.SetInputModelPath(input_model_p);
    compile_options.SetEpContextEmbedMode(false);
    compile_options.SetOutputModelPath(output_ctx_p);
    compile_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

    // Resolve and register the write callback (ORT 1.28 experimental API).
    auto* set_fn =
        Ort::Experimental::Get_OrtCompileApi_ModelCompilationOptions_SetEpContextDataWriteFunc_SinceV28_Fn(&c_api);
    if (set_fn == nullptr) {
      std::fprintf(stderr,
                   "[error] SetEpContextDataWriteFunc_SinceV28 not available — "
                   "ORT runtime is older than 1.28.\n");
      ws.out.close();
      env.UnregisterExecutionProviderLibrary(encapp::QnnEpName());
      return 1;
    }
    if (auto* st = set_fn(compile_options, WriteCb, &ws)) {
      c_api.ReleaseStatus(st);
      std::fprintf(stderr, "[error] SetEpContextDataWriteFunc returned non-OK\n");
      ws.out.close();
      env.UnregisterExecutionProviderLibrary(encapp::QnnEpName());
      return 1;
    }

    Ort::Status cs = Ort::CompileModel(env, compile_options);
    ws.out.close();

    if (!cs.IsOK()) {
      std::fprintf(stderr, "[error] CompileModel failed: %s\n", cs.GetErrorMessage().c_str());
      env.UnregisterExecutionProviderLibrary(encapp::QnnEpName());
      return 1;
    }
    if (ws.io_error) {
      std::fprintf(stderr, "[error] cipher file write error during compile\n");
      env.UnregisterExecutionProviderLibrary(encapp::QnnEpName());
      return 1;
    }
    if (ws.total == 0) {
      std::fprintf(stderr,
                   "[error] write callback never fired — the model was not offloaded "
                   "to QNN HTP (no _qnn.bin produced). Use a model QNN can compile.\n");
      env.UnregisterExecutionProviderLibrary(encapp::QnnEpName());
      return 1;
    }

    std::fprintf(stdout,
                 "[prepare] OK: compiled \"%s\" -> ctx \"%s\", encrypted %zu bytes -> \"%s\" (key=0x%02x)\n",
                 input_model, output_ctx.c_str(), ws.total, output_cipher, key);

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
               "[error] prepare_app requires ORT_API_VERSION >= 28 (ONNX Runtime 1.28+); "
               "this build was compiled against API version %d.\n",
               static_cast<int>(ORT_API_VERSION));
  return 1;
}

#endif  // ORT_API_VERSION >= 28
