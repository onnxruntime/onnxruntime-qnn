// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#ifdef _WIN32
#include <iostream>
#include <locale>
#endif

#ifndef USE_ONNXRUNTIME_DLL
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <google/protobuf/message_lite.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#endif

#include "gtest/gtest.h"
#include "onnxruntime_cxx_api.h"

#include "test/util/env_var_utils.h"

std::unique_ptr<Ort::Env> ort_env;

// define environment variable name constants here
namespace env_var_names {
// Set ORT log level to the specified numeric log level.
constexpr const char* kLogLevel = "ORT_UNIT_TEST_MAIN_LOG_LEVEL";
}  // namespace env_var_names

// ortenv_setup() and ortenv_teardown() are used by onnxruntime/test/xctest/xcgtest.mm so can't be file local
extern "C" void ortenv_setup() {
  try {
#ifdef _WIN32
    // Set the locale to UTF-8 to ensure proper handling of wide characters on Windows
    std::wclog.imbue(std::locale(".UTF-8", std::locale::ctype));
#endif

    OrtLoggingLevel log_level = ORT_LOGGING_LEVEL_WARNING;
    if (auto log_level_override = ParseEnvironmentVariable<int>(env_var_names::kLogLevel);
        log_level_override.has_value()) {
      *log_level_override = std::clamp(*log_level_override,
                                       static_cast<int>(ORT_LOGGING_LEVEL_VERBOSE),
                                       static_cast<int>(ORT_LOGGING_LEVEL_FATAL));
      std::cout << "Setting log level to " << *log_level_override << "\n";
      log_level = static_cast<OrtLoggingLevel>(*log_level_override);
    }

    Ort::ThreadingOptions tpo;
    ort_env.reset(new Ort::Env(tpo, log_level, "Default"));
  } catch (const std::exception& ex) {
    std::cerr << ex.what();
    std::exit(1);
  }
}

extern "C" void ortenv_teardown() {
  ort_env.reset();
}

static std::vector<std::unique_ptr<::testing::TestEventListener>> MakeTestEventListeners() {
  std::vector<std::unique_ptr<::testing::TestEventListener>> result{};
  return result;
}

#define TEST_MAIN main

#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_SIMULATOR || TARGET_OS_IOS
#undef TEST_MAIN
#define TEST_MAIN main_no_link_  // there is a UI test app for iOS.
#endif
#endif

int TEST_MAIN(int argc, char** argv) {
  int status = 0;

  try {
    ortenv_setup();
    ::testing::InitGoogleTest(&argc, argv);

    {
      auto& test_listeners = ::testing::UnitTest::GetInstance()->listeners();
      auto test_listeners_to_add = MakeTestEventListeners();
      for (auto& test_listener_to_add : test_listeners_to_add) {
        test_listeners.Append(test_listener_to_add.release());
      }
    }

    status = RUN_ALL_TESTS();
  } catch (const std::exception& ex) {
    std::cerr << ex.what();
    status = -1;
  }

  // TODO: Fix the C API issue
  ortenv_teardown();  // If we don't do this, it will crash

#ifndef USE_ONNXRUNTIME_DLL
  // make memory leak checker happy
  ::google::protobuf::ShutdownProtobufLibrary();
#endif
  return status;
}
