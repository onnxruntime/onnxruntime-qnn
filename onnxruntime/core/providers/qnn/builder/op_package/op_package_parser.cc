// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/op_package/op_package_parser.h"

#include <cctype>

#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"  // Ort::Logger full definition + ORT_CXX_LOG_PTR / IsNullLogger

namespace onnxruntime {

void ParseOpPackages(const std::string& op_packages_string,
                     std::vector<onnxruntime::qnn::OpPackage>& op_packages,
                     const Ort::Logger& logger) {
  // Use ORT_CXX_LOG_PTR (instead of ORT_CXX_LOG) so unit tests can pass a default-constructed
  // Ort::Logger() without crashing. ORT_CXX_LOG dereferences the null OrtLogger* internally.
  const Ort::Logger* logger_ptr = &logger;
  for (const auto& op_package : qnn::utils::SplitString(op_packages_string, ",", true)) {
    auto splitStrings = qnn::utils::SplitString(op_package, ":", true);
#if defined(_WIN32)
    // On Windows, paths include a drive letter followed by ":" (e.g., "C:\foo.dll").
    // The split-on-':' produces a separate token for the drive letter; rejoin it with the
    // following segment when token[1] looks like a single-letter drive prefix. Gate on
    // token shape only — parsing of the config string must be deterministic in the input
    // and must not depend on filesystem state.
    // `merged_win_path` owns the merged string so `splitStrings[1]` (a string_view) remains
    // valid for the rest of this iteration.
    std::string merged_win_path;
    if (splitStrings.size() >= 3 &&
        splitStrings[1].size() == 1 &&
        std::isalpha(static_cast<unsigned char>(splitStrings[1][0]))) {
      merged_win_path = std::string(splitStrings[1]) + ":" + std::string(splitStrings[2]);
      splitStrings[1] = merged_win_path;
      splitStrings.erase(splitStrings.begin() + 2);
    }
#endif
    if (splitStrings.size() < 3 || splitStrings.size() > 4) {
      std::string msg =
          "Invalid op_package passed, "
          "expected <OpType>:<PackagePath>:<InterfaceSymbolName>[:<Target>], "
          "got ";
      ORT_CXX_LOG_PTR(logger_ptr, ORT_LOGGING_LEVEL_WARNING, (msg + std::string(op_package)).c_str());
      ORT_CXX_LOG_PTR(logger_ptr, ORT_LOGGING_LEVEL_WARNING, "Skip registration.");
      continue;
    }

    std::string op_type = std::string(splitStrings[0]);
    std::string op_package_path = std::string(splitStrings[1]);
    std::string op_package_interface = std::string(splitStrings[2]);
    std::string op_package_target;

    if (op_type.empty()) {
      ORT_CXX_LOG_PTR(logger_ptr, ORT_LOGGING_LEVEL_WARNING, "Op type is empty. Skip registration.");
      continue;
    }

    if (op_package_path.empty()) {
      ORT_CXX_LOG_PTR(logger_ptr, ORT_LOGGING_LEVEL_WARNING, "Op package path is empty. Skip registration");
      continue;
    }

    if (op_package_interface.empty()) {
      ORT_CXX_LOG_PTR(logger_ptr, ORT_LOGGING_LEVEL_WARNING, "Op package interface is empty. Skip registration");
      continue;
    }

    ORT_CXX_LOG_PTR(logger_ptr,
                    ORT_LOGGING_LEVEL_VERBOSE,
                    ("Loading op package from path: " + op_package_path + " for op " + op_type).c_str());
    ORT_CXX_LOG_PTR(logger_ptr, ORT_LOGGING_LEVEL_VERBOSE, ("Op package interface " + op_package_interface).c_str());
    if (splitStrings.size() > 3 && splitStrings[3].size()) {
      op_package_target = std::string(splitStrings[3]);
      ORT_CXX_LOG_PTR(logger_ptr, ORT_LOGGING_LEVEL_VERBOSE, ("Op package target: " + op_package_target).c_str());
    }
    op_packages.push_back({op_type, op_package_path, op_package_interface, op_package_target});
  }
}

}  // namespace onnxruntime
