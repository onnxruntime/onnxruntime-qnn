// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/qnn_custom_op_domain_parser.h"

#include <string>

#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"  // Ort::Logger full definition + ORT_CXX_LOG_PTR

namespace onnxruntime {

void ParseCustomOpDomains(const std::string& spec,
                          std::vector<CustomOpDomainSpec>& out,
                          const Ort::Logger& logger) {
  // Use ORT_CXX_LOG_PTR (instead of ORT_CXX_LOG) so unit tests can pass a default-constructed
  // Ort::Logger() without crashing. ORT_CXX_LOG dereferences the null OrtLogger* internally.
  const Ort::Logger* logger_ptr = &logger;

  if (spec.empty()) {
    return;
  }

  // Format: domain_name:OpType1[,OpType2][;domain_name2:OpType3]
  for (const auto& entry : qnn::utils::SplitString(spec, ";", /*keep_empty=*/false)) {
    // Split on the first ':' to separate domain name from op-types CSV.
    // string_view::find returns npos if ':' is absent.
    const auto colon_pos = entry.find(':');
    if (colon_pos == std::string_view::npos) {
      ORT_CXX_LOG_PTR(logger_ptr, ORT_LOGGING_LEVEL_WARNING,
                      ("Invalid ORT_QNN_CUSTOM_OP_DOMAINS entry (missing ':'): " + std::string(entry) +
                       ". Expected domain_name:OpType1[,OpType2]. Skipping.")
                          .c_str());
      continue;
    }

    const std::string domain(entry.substr(0, colon_pos));
    const std::string_view op_types_csv = entry.substr(colon_pos + 1);

    if (domain.empty()) {
      ORT_CXX_LOG_PTR(logger_ptr, ORT_LOGGING_LEVEL_WARNING,
                      ("ORT_QNN_CUSTOM_OP_DOMAINS entry has empty domain name: " + std::string(entry) +
                       ". Skipping.")
                          .c_str());
      continue;
    }

    if (op_types_csv.empty()) {
      ORT_CXX_LOG_PTR(logger_ptr, ORT_LOGGING_LEVEL_WARNING,
                      ("ORT_QNN_CUSTOM_OP_DOMAINS entry has empty op-types list for domain '" + domain +
                       "'. Skipping.")
                          .c_str());
      continue;
    }

    std::vector<std::string> op_types;
    for (const auto& op_type_sv : qnn::utils::SplitString(op_types_csv, ",", /*keep_empty=*/false)) {
      const std::string op_type(op_type_sv);
      if (op_type.empty()) {
        ORT_CXX_LOG_PTR(logger_ptr, ORT_LOGGING_LEVEL_WARNING,
                        ("ORT_QNN_CUSTOM_OP_DOMAINS: empty op-type in domain '" + domain + "'. Skipping entry.")
                            .c_str());
        continue;
      }
      op_types.push_back(op_type);
    }

    if (op_types.empty()) {
      ORT_CXX_LOG_PTR(logger_ptr, ORT_LOGGING_LEVEL_WARNING,
                      ("ORT_QNN_CUSTOM_OP_DOMAINS: no valid op-types for domain '" + domain + "'. Skipping.")
                          .c_str());
      continue;
    }

    out.push_back({domain, std::move(op_types)});
  }
}

}  // namespace onnxruntime
