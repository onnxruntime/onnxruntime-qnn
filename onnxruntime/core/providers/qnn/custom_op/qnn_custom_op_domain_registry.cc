// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/custom_op/qnn_custom_op_domain_registry.h"

#include <cstdlib>
#include <utility>

#include "core/providers/qnn/custom_op/qnn_custom_op_domain_parser.h"

namespace onnxruntime {

namespace {

// Returns the raw value of ORT_QNN_CUSTOM_OP_DOMAINS (or empty string if unset).
// Mirrors the platform-safe env-var read pattern used elsewhere in the EP
// (see QnnCpuBackendEnabled() in qnn_provider_factory.cc).
std::string ReadCustomOpDomainsEnvVar() {
#if defined(_WIN32)
  char* value = nullptr;
  size_t value_size = 0;
  const bool found = _dupenv_s(&value, &value_size, "ORT_QNN_CUSTOM_OP_DOMAINS") == 0 && value != nullptr;
  std::string result = found ? std::string(value) : std::string{};
  free(value);
  return result;
#else
  const char* value = std::getenv("ORT_QNN_CUSTOM_OP_DOMAINS");
  return value != nullptr ? std::string(value) : std::string{};
#endif
}

}  // namespace

void BuildCustomOpDomainsFromEnv(const Ort::Logger& logger,
                                 const std::string& ep_name,
                                 std::vector<Ort::CustomOpDomain>& out_domains,
                                 std::vector<std::unique_ptr<qnn::QnnUdoPlaceholderOp>>& out_ops) {
  const std::string custom_op_domains_spec = ReadCustomOpDomainsEnvVar();
  if (custom_op_domains_spec.empty()) {
    return;
  }

  std::vector<CustomOpDomainSpec> specs;
  ParseCustomOpDomains(custom_op_domains_spec, specs, logger);
  for (const auto& spec : specs) {
    Ort::CustomOpDomain domain{spec.domain.c_str()};
    for (const auto& op_type : spec.op_types) {
      auto op = std::make_unique<qnn::QnnUdoPlaceholderOp>(op_type, ep_name);
      domain.Add(op.get());
      out_ops.push_back(std::move(op));
    }
    out_domains.push_back(std::move(domain));
  }
}

}  // namespace onnxruntime
