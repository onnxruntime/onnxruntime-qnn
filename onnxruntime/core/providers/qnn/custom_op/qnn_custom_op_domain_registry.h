// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/providers/qnn/ort_api.h"  // Ort::Logger, Ort::CustomOpDomain
#include "core/providers/qnn/custom_op/qnn_custom_op.h"

namespace onnxruntime {

// Reads ORT_QNN_CUSTOM_OP_DOMAINS, parses it (see ParseCustomOpDomains), and builds one
// Ort::CustomOpDomain plus one QnnUdoPlaceholderOp per (domain, op_type) pair. Ownership of
// the built placeholder ops is returned via `out_ops` so the caller (QnnEpFactory) can keep
// them alive for as long as the domains are registered with ORT.
//
// No-op (leaves out_domains/out_ops untouched) if the env var is unset or empty.
void BuildCustomOpDomainsFromEnv(const Ort::Logger& logger,
                                 const std::string& ep_name,
                                 std::vector<Ort::CustomOpDomain>& out_domains,
                                 std::vector<std::unique_ptr<qnn::QnnUdoPlaceholderOp>>& out_ops);

}  // namespace onnxruntime
