// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string_view>
#include <string>
#include <variant>
#include <vector>

#include <gsl/gsl>
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshorten-64-to-32"
#endif
#include <onnx/onnx_pb.h>
#ifdef __clang__
#pragma clang diagnostic pop
#endif

namespace onnxruntime {
namespace test {

// If set to All: verify the entire graph is taken by ep
// If set to Some: verify that at least one node is assigned to ep
// If set to None: verify that no nodes is assigned to ep (typically for an expected failure path test case)
enum class ExpectedEPNodeAssignment { None,
                                      Some,
                                      All,
};

struct ElementwiseAbsoluteVerifier {
  explicit ElementwiseAbsoluteVerifier(float fp32_abs_err = 1e-5f) : fp32_abs_err(fp32_abs_err) {}

  // Some EP may use different rounding than ORT CPU EP, which may cause a bigger abs error than
  // the default of 1e-5f, especially for scenarios such as [Q -> Quantized op -> DQ]
  // Set this only if this is necessary
  float fp32_abs_err;
};

struct CosineSimilarityVerifier {
  explicit CosineSimilarityVerifier(float fp32_cs_threshold) : fp32_cs_threshold(fp32_cs_threshold) {}

  float fp32_cs_threshold;
};

using TensorVerifier = std::variant<ElementwiseAbsoluteVerifier, CosineSimilarityVerifier>;

// The struct to hold some verification params for RunAndVerifyOutputsWithEP
struct EPVerificationParams {
  ExpectedEPNodeAssignment ep_node_assignment = ExpectedEPNodeAssignment::Some;
  TensorVerifier tensor_verifier = ElementwiseAbsoluteVerifier{};

  // optional graph verification function (uses public ORT Session API)
  const std::function<void(const Ort::Session&)>* graph_verifier{nullptr};
};

// Verify equality of two output tensors.
void VerifyOutput(const std::string& output_name,
                  const Ort::Value& expected_value,
                  const Ort::Value& actual_value,
                  const TensorVerifier& tensor_verifier);

size_t CountNodes(const Ort::Session& current_session);

size_t CountAssignedNodes(const Ort::Session& current_session, const std::string& ep_type);

// Verify the assignment of nodes to the EP specified by `provider_type`.
void VerifyEPNodeAssignment(const Ort::Session& current_session, const std::string& provider_type,
                            ExpectedEPNodeAssignment assignment);

using ModelPathOrBytes = std::variant<std::basic_string_view<ORTCHAR_T>,
                                      gsl::span<const std::byte>>;

// Run the model using the CPU EP to get expected output, comparing to the output when the 'execution_provider'
// is enabled.
// session_options_updater can be used to update the SessionOptions the inference session is created with.
void RunAndVerifyOutputsWithEP(ModelPathOrBytes model_path_or_bytes,
                               Ort::SessionOptions& ort_so,
                               const std::string& provider_type,
                               std::string_view log_id,
                               const std::unordered_map<std::string, Ort::Value>& feeds,
                               const EPVerificationParams& params = EPVerificationParams(),
                               bool verify_outputs = true,
                               Ort::CustomOpDomain* custom_op_domain = nullptr);

void RunWithEP(Ort::Session& ort_session,
               const Ort::RunOptions& ort_ro,
               const std::unordered_map<std::string, Ort::Value>& feeds,
               std::vector<Ort::Value>& output_vals);

// QNN-EP COPY START
// Below are ONNX Attributes utilities copied from MS onnxruntime\core\graph\node_attr_utils.h directly.
// keep these signatures in sync with DECLARE_MAKE_ATTRIBUTE_FNS below
/** Creates an AttributeProto with the specified name and value. */
ONNX_NAMESPACE::AttributeProto MakeAttribute(std::string attr_name, int64_t value);
/** Creates an AttributeProto with the specified name and values. */
ONNX_NAMESPACE::AttributeProto MakeAttribute(std::string attr_name, gsl::span<const int64_t> values);

#define DECLARE_MAKE_ATTRIBUTE_FNS(type)                                           \
  ONNX_NAMESPACE::AttributeProto MakeAttribute(std::string attr_name, type value); \
  ONNX_NAMESPACE::AttributeProto MakeAttribute(std::string attr_name, gsl::span<const type> values)

DECLARE_MAKE_ATTRIBUTE_FNS(float);
DECLARE_MAKE_ATTRIBUTE_FNS(std::string);
DECLARE_MAKE_ATTRIBUTE_FNS(ONNX_NAMESPACE::TensorProto);
#if !defined(DISABLE_SPARSE_TENSORS)
DECLARE_MAKE_ATTRIBUTE_FNS(ONNX_NAMESPACE::SparseTensorProto);
#endif
DECLARE_MAKE_ATTRIBUTE_FNS(ONNX_NAMESPACE::TypeProto);
DECLARE_MAKE_ATTRIBUTE_FNS(ONNX_NAMESPACE::GraphProto);

#undef DECLARE_MAKE_ATTRIBUTE_FNS
// QNN-EP COPY END

// QNN_ASSERT macro that imitates ORT_ENFORCE pattern
#define QNN_ASSERT(condition)                                             \
  do {                                                                    \
    if (!(condition)) {                                                   \
      ORT_CXX_API_THROW(#condition, OrtErrorCode::ORT_RUNTIME_EXCEPTION); \
    }                                                                     \
  } while (false)

}  // namespace test
}  // namespace onnxruntime
