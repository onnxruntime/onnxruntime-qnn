// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/opbuilder/normalize_indices_utils.h"

#include <cstdint>
#include <cstddef>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gsl/gsl>

#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {
namespace utils {

template <typename SrcType>
bool NormalizeIndicesBytes(gsl::span<const uint8_t> onnx_bytes,
                           const std::function<int64_t(size_t)>& axis_dim_for_element,
                           std::vector<uint8_t>& qnn_bytes,
                           bool& has_negative_indices) {
  const size_t num_elems = onnx_bytes.size() / sizeof(SrcType);
  const auto onnx_indices = gsl::span<const SrcType>{
      reinterpret_cast<const SrcType*>(onnx_bytes.data()), num_elems};

  qnn_bytes.resize(num_elems * sizeof(int32_t));
  const auto qnn_indices = gsl::span<int32_t>{
      reinterpret_cast<int32_t*>(qnn_bytes.data()), num_elems};

  for (size_t i = 0; i < num_elems; ++i) {
    const int64_t axis_dim = axis_dim_for_element(i);
    // int64 prevents wraparound on int32 idx + axis_dim >= 2^31.
    int64_t idx = static_cast<int64_t>(onnx_indices[i]);
    if (idx < 0) {
      has_negative_indices = true;
      idx += axis_dim;
    }
    // ORT does not reject out-of-range initializer values; guard here.
    if (idx < 0 || idx >= axis_dim ||
        idx > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
      return false;
    }
    qnn_indices[i] = static_cast<int32_t>(idx);
  }

  return true;
}

template bool NormalizeIndicesBytes<int32_t>(
    gsl::span<const uint8_t>, const std::function<int64_t(size_t)>&,
    std::vector<uint8_t>&, bool&);
template bool NormalizeIndicesBytes<int64_t>(
    gsl::span<const uint8_t>, const std::function<int64_t(size_t)>&,
    std::vector<uint8_t>&, bool&);

Ort::Status AddNormalizedIndicesTensor(QnnModelWrapper& qnn_model_wrapper,
                                       TensorInfo indices_info,
                                       const std::string& indices_tensor_name,
                                       std::vector<uint8_t>&& qnn_indices_bytes,
                                       const Ort::Logger& logger,
                                       std::vector<std::string>& input_names,
                                       bool do_op_validation) {
  std::vector<uint32_t> cast_output_shape(indices_info.shape);

  if (!qnn_model_wrapper.IsQnnTensorWrapperExist(indices_tensor_name)) {
    const Qnn_TensorType_t tensor_type = indices_info.is_initializer
                                             ? QNN_TENSOR_TYPE_STATIC
                                             : qnn_model_wrapper.GetTensorType(indices_tensor_name);
    QnnTensorWrapper input_tensorwrapper(indices_tensor_name,
                                         tensor_type,
                                         indices_info.qnn_data_type, QnnQuantParamsWrapper(),
                                         std::move(indices_info.shape),
                                         std::move(qnn_indices_bytes));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)),
                  "Failed to add tensor.");
  } else {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                ("Tensor already added, skip it: " + indices_tensor_name).c_str());
  }

  auto& input_tensorwrapper = qnn_model_wrapper.GetQnnTensorWrapper(indices_tensor_name);
  std::string indices_casted_name = indices_tensor_name;
  // Initializers are rewritten to INT_32 before this point, so INT_64 here
  // implies a dynamic input that needs a runtime Cast.
  if (input_tensorwrapper.GetTensorDataType() == QNN_DATATYPE_INT_64) {
    indices_casted_name += "_int32";
    RETURN_IF_ERROR(qnn_model_wrapper.AddCastNode(
        UniqueNameGenerator().New(indices_tensor_name, QNN_OP_CAST),
        indices_tensor_name,
        indices_casted_name,
        QNN_TENSOR_TYPE_NATIVE,
        QNN_DATATYPE_INT_32,
        QnnQuantParamsWrapper(),
        std::move(cast_output_shape),
        do_op_validation));
  }
  input_names.push_back(indices_casted_name);
  return Ort::Status();
}

}  // namespace utils
}  // namespace qnn
}  // namespace onnxruntime
