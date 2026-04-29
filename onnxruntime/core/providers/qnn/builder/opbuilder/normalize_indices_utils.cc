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
  if (onnx_bytes.size() % sizeof(SrcType) != 0) {
    return false;
  }

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

namespace {

constexpr const char* kOutOfRangeMsg =
    "QNN does not support negative or out-of-range index values for ScatterND-style ops";

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
  if (input_tensorwrapper.GetTensorDataType() == QNN_DATATYPE_INT_64) {
    // Initializers are INT_32 by this point, so INT_64 means dynamic input.
    RETURN_IF_NOT(!indices_info.is_initializer,
                  "Internal error: static indices tensor registered with INT_64 dtype.");
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

}  // namespace

Ort::Status NormalizeIndicesForScatterND(QnnModelWrapper& qnn_model_wrapper,
                                         const OrtNodeUnitIODef& indices_input,
                                         const std::vector<uint32_t>& data_shape,
                                         const Ort::Logger& logger,
                                         std::vector<std::string>& input_names,
                                         bool do_op_validation) {
  std::string indices_tensor_name = indices_input.name;

  TensorInfo indices_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(indices_input, indices_info));

  RETURN_IF_NOT(!indices_info.shape.empty(),
                "ScatterND-style indices tensor must have rank >= 1.");
  const uint32_t k = indices_info.shape.back();
  RETURN_IF_NOT(k > 0 && static_cast<size_t>(k) <= data_shape.size(),
                "ScatterND-style indices last-dim must be in (0, rank(data)].");

  std::vector<uint8_t> qnn_indices_bytes;
  bool has_negative_indices = false;
  bool rewrote_bytes = false;

  const auto axis_dim_for_element = [k, &data_shape](size_t element_index) -> int64_t {
    const size_t col = element_index % static_cast<size_t>(k);
    return static_cast<int64_t>(data_shape[col]);
  };

  if (indices_info.is_initializer) {
    std::vector<uint8_t> onnx_indices_bytes;
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(indices_info.initializer_tensor,
                                                            onnx_indices_bytes));

    if (indices_info.qnn_data_type == QNN_DATATYPE_INT_64) {
      RETURN_IF_NOT((NormalizeIndicesBytes<int64_t>(onnx_indices_bytes, axis_dim_for_element,
                                                    qnn_indices_bytes, has_negative_indices)),
                    kOutOfRangeMsg);
      indices_info.qnn_data_type = QNN_DATATYPE_INT_32;
      rewrote_bytes = true;
    } else if (indices_info.qnn_data_type == QNN_DATATYPE_INT_32) {
      RETURN_IF_NOT((NormalizeIndicesBytes<int32_t>(onnx_indices_bytes, axis_dim_for_element,
                                                    qnn_indices_bytes, has_negative_indices)),
                    kOutOfRangeMsg);
      rewrote_bytes = has_negative_indices;
      if (!rewrote_bytes) {
        qnn_indices_bytes = std::move(onnx_indices_bytes);
      }
    } else {
      qnn_indices_bytes = std::move(onnx_indices_bytes);
    }
  }

  // Rename so a sibling op reusing the same ONNX initializer under a different
  // axis bound cannot alias our rewritten copy.
  if (indices_info.is_initializer && rewrote_bytes) {
    indices_tensor_name = UniqueNameGenerator().New(indices_tensor_name, "_qnn_idx");
    RETURN_IF(qnn_model_wrapper.IsQnnTensorWrapperExist(indices_tensor_name),
              ("Rewritten ScatterND indices name collided with existing tensor: " +
               indices_tensor_name)
                  .c_str());
  }

  return AddNormalizedIndicesTensor(qnn_model_wrapper, std::move(indices_info), indices_tensor_name,
                                    std::move(qnn_indices_bytes), logger, input_names, do_op_validation);
}

}  // namespace utils
}  // namespace qnn
}  // namespace onnxruntime
