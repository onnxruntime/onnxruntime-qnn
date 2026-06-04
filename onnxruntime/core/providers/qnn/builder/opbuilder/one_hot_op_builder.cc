// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

// ONNX OneHot has three tensor inputs (indices, depth, values); QNN OneHot has one
// tensor input (indices) and three parameters (depth, on_value, off_value) plus an
// optional axis parameter. This builder:
//   - Passes only input[0] (indices) as a QNN tensor input.
//   - Extracts depth from input[1] (must be a constant initializer).
//   - Extracts on_value / off_value from input[2] (values[1] / values[0]).
//   - Normalises the ONNX axis (default -1, allows negatives) to a non-negative
//     UINT_32 for QNN.
//   - Handles INT_64 indices by inserting a runtime Cast(INT_32) node.
//   - Rejects unsupported types (string, complex) in IsOpSupported.

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class OneHotOpBuilder : public BaseOpBuilder {
 public:
  OneHotOpBuilder() : BaseOpBuilder("OneHotOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OneHotOpBuilder);

 protected:
  Ort::Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger,
                            std::vector<std::string>& input_names,
                            bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

// Reads a scalar initializer value and sets the corresponding field in a Qnn_Scalar_t.
// Handles all non-quantized numeric types that QNN OneHot on_value/off_value can accept.
// If the values tensor is quantized (DQ-wrapped), dequantizes the raw integer to float32
// following the same convention as PadOpBuilder::ProcessConstantValue.
static Ort::Status ExtractScalarFromInitializer(QnnModelWrapper& qnn_model_wrapper,
                                                const OrtNodeUnitIODef& input_def,
                                                size_t byte_offset,
                                                Qnn_Scalar_t& out_scalar) {
  TensorInfo info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input_def, info));
  RETURN_IF_NOT(info.initializer_tensor != nullptr,
                "OneHot: values input must be a constant initializer.");

  std::vector<uint8_t> bytes;
  RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(info.initializer_tensor, bytes));

  const uint8_t* data = bytes.data() + byte_offset;

  // If the values tensor is quantized, dequantize back to float32 for the QNN scalar param.
  // QNN expects on_value/off_value in the logical (float) domain; quantization of the output
  // is handled separately via the output tensor's quant params.
  if (input_def.quant_param.has_value()) {
    RETURN_IF_NOT(info.quant_param.IsPerTensor(),
                  "OneHot: values tensor must use per-tensor quantization.");
    const Qnn_QuantizeParams_t& qp = info.quant_param.Get();
    const float scale = qp.scaleOffsetEncoding.scale;
    const int32_t offset = qp.scaleOffsetEncoding.offset;

    double raw_val = 0.0;
    switch (info.qnn_data_type) {
      case QNN_DATATYPE_SFIXED_POINT_8: {
        int8_t v = 0;
        memcpy(&v, data, sizeof(int8_t));
        raw_val = static_cast<double>(v);
        break;
      }
      case QNN_DATATYPE_SFIXED_POINT_16: {
        int16_t v = 0;
        memcpy(&v, data, sizeof(int16_t));
        raw_val = static_cast<double>(v);
        break;
      }
      case QNN_DATATYPE_SFIXED_POINT_32: {
        int32_t v = 0;
        memcpy(&v, data, sizeof(int32_t));
        raw_val = static_cast<double>(v);
        break;
      }
      case QNN_DATATYPE_UFIXED_POINT_8: {
        raw_val = static_cast<double>(data[0]);
        break;
      }
      case QNN_DATATYPE_UFIXED_POINT_16: {
        uint16_t v = 0;
        memcpy(&v, data, sizeof(uint16_t));
        raw_val = static_cast<double>(v);
        break;
      }
      case QNN_DATATYPE_UFIXED_POINT_32: {
        uint32_t v = 0;
        memcpy(&v, data, sizeof(uint32_t));
        raw_val = static_cast<double>(v);
        break;
      }
      default:
        return MAKE_EP_FAIL("OneHot: unexpected quantized data type for values tensor.");
    }
    out_scalar.dataType = QNN_DATATYPE_FLOAT_32;
    out_scalar.floatValue = static_cast<float>(utils::Dequantize(offset, scale, raw_val));
    return Ort::Status();
  }

  // Non-quantized path: read raw bytes directly.
  out_scalar.dataType = info.qnn_data_type;

  switch (info.qnn_data_type) {
    case QNN_DATATYPE_FLOAT_32: {
      float v = 0.0f;
      memcpy(&v, data, sizeof(float));
      out_scalar.floatValue = v;
      break;
    }
    case QNN_DATATYPE_FLOAT_16: {
      // Qnn_Scalar_t stores float16 as floatValue (no separate float16 field).
      // Read the raw fp16 bits and convert to float32 for the scalar.
      uint16_t bits = 0;
      memcpy(&bits, data, sizeof(uint16_t));
      Ort::Float16_t fp16_val;
      fp16_val.val = bits;
      out_scalar.floatValue = static_cast<float>(fp16_val);
      break;
    }
    case QNN_DATATYPE_BFLOAT_16: {
      // QNN does not support BFLOAT_16 as a scalar param type; represent as float32.
      uint16_t bits = 0;
      memcpy(&bits, data, sizeof(uint16_t));
      Ort::BFloat16_t bf16_val;
      bf16_val.val = bits;
      out_scalar.dataType = QNN_DATATYPE_FLOAT_32;
      out_scalar.floatValue = static_cast<float>(bf16_val);
      break;
    }
    case QNN_DATATYPE_FLOAT_64: {
      double v = 0.0;
      memcpy(&v, data, sizeof(double));
      out_scalar.doubleValue = v;
      break;
    }
    case QNN_DATATYPE_INT_8: {
      int8_t v = 0;
      memcpy(&v, data, sizeof(int8_t));
      out_scalar.int8Value = v;
      break;
    }
    case QNN_DATATYPE_INT_16: {
      int16_t v = 0;
      memcpy(&v, data, sizeof(int16_t));
      out_scalar.int16Value = v;
      break;
    }
    case QNN_DATATYPE_INT_32: {
      int32_t v = 0;
      memcpy(&v, data, sizeof(int32_t));
      out_scalar.int32Value = v;
      break;
    }
    case QNN_DATATYPE_INT_64: {
      int64_t v = 0;
      memcpy(&v, data, sizeof(int64_t));
      out_scalar.int64Value = v;
      break;
    }
    case QNN_DATATYPE_UINT_8: {
      out_scalar.uint8Value = data[0];
      break;
    }
    case QNN_DATATYPE_UINT_16: {
      uint16_t v = 0;
      memcpy(&v, data, sizeof(uint16_t));
      out_scalar.uint16Value = v;
      break;
    }
    case QNN_DATATYPE_UINT_32: {
      uint32_t v = 0;
      memcpy(&v, data, sizeof(uint32_t));
      out_scalar.uint32Value = v;
      break;
    }
    case QNN_DATATYPE_UINT_64: {
      uint64_t v = 0;
      memcpy(&v, data, sizeof(uint64_t));
      out_scalar.uint64Value = v;
      break;
    }
    case QNN_DATATYPE_BOOL_8: {
      out_scalar.bool8Value = data[0];
      break;
    }
    default:
      return MAKE_EP_FAIL("OneHot: unsupported data type for on_value/off_value.");
  }
  return Ort::Status();
}

Ort::Status OneHotOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                           const OrtNodeUnit& node_unit,
                                           const Ort::Logger& logger,
                                           std::vector<std::string>& input_names,
                                           bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  RETURN_IF(inputs.size() < 3, "OneHot requires 3 inputs: indices, depth, values.");

  if (do_op_validation) {
    // depth must be a constant initializer — we extract it at compile time.
    RETURN_IF_NOT(qnn_model_wrapper.IsConstantInput(inputs[1].name),
                  "OneHot: depth input must be a constant initializer.");

    // values must be a constant initializer.
    RETURN_IF_NOT(qnn_model_wrapper.IsConstantInput(inputs[2].name),
                  "OneHot: values input must be a constant initializer.");

    // Reject unsupported indices types.
    TensorInfo indices_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], indices_info));
    const Qnn_DataType_t idx_type = indices_info.qnn_data_type;
    RETURN_IF(idx_type != QNN_DATATYPE_INT_32 &&
                  idx_type != QNN_DATATYPE_INT_64 &&
                  idx_type != QNN_DATATYPE_UINT_32,
              "OneHot: indices must be INT_32, INT_64, or UINT_32.");

    // Reject unsupported values types (string and complex have no QNN equivalent).
    TensorInfo values_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[2], values_info));
    RETURN_IF(values_info.qnn_data_type == QNN_DATATYPE_UNDEFINED,
              "OneHot: unsupported data type for values (string/complex not supported).");
  }

  // Only input[0] (indices) is passed as a QNN tensor input.
  // input[1] (depth) and input[2] (values) become QNN parameters.
  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));

  // If indices are INT_64, insert a runtime Cast(INT_32) node.
  const std::string& indices_name = inputs[0].name;
  if (qnn_model_wrapper.IsQnnTensorWrapperExist(indices_name)) {
    auto& indices_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(indices_name);
    if (indices_wrapper.GetTensorDataType() == QNN_DATATYPE_INT_64) {
      const std::string cast_name = utils::UniqueNameGenerator().New(indices_name, "_cast_int32");
      std::vector<uint32_t> indices_shape;
      RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].shape, indices_shape),
                    "OneHot: cannot get shape of indices input.");
      RETURN_IF_ERROR(qnn_model_wrapper.AddCastNode(
          utils::UniqueNameGenerator().New(indices_name, QNN_OP_CAST),
          indices_name,
          cast_name,
          QNN_TENSOR_TYPE_NATIVE,
          QNN_DATATYPE_INT_32,
          QnnQuantParamsWrapper(),
          std::move(indices_shape),
          do_op_validation));
      input_names.back() = cast_name;
    }
  }

  return Ort::Status();
}

Ort::Status OneHotOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                         const OrtNodeUnit& node_unit,
                                                         std::vector<std::string>&& input_names,
                                                         const Ort::Logger& logger,
                                                         bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  std::vector<std::string> param_tensor_names;

  // -----------------------------------------------------------------------
  // Parameter 1: depth (UINT_32 scalar) — extracted from input[1] initializer
  // -----------------------------------------------------------------------
  {
    TensorInfo depth_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], depth_info));
    RETURN_IF_NOT(depth_info.initializer_tensor != nullptr,
                  "OneHot: depth input must be a constant initializer.");

    std::vector<uint8_t> depth_bytes;
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(depth_info.initializer_tensor, depth_bytes));

    uint32_t depth_val = 0;
    switch (depth_info.qnn_data_type) {
      case QNN_DATATYPE_INT_32: {
        int32_t v = 0;
        memcpy(&v, depth_bytes.data(), sizeof(int32_t));
        RETURN_IF(v < 0, "OneHot: depth must be non-negative.");
        depth_val = static_cast<uint32_t>(v);
        break;
      }
      case QNN_DATATYPE_INT_64: {
        int64_t v = 0;
        memcpy(&v, depth_bytes.data(), sizeof(int64_t));
        RETURN_IF(v < 0, "OneHot: depth must be non-negative.");
        depth_val = static_cast<uint32_t>(v);
        break;
      }
      case QNN_DATATYPE_UINT_32: {
        memcpy(&depth_val, depth_bytes.data(), sizeof(uint32_t));
        break;
      }
      case QNN_DATATYPE_FLOAT_32: {
        float v = 0.0f;
        memcpy(&v, depth_bytes.data(), sizeof(float));
        RETURN_IF(v < 0.0f, "OneHot: depth must be non-negative.");
        depth_val = static_cast<uint32_t>(v);
        break;
      }
      case QNN_DATATYPE_FLOAT_64: {
        double v = 0.0;
        memcpy(&v, depth_bytes.data(), sizeof(double));
        RETURN_IF(v < 0.0, "OneHot: depth must be non-negative.");
        depth_val = static_cast<uint32_t>(v);
        break;
      }
      default:
        return MAKE_EP_FAIL("OneHot: unsupported data type for depth input.");
    }

    Qnn_Scalar_t depth_scalar = QNN_SCALAR_INIT;
    depth_scalar.dataType = QNN_DATATYPE_UINT_32;
    depth_scalar.uint32Value = depth_val;
    QnnParamWrapper depth_param(node_unit.Index(), node_unit.Name(),
                                QNN_OP_ONE_HOT_PARAM_DEPTH, depth_scalar);
    param_tensor_names.push_back(depth_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(depth_param));
  }

  // -----------------------------------------------------------------------
  // Parameters 2 & 3: off_value (values[0]) and on_value (values[1])
  // -----------------------------------------------------------------------
  {
    TensorInfo values_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[2], values_info));
    RETURN_IF_NOT(values_info.initializer_tensor != nullptr,
                  "OneHot: values input must be a constant initializer.");

    const size_t elem_size = utils::GetElementSizeByType(values_info.qnn_data_type);
    RETURN_IF(elem_size == 0, "OneHot: cannot determine element size for values data type.");

    // off_value is values[0]
    Qnn_Scalar_t off_scalar = QNN_SCALAR_INIT;
    RETURN_IF_ERROR(ExtractScalarFromInitializer(qnn_model_wrapper, inputs[2],
                                                 /*byte_offset=*/0, off_scalar));
    QnnParamWrapper off_param(node_unit.Index(), node_unit.Name(),
                              QNN_OP_ONE_HOT_PARAM_OFF_VALUE, off_scalar);
    param_tensor_names.push_back(off_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(off_param));

    // on_value is values[1]
    Qnn_Scalar_t on_scalar = QNN_SCALAR_INIT;
    RETURN_IF_ERROR(ExtractScalarFromInitializer(qnn_model_wrapper, inputs[2],
                                                 /*byte_offset=*/elem_size, on_scalar));
    QnnParamWrapper on_param(node_unit.Index(), node_unit.Name(),
                             QNN_OP_ONE_HOT_PARAM_ON_VALUE, on_scalar);
    param_tensor_names.push_back(on_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(on_param));
  }

  // -----------------------------------------------------------------------
  // Parameter 4: axis (optional) — ONNX default -1, QNN default N (innermost).
  // Normalise to non-negative UINT_32 for QNN.
  // -----------------------------------------------------------------------
  {
    std::vector<uint32_t> indices_shape;
    RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].shape, indices_shape),
                  "OneHot: cannot get shape of indices input.");
    const int32_t rank = static_cast<int32_t>(indices_shape.size());

    OrtNodeAttrHelper node_helper(node_unit);
    int32_t onnx_axis = node_helper.Get("axis", static_cast<int32_t>(-1));
    if (onnx_axis < 0) {
      onnx_axis += rank + 1;  // output rank = indices rank + 1
    }
    RETURN_IF(onnx_axis < 0 || onnx_axis > rank,
              "OneHot: axis out of valid range [-(rank+1), rank].");

    Qnn_Scalar_t axis_scalar = QNN_SCALAR_INIT;
    axis_scalar.dataType = QNN_DATATYPE_UINT_32;
    axis_scalar.uint32Value = static_cast<uint32_t>(onnx_axis);
    QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name(),
                               QNN_OP_ONE_HOT_PARAM_AXIS, axis_scalar);
    param_tensor_names.push_back(axis_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(axis_param));
  }

  RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit,
                                 std::move(input_names),
                                 std::move(param_tensor_names),
                                 logger, do_op_validation, QNN_OP_ONE_HOT));

  return Ort::Status();
}

void CreateOneHotOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<OneHotOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
