// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#ifdef USE_QAIRT_API

#include "core/providers/qnn/builder/qairt_type_convert.h"
#include "core/providers/qnn/builder/qnn_def.h"

#define QAIRT_RETURN_IF_ERROR(expr) \
  do {                              \
    auto _s = (expr);               \
    if (!_s.IsOK()) return _s;      \
  } while (0)

namespace onnxruntime {
namespace qnn {
namespace qairt_convert {

namespace {

// Maps Qnn_TensorType_t to qairt::TensorProperties flags.
// ponytail: QAIRT Graph::addNode requires native inputs to be outputs of a prior node
// (QairtGraph.hpp line 501). Graph I/O tensors MUST be flagged isInput/isOutput so
// the backend knows they're externally provided, not intermediate.
//
// IMPORTANT: caller must set an explicit TensorProperties on `out` via
// setTensorProperties(api.make<TensorProperties>()) BEFORE calling this function.
void ApplyTensorType(Qnn_TensorType_t type, qairt::TensorProperties& props) {
  switch (type) {
    case QNN_TENSOR_TYPE_APP_WRITE:
      props.setIsInput(true);
      break;
    case QNN_TENSOR_TYPE_APP_READ:
      props.setIsOutput(true);
      break;
    case QNN_TENSOR_TYPE_APP_READWRITE:
      props.setIsInput(true);
      props.setIsOutput(true);
      break;
    case QNN_TENSOR_TYPE_NATIVE:
      props.setIsNative(true);
      break;
    case QNN_TENSOR_TYPE_STATIC:
      props.setIsStatic(true);
      break;
    case QNN_TENSOR_TYPE_UPDATEABLE_STATIC:
      props.setIsStatic(true);
      props.setIsUpdatable(true);
      break;
    case QNN_TENSOR_TYPE_NULL:
      props.setIsNull(true);
      break;
    default:
      props.setIsNative(true);
      break;
  }
}

// Converts a QNN C Qnn_Scalar_t to a qairt::Scalar via the Api factory.
Ort::Status ConvertScalar(const Qnn_Scalar_t& qnn_scalar, const qairt::Api& api, qairt::Scalar& out) {
  out = api.make<qairt::Scalar>();
  switch (qnn_scalar.dataType) {
    case QNN_DATATYPE_INT_8:
      out.setInt8Value(qnn_scalar.int8Value);
      break;
    case QNN_DATATYPE_INT_16:
      out.setInt16Value(qnn_scalar.int16Value);
      break;
    case QNN_DATATYPE_INT_32:
      out.setInt32Value(qnn_scalar.int32Value);
      break;
    case QNN_DATATYPE_INT_64:
      out.setInt64Value(qnn_scalar.int64Value);
      break;
    case QNN_DATATYPE_UINT_8:
      out.setUint8Value(qnn_scalar.uint8Value);
      break;
    case QNN_DATATYPE_UINT_16:
      out.setUint16Value(qnn_scalar.uint16Value);
      break;
    case QNN_DATATYPE_UINT_32:
      out.setUint32Value(qnn_scalar.uint32Value);
      break;
    case QNN_DATATYPE_UINT_64:
      out.setUint64Value(qnn_scalar.uint64Value);
      break;
    case QNN_DATATYPE_FLOAT_32:
      out.setFloatValue(qnn_scalar.floatValue);
      break;
    case QNN_DATATYPE_FLOAT_64:
      out.setDoubleValue(qnn_scalar.doubleValue);
      break;
    case QNN_DATATYPE_BOOL_8:
      out.setBoolValue(qnn_scalar.bool8Value != 0);
      break;
    case QNN_DATATYPE_STRING:
      out.setStringValue(qnn_scalar.stringValue);
      break;
    default:
      return MAKE_EP_FAIL("qairt_convert::ConvertScalar: unsupported data type");
  }
  return Ort::Status();
}

}  // namespace

// ponytail: ID-only tensor ref for addNode OpConfig. Backend matches by ID to
// the tensor registered at createGraphTensor time.
// Do NOT set any TensorProperties — the default-constructed tensor has no type flags,
// which is what the backend expects for OpConfig refs (it uses the registered tensor's type).
Ort::Status FromQnnTensorRef(const Qnn_Tensor_t& qnn_tensor,
                             const qairt::Api& api,
                             qairt::Tensor& out) {
  out = api.make<qairt::Tensor>();
  out.setId(GetQnnTensorID(qnn_tensor));
  const char* name = GetQnnTensorName(qnn_tensor);
  if (name) out.setName(name);
  out.setDataType(ToQairtDataType(GetQnnTensorDataType(qnn_tensor)));

  uint32_t rank = GetQnnTensorRank(qnn_tensor);
  uint32_t* dims = GetQnnTensorDims(qnn_tensor);
  if (rank > 0 && dims) {
    out.setDimensions(std::vector<uint32_t>(dims, dims + rank));
  }
  return Ort::Status();
}

Ort::Status ConvertScalarPublic(const Qnn_Scalar_t& qnn_scalar,
                                const qairt::Api& api,
                                qairt::Scalar& out) {
  return ConvertScalar(qnn_scalar, api, out);
}

Ort::Status ApplyQuantizeParams(const Qnn_QuantizeParams_t& qnn_qp,
                                const qairt::Api& /*api*/,
                                qairt::Tensor& out) {
  if (qnn_qp.encodingDefinition == QNN_DEFINITION_UNDEFINED) {
    return Ort::Status();
  }

  auto& qp = out.getQuantizeParams();

  switch (qnn_qp.quantizationEncoding) {
    case QNN_QUANTIZATION_ENCODING_UNDEFINED:
      break;

    case QNN_QUANTIZATION_ENCODING_SCALE_OFFSET: {
      const auto& so = qnn_qp.scaleOffsetEncoding;
      qp.setScaleOffsetEncoding(qairt::ScaleOffset(so.scale, so.offset));
      break;
    }

    case QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET: {
      const auto& aso = qnn_qp.axisScaleOffsetEncoding;
      qairt::AxisScaleOffset axis_so;
      axis_so.setAxis(aso.axis);
      std::vector<qairt::ScaleOffset> scale_offsets;
      scale_offsets.reserve(aso.numScaleOffsets);
      for (uint32_t i = 0; i < aso.numScaleOffsets; ++i) {
        scale_offsets.emplace_back(aso.scaleOffset[i].scale, aso.scaleOffset[i].offset);
      }
      axis_so.setScaleOffsets(scale_offsets);
      qp.setAxisScaleOffsetEncoding(std::move(axis_so));
      break;
    }

    case QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET: {
      const auto& bso = qnn_qp.bwScaleOffsetEncoding;
      qp.setBwScaleOffsetEncoding(qairt::BwScaleOffset(bso.bitwidth, bso.scale, bso.offset));
      break;
    }

    case QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET: {
      const auto& baso = qnn_qp.bwAxisScaleOffsetEncoding;
      std::vector<float> scales(baso.scales, baso.scales + baso.numElements);
      std::vector<int32_t> offsets(baso.offsets, baso.offsets + baso.numElements);
      qp.setBwAxisScaleOffsetEncoding(
          qairt::BwAxisScaleOffset(baso.bitwidth, baso.axis, scales, offsets));
      break;
    }

    case QNN_QUANTIZATION_ENCODING_BLOCK: {
      const auto& blk = qnn_qp.blockEncoding;
      // ponytail: block encoding has blockSize array + scaleOffset array.
      // QAIRT BlockEncoding API not fully explored — use shallowCopy pattern.
      // Deferring full implementation until we have a model that actually uses BLOCK encoding
      // through this path. EP uses it only for weight tensors which are pre-quantized.
      (void)blk;
      return MAKE_EP_FAIL("qairt_convert::ApplyQuantizeParams: BLOCK encoding not yet implemented");
    }

    case QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK: {
      const auto& bfb = qnn_qp.bwFloatBlockEncoding;
      (void)bfb;
      return MAKE_EP_FAIL("qairt_convert::ApplyQuantizeParams: BW_FLOAT_BLOCK encoding not yet implemented");
    }

    case QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION: {
      const auto* bwe = qnn_qp.blockwiseExpansion;
      (void)bwe;
      return MAKE_EP_FAIL("qairt_convert::ApplyQuantizeParams: BLOCKWISE_EXPANSION encoding not yet implemented");
    }

    default:
      return MAKE_EP_FAIL("qairt_convert::ApplyQuantizeParams: unknown quantization encoding");
  }

  return Ort::Status();
}

Ort::Status FromQnnTensor(const Qnn_Tensor_t& qnn_tensor,
                          const qairt::Api& api,
                          qairt::Tensor& out) {
  out = api.make<qairt::Tensor>();

  const char* name = GetQnnTensorName(qnn_tensor);
  if (name) {
    out.setName(name);
  }

  out.setDataType(ToQairtDataType(GetQnnTensorDataType(qnn_tensor)));
  out.setDataFormat(GetQnnTensorDataFormat(qnn_tensor));
  out.setId(GetQnnTensorID(qnn_tensor));

  // ponytail: must create explicit TensorProperties and assign BEFORE setting flags.
  // A freshly-constructed Tensor's internal props handle is NULL — the non-owning
  // accessor wraps NULL and flag setters become no-ops.
  auto props = api.make<qairt::TensorProperties>();
  ApplyTensorType(GetQnnTensorType(qnn_tensor), props);
  out.setTensorProperties(props);

  uint32_t rank = GetQnnTensorRank(qnn_tensor);
  uint32_t* dims = GetQnnTensorDims(qnn_tensor);
  if (rank > 0 && dims) {
    out.setDimensions(std::vector<uint32_t>(dims, dims + rank));
  }

  // ponytail: attach client buffer for STATIC (weight data at graph build) and
  // APP_WRITE/APP_READ (inference data at execute time). Only skip for NATIVE
  // intermediate tensors which have no external buffer.
  Qnn_TensorType_t tensor_type = GetQnnTensorType(qnn_tensor);
  Qnn_TensorMemType_t mem_type = GetQnnTensorMemType(qnn_tensor);
  bool needs_buffer = (tensor_type != QNN_TENSOR_TYPE_NATIVE &&
                       tensor_type != QNN_TENSOR_TYPE_NULL);
  if (needs_buffer && mem_type == QNN_TENSORMEMTYPE_RAW) {
    const auto& buf = GetQnnTensorClientBuf(qnn_tensor);
    if (buf.data && buf.dataSize > 0) {
      auto& cb = out.getClientBuffer();
      cb.setData(const_cast<void*>(static_cast<const void*>(buf.data)));
      cb.setDataSize(buf.dataSize);
    }
  }

  QAIRT_RETURN_IF_ERROR(ApplyQuantizeParams(GetQnnTensorQParams(qnn_tensor), api, out));

  return Ort::Status();
}

Ort::Status FromQnnOpConfig(const Qnn_OpConfig_t& qnn_op,
                            const qairt::Api& api,
                            qairt::OpConfig& out) {
  out = api.make<qairt::OpConfig>();

  const auto& v1 = qnn_op.v1;
  if (v1.name) out.setName(v1.name);
  if (v1.packageName) out.setPackageName(v1.packageName);
  if (v1.typeName) out.setTypeName(v1.typeName);

  // Convert params
  std::vector<qairt::Param> params;
  params.reserve(v1.numOfParams);
  for (uint32_t i = 0; i < v1.numOfParams; ++i) {
    const auto& qp = v1.params[i];
    auto param = api.make<qairt::Param>();
    if (qp.name) param.setName(qp.name);

    if (qp.paramType == QNN_PARAMTYPE_SCALAR) {
      qairt::Scalar scalar;
      QAIRT_RETURN_IF_ERROR(ConvertScalar(qp.scalarParam, api, scalar));
      param.setScalar(scalar);
    } else if (qp.paramType == QNN_PARAMTYPE_TENSOR) {
      qairt::Tensor tensor;
      QAIRT_RETURN_IF_ERROR(FromQnnTensorRef(qp.tensorParam, api, tensor));
      param.setTensor(tensor);
    }
    params.push_back(std::move(param));
  }
  out.setParams(params);

  // Convert input tensors — ID-only refs, no type properties (QAIRT addNode rejects them)
  std::vector<qairt::Tensor> inputs;
  inputs.reserve(v1.numOfInputs);
  for (uint32_t i = 0; i < v1.numOfInputs; ++i) {
    qairt::Tensor t;
    QAIRT_RETURN_IF_ERROR(FromQnnTensorRef(v1.inputTensors[i], api, t));
    inputs.push_back(std::move(t));
  }
  out.setInputs(inputs);

  // Convert output tensors — same: ID-only refs
  std::vector<qairt::Tensor> outputs;
  outputs.reserve(v1.numOfOutputs);
  for (uint32_t i = 0; i < v1.numOfOutputs; ++i) {
    qairt::Tensor t;
    QAIRT_RETURN_IF_ERROR(FromQnnTensorRef(v1.outputTensors[i], api, t));
    outputs.push_back(std::move(t));
  }
  out.setOutputs(outputs);

  return Ort::Status();
}

}  // namespace qairt_convert
}  // namespace qnn
}  // namespace onnxruntime

#endif  // USE_QAIRT_API
