// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#ifdef USE_QAIRT_API

#include <algorithm>
#include <limits>

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
// Graph I/O tensors must be flagged as inputs or outputs so the backend can
// distinguish externally provided tensors from intermediate tensors.
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

Ort::Status ConvertScalarPublic(const Qnn_Scalar_t& qnn_scalar,
                                const qairt::Api& api,
                                qairt::Scalar& out) {
  return ConvertScalar(qnn_scalar, api, out);
}

Ort::Status ApplyQuantizeParams(const Qnn_QuantizeParams_t& qnn_qp,
                                const qairt::Api& api,
                                qairt::Tensor& out) {
  if (qnn_qp.encodingDefinition == QNN_DEFINITION_UNDEFINED) {
    return Ort::Status();
  }

  auto qp = api.make<qairt::QuantizeParams>();
  const auto& dimensions = out.getDimensions();
  auto block_count = [&](const uint32_t* sizes, size_t& count) -> Ort::Status {
    if (sizes == nullptr || dimensions.empty()) {
      return MAKE_EP_FAIL("Block quantization requires dimensions and block sizes.");
    }
    count = 1;
    for (size_t i = 0; i < dimensions.size(); ++i) {
      if (sizes[i] == 0 || dimensions[i] == 0) {
        return MAKE_EP_FAIL("Block quantization requires nonzero dimensions and block sizes.");
      }
      const size_t blocks = dimensions[i] / sizes[i] + (dimensions[i] % sizes[i] != 0);
      if (count > std::numeric_limits<uint32_t>::max() / blocks) {
        return MAKE_EP_FAIL("Block quantization scale count exceeds the QAIRT API limit.");
      }
      count *= blocks;
    }
    return Ort::Status();
  };

  switch (qnn_qp.quantizationEncoding) {
    case QNN_QUANTIZATION_ENCODING_UNDEFINED:
      break;

    case QNN_QUANTIZATION_ENCODING_SCALE_OFFSET: {
      const auto& so = qnn_qp.scaleOffsetEncoding;
      auto scale_offset = api.make<qairt::ScaleOffset>();
      scale_offset.setScale(so.scale);
      scale_offset.setOffset(so.offset);
      qp.setScaleOffsetEncoding(scale_offset);
      break;
    }

    case QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET: {
      const auto& aso = qnn_qp.axisScaleOffsetEncoding;
      auto axis_so = api.make<qairt::AxisScaleOffset>();
      axis_so.setAxis(aso.axis);
      std::vector<qairt::ScaleOffset> scale_offsets;
      scale_offsets.reserve(aso.numScaleOffsets);
      for (uint32_t i = 0; i < aso.numScaleOffsets; ++i) {
        auto& scale_offset = scale_offsets.emplace_back(api.make<qairt::ScaleOffset>());
        scale_offset.setScale(aso.scaleOffset[i].scale);
        scale_offset.setOffset(aso.scaleOffset[i].offset);
      }
      axis_so.setScaleOffsets(scale_offsets);
      qp.setAxisScaleOffsetEncoding(std::move(axis_so));
      break;
    }

    case QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET: {
      const auto& bso = qnn_qp.bwScaleOffsetEncoding;
      auto scale_offset = api.make<qairt::BwScaleOffset>();
      scale_offset.setBitwidth(bso.bitwidth);
      scale_offset.setScale(bso.scale);
      scale_offset.setOffset(bso.offset);
      qp.setBwScaleOffsetEncoding(scale_offset);
      break;
    }

    case QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET: {
      const auto& baso = qnn_qp.bwAxisScaleOffsetEncoding;
      std::vector<float> scales(baso.scales, baso.scales + baso.numElements);
      std::vector<int32_t> offsets(baso.numElements, 0);
      if (baso.offsets != nullptr) {
        std::copy_n(baso.offsets, baso.numElements, offsets.begin());
      }
      auto scale_offset = api.make<qairt::BwAxisScaleOffset>();
      scale_offset.setBitwidth(baso.bitwidth);
      scale_offset.setAxis(baso.axis);
      scale_offset.setScales(std::move(scales));
      scale_offset.setOffsets(std::move(offsets));
      qp.setBwAxisScaleOffsetEncoding(scale_offset);
      break;
    }

    case QNN_QUANTIZATION_ENCODING_BLOCK: {
      const auto& blk = qnn_qp.blockEncoding;
      size_t count;
      QAIRT_RETURN_IF_ERROR(block_count(blk.blockSize, count));
      if (!blk.scaleOffset) return MAKE_EP_FAIL("Missing block scale offsets.");
      auto encoding = api.make<qairt::BlockEncoding>();
      encoding.setBlockSizes(std::vector<uint32_t>(blk.blockSize, blk.blockSize + dimensions.size()));
      std::vector<qairt::ScaleOffset> scales;
      scales.reserve(count);
      for (size_t i = 0; i < count; ++i) {
        auto& scale = scales.emplace_back(api.make<qairt::ScaleOffset>());
        scale.setScale(blk.scaleOffset[i].scale);
        scale.setOffset(blk.scaleOffset[i].offset);
      }
      encoding.setScaleOffsets(std::move(scales));
      qp.setBlockEncoding(std::move(encoding));
      break;
    }

    case QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK: {
      const auto& bfb = qnn_qp.bwFloatBlockEncoding;
      size_t count;
      QAIRT_RETURN_IF_ERROR(block_count(bfb.blockSize, count));
      if (!bfb.floatScaleOffset) return MAKE_EP_FAIL("Missing float block scale offsets.");
      auto encoding = api.make<qairt::BwFloatBlockEncoding>();
      encoding.setBitwidth(bfb.bitwidth);
      encoding.setBlockSizes(std::vector<uint32_t>(bfb.blockSize, bfb.blockSize + dimensions.size()));
      std::vector<qairt::FloatScaleOffset> scales;
      scales.reserve(count);
      for (size_t i = 0; i < count; ++i) {
        auto& scale = scales.emplace_back(api.make<qairt::FloatScaleOffset>());
        scale.setScale(bfb.floatScaleOffset[i].scale);
        scale.setOffset(bfb.floatScaleOffset[i].offset);
      }
      encoding.setFloatScaleOffsets(std::move(scales));
      qp.setBwFloatBlockEncoding(std::move(encoding));
      break;
    }

    case QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION: {
      const auto* bwe = qnn_qp.blockwiseExpansion;
      if (!bwe || bwe->axis < 0 || static_cast<size_t>(bwe->axis) >= dimensions.size() ||
          !bwe->scaleOffsets || bwe->numBlocksPerAxis == 0) {
        return MAKE_EP_FAIL("Invalid blockwise expansion parameters.");
      }
      const size_t axis_size = dimensions[bwe->axis];
      if (axis_size == 0 || axis_size > std::numeric_limits<uint32_t>::max() / bwe->numBlocksPerAxis) {
        return MAKE_EP_FAIL("Invalid blockwise expansion scale count.");
      }
      const size_t count = axis_size * bwe->numBlocksPerAxis;
      auto encoding = api.make<qairt::BlockwiseExpansion>();
      encoding.setAxis(bwe->axis);
      encoding.setNumBlocksPerAxis(bwe->numBlocksPerAxis);
      encoding.setBlockScaleBitwidth(bwe->blockScaleBitwidth);
      std::vector<qairt::ScaleOffset> scales;
      scales.reserve(axis_size);
      for (size_t i = 0; i < axis_size; ++i) {
        auto& scale = scales.emplace_back(api.make<qairt::ScaleOffset>());
        scale.setScale(bwe->scaleOffsets[i].scale);
        scale.setOffset(bwe->scaleOffsets[i].offset);
      }
      encoding.setScaleOffsets(scales);
      if (bwe->blockScaleStorageType == QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_8 && bwe->blocksScale8) {
        encoding.setBlocksScale8(std::vector<uint8_t>(bwe->blocksScale8, bwe->blocksScale8 + count));
      } else if (bwe->blockScaleStorageType == QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_16 && bwe->blocksScale16) {
        encoding.setBlocksScale16(std::vector<uint16_t>(bwe->blocksScale16, bwe->blocksScale16 + count));
      } else {
        return MAKE_EP_FAIL("Invalid blockwise expansion scale storage.");
      }
      qp.setBlockwiseExpansion(std::move(encoding));
      break;
    }

    default:
      return MAKE_EP_FAIL("qairt_convert::ApplyQuantizeParams: unknown quantization encoding");
  }

  out.setQuantizeParams(qp);
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

  // Establish RAW memory even when graph I/O has no data bound yet. Accessing
  // getClientBuffer() on an uninitialized TensorMemory does not create a binding.
  Qnn_TensorMemType_t mem_type = GetQnnTensorMemType(qnn_tensor);
  if (mem_type == QNN_TENSORMEMTYPE_RAW) {
    const auto& buf = GetQnnTensorClientBuf(qnn_tensor);
    auto client_buffer = api.make<qairt::ClientBuffer>();
    client_buffer.setData(buf.data);
    client_buffer.setDataSize(buf.dataSize);
    auto memory = api.make<qairt::TensorMemory>();
    memory.setClientBuffer(client_buffer);
    out.setTensorMemory(memory);
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
      QAIRT_RETURN_IF_ERROR(FromQnnTensor(qp.tensorParam, api, tensor));
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
    QAIRT_RETURN_IF_ERROR(FromQnnTensor(v1.inputTensors[i], api, t));
    inputs.push_back(std::move(t));
  }
  out.setInputs(inputs);

  // Convert output tensors — same: ID-only refs
  std::vector<qairt::Tensor> outputs;
  outputs.reserve(v1.numOfOutputs);
  for (uint32_t i = 0; i < v1.numOfOutputs; ++i) {
    qairt::Tensor t;
    QAIRT_RETURN_IF_ERROR(FromQnnTensor(v1.outputTensors[i], api, t));
    outputs.push_back(std::move(t));
  }
  out.setOutputs(outputs);

  return Ort::Status();
}

}  // namespace qairt_convert
}  // namespace qnn
}  // namespace onnxruntime

#endif  // USE_QAIRT_API
