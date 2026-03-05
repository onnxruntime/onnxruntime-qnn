// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/unittest_util/graph_transform_test_builder.h"

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>

#include "core/common/inlined_containers_fwd.h"
#include "core/common/span_utils.h"
#include "core/graph/model.h"
#include "core/session/inference_session.h"
#include "test/compare_ortvalue.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/test_utils.h"

// enable to dump model for debugging
#define SAVE_TEST_GRAPH 0

namespace onnxruntime {
namespace test {

static InlinedVector<std::byte> GetZeroPointBytes(int64_t zero_point, ONNX_NAMESPACE::TensorProto_DataType type) {
  switch (type) {
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
      int8_t val = static_cast<int8_t>(zero_point);
      auto span = gsl::as_bytes(gsl::make_span(&val, 1));
      return InlinedVector<std::byte>(span.begin(), span.end());
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {
      uint8_t val = static_cast<uint8_t>(zero_point);
      auto span = gsl::as_bytes(gsl::make_span(&val, 1));
      return InlinedVector<std::byte>(span.begin(), span.end());
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT16: {
      int16_t val = static_cast<int16_t>(zero_point);
      auto span = gsl::as_bytes(gsl::make_span(&val, 1));
      return InlinedVector<std::byte>(span.begin(), span.end());
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16: {
      uint16_t val = static_cast<uint16_t>(zero_point);
      auto span = gsl::as_bytes(gsl::make_span(&val, 1));
      return InlinedVector<std::byte>(span.begin(), span.end());
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
      int32_t val = static_cast<int32_t>(zero_point);
      auto span = gsl::as_bytes(gsl::make_span(&val, 1));
      return InlinedVector<std::byte>(span.begin(), span.end());
    }
    default:
      throw std::runtime_error("Unhandled zero-point type " + std::to_string(type) + ".");
  }
}

const ONNX_NAMESPACE::TensorProto* ModelTestBuilder::MakeInitializer(std::string name,
                                                                     gsl::span<const int64_t> shape,
                                                                     ONNX_NAMESPACE::TensorProto_DataType elem_type,
                                                                     gsl::span<const std::byte> raw_data) {
  ONNX_NAMESPACE::TensorProto* tensor_proto = graph_->add_initializer();
  tensor_proto->set_name(name);
  tensor_proto->set_data_type(elem_type);
  utils::SetRawDataInTensorProto(
      *tensor_proto,
      raw_data.data(),
      raw_data.size());

  for (auto& dim : shape) {
    tensor_proto->add_dims(dim);
  }

  return tensor_proto;
}

const ONNX_NAMESPACE::NodeProto* ModelTestBuilder::AddQuantizeLinearNode(const std::string& node_name,
                                                                         const std::string& input_name,
                                                                         float input_scale,
                                                                         int64_t input_zero_point,
                                                                         ONNX_NAMESPACE::TensorProto_DataType zero_point_type,
                                                                         const std::string& output_name,
                                                                         bool use_ms_domain) {
  std::vector<std::string> input_names;
  input_names.push_back(input_name);

  auto scale = MakeScalarInitializer<float>(node_name + "_inp_scale", input_scale);
  input_names.push_back(scale->name());

  InlinedVector<std::byte> zp_bytes = GetZeroPointBytes(input_zero_point, zero_point_type);
  auto zp = MakeInitializer(node_name + "_inp_zp", {}, zero_point_type, zp_bytes);
  input_names.push_back(zp->name());

  std::string domain = use_ms_domain ? kMSDomain : "";
  return AddNode(node_name, "QuantizeLinear", input_names, {output_name}, domain);
}

const ONNX_NAMESPACE::NodeProto* ModelTestBuilder::AddDequantizeLinearNode(const std::string& node_name,
                                                                           const std::string& input_name,
                                                                           float input_scale,
                                                                           int64_t input_zero_point,
                                                                           ONNX_NAMESPACE::TensorProto_DataType zero_point_type,
                                                                           const std::string& output_name,
                                                                           bool use_ms_domain) {
  std::vector<std::string> input_names;
  input_names.push_back(input_name);

  auto scale = MakeScalarInitializer<float>(node_name + "_inp_scale", input_scale);
  input_names.push_back(scale->name());

  InlinedVector<std::byte> zp_bytes = GetZeroPointBytes(input_zero_point, zero_point_type);
  auto zp = MakeInitializer(node_name + "_inp_zp", {}, zero_point_type, zp_bytes);
  input_names.push_back(zp->name());

  std::string domain = use_ms_domain ? kMSDomain : "";
  return AddNode(node_name, "DequantizeLinear", input_names, {output_name}, domain);
}

}  // namespace test
}  // namespace onnxruntime
