// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <functional>
#include <type_traits>
#include <vector>
#include <random>
#include "gtest/gtest.h"

#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/test_random_seed.h"

namespace onnxruntime {
namespace test {

constexpr const char* kMSDomain = "com.microsoft";

template <typename T>
struct IsByteType : std::false_type {};

template <>
struct IsByteType<uint8_t> : std::true_type {};

template <>
struct IsByteType<int8_t> : std::true_type {};

class RandomValueGenerator {
 public:
  using RandomEngine = std::default_random_engine;
  using RandomSeedType = RandomEngine::result_type;

  explicit RandomValueGenerator(std::optional<RandomSeedType> seed = {})
      : random_seed_{seed.has_value() ? *seed : static_cast<RandomSeedType>(GetTestRandomSeed())},
        generator_{random_seed_},
        output_trace_{__FILE__, __LINE__, "ORT test random seed: " + std::to_string(random_seed_)} {
  }

  RandomSeedType GetRandomSeed() const {
    return random_seed_;
  }

  // Random values generated are in the range [min, max).
  template <typename TFloat>
  typename std::enable_if<
      std::is_floating_point<TFloat>::value,
      std::vector<TFloat>>::type
  Uniform(gsl::span<const int64_t> dims, TFloat min, TFloat max) {
    std::vector<TFloat> val(SizeFromDims(dims));
    std::uniform_real_distribution<TFloat> distribution(min, max);
    for (size_t i = 0; i < val.size(); ++i) {
      val[i] = distribution(generator_);
    }
    return val;
  }

  // Random values generated are in the range [min, max).
  template <typename TFloat16>
  typename std::enable_if<
      std::is_same_v<TFloat16, Ort::Float16_t> || std::is_same_v<TFloat16, Ort::BFloat16_t>,
      std::vector<TFloat16>>::type
  Uniform(gsl::span<const int64_t> dims, TFloat16 min, TFloat16 max) {
    std::vector<TFloat16> val(SizeFromDims(dims));
    std::uniform_real_distribution<float> distribution(static_cast<float>(min), static_cast<float>(max));
    for (size_t i = 0; i < val.size(); ++i) {
      val[i] = TFloat16(static_cast<float>(distribution(generator_)));
    }
    return val;
  }

  // Random values generated are in the range [min, max).
  template <typename TInt>
  typename std::enable_if<
      std::is_integral<TInt>::value && !IsByteType<TInt>::value,
      std::vector<TInt>>::type
  Uniform(gsl::span<const int64_t> dims, TInt min, TInt max) {
    std::vector<TInt> val(SizeFromDims(dims));
    std::uniform_int_distribution<TInt> distribution(min, max - 1);
    for (size_t i = 0; i < val.size(); ++i) {
      val[i] = distribution(generator_);
    }
    return val;
  }

  template <typename TByte>
  typename std::enable_if<
      IsByteType<TByte>::value,
      std::vector<TByte>>::type
  Uniform(gsl::span<const int64_t> dims, TByte min, TByte max) {
    std::vector<TByte> val(SizeFromDims(dims));
    std::uniform_int_distribution<int32_t> distribution(min, max - 1);
    for (size_t i = 0; i < val.size(); ++i) {
      val[i] = static_cast<TByte>(distribution(generator_));
    }
    return val;
  }

  template <typename TInt4>
  typename std::enable_if<
      std::is_same_v<TInt4, Int4x2> || std::is_same_v<TInt4, UInt4x2>,
      std::vector<TInt4>>::type
  Uniform(gsl::span<const int64_t> dims, TInt4 min, TInt4 max) {
    using UnpackedType = typename TInt4::UnpackedType;
    std::vector<UnpackedType> data_int8 = Uniform<UnpackedType>(dims, min.GetElem(0), max.GetElem(0));
    std::vector<TInt4> data(TInt4::CalcNumInt4Pairs(data_int8.size()));
    for (size_t i = 0; i < data_int8.size(); i++) {
      size_t r = i >> 1;
      size_t c = i & 0x1;
      data[r].SetElem(c, data_int8[i]);
    }
    return data;
  }

  // Gaussian distribution for float
  template <typename TFloat>
  typename std::enable_if<
      std::is_floating_point<TFloat>::value,
      std::vector<TFloat>>::type
  Gaussian(gsl::span<const int64_t> dims, TFloat mean, TFloat stddev) {
    std::vector<TFloat> val(SizeFromDims(dims));
    std::normal_distribution<TFloat> distribution(mean, stddev);
    for (size_t i = 0; i < val.size(); ++i) {
      val[i] = distribution(generator_);
    }
    return val;
  }

  // Gaussian distribution for Integer
  template <typename TInt>
  typename std::enable_if<
      std::is_integral<TInt>::value,
      std::vector<TInt>>::type
  Gaussian(const std::vector<int64_t>& dims, TInt mean, TInt stddev) {
    std::vector<TInt> val(SizeFromDims(dims));
    std::normal_distribution<float> distribution(static_cast<float>(mean), static_cast<float>(stddev));
    for (size_t i = 0; i < val.size(); ++i) {
      val[i] = static_cast<TInt>(std::round(distribution(generator_)));
    }
    return val;
  }

  // Gaussian distribution for Integer and Clamp to [min, max]
  template <typename TInt>
  typename std::enable_if<
      std::is_integral<TInt>::value,
      std::vector<TInt>>::type
  Gaussian(const std::vector<int64_t>& dims, TInt mean, TInt stddev, TInt min, TInt max) {
    std::vector<TInt> val(SizeFromDims(dims));
    std::normal_distribution<float> distribution(static_cast<float>(mean), static_cast<float>(stddev));
    for (size_t i = 0; i < val.size(); ++i) {
      int64_t round_val = static_cast<int64_t>(std::round(distribution(generator_)));
      val[i] = static_cast<TInt>(std::min<int64_t>(std::max<int64_t>(round_val, min), max));
    }
    return val;
  }

  template <class T>
  inline std::vector<T> OneHot(const std::vector<int64_t>& dims, int64_t stride) {
    std::vector<T> val(SizeFromDims(dims), T(0));
    std::uniform_int_distribution<int64_t> distribution(0, stride - 1);
    for (size_t offset = 0; offset < val.size(); offset += stride) {
      size_t rand_index = static_cast<size_t>(distribution(generator_));
      val[offset + rand_index] = T(1);
    }
    return val;
  }

 private:
  const RandomSeedType random_seed_;
  RandomEngine generator_;
  // while this instance is in scope, output some context information on test failure like the random seed value
  const ::testing::ScopedTrace output_trace_;
  inline size_t SizeFromDims(gsl::span<const int64_t> dims, gsl::span<const int64_t> strides = {}) {
    int64_t size = 1;
    if (strides.empty()) {
      size = std::accumulate(dims.begin(), dims.end(), static_cast<int64_t>(1), std::multiplies<int64_t>{});
    } else {
      assert(dims.size() == strides.size());
      for (size_t dim = 0; dim < dims.size(); ++dim) {
        if (dims[dim] == 0) {
          size = 0;
          break;
        }
        size += strides[dim] * (dims[dim] - 1);
      }
    }

    return static_cast<size_t>(size);
  }
};

template <typename T>
struct IsTypeQuantLinearCompatible : IsByteType<T> {};

template <>
struct IsTypeQuantLinearCompatible<int16_t> : std::true_type {};

template <>
struct IsTypeQuantLinearCompatible<uint16_t> : std::true_type {};

template <>
struct IsTypeQuantLinearCompatible<Int4x2> : std::true_type {};

template <>
struct IsTypeQuantLinearCompatible<UInt4x2> : std::true_type {};

template <typename T>
struct IsTypeDequantLinearCompatible : IsByteType<T> {};

template <>
struct IsTypeDequantLinearCompatible<int16_t> : std::true_type {};

template <>
struct IsTypeDequantLinearCompatible<uint16_t> : std::true_type {};

template <>
struct IsTypeDequantLinearCompatible<int32_t> : std::true_type {};

template <>
struct IsTypeDequantLinearCompatible<Int4x2> : std::true_type {};

template <>
struct IsTypeDequantLinearCompatible<UInt4x2> : std::true_type {};

class ModelTestBuilder {
 public:
  ModelTestBuilder() {
    return;
  }

  template <typename T>
  const ONNX_NAMESPACE::ValueInfoProto* MakeInput(const std::string& name,
                                                  const std::vector<int64_t>& shape,
                                                  const std::vector<T>& data,
                                                  AllocatorPtr /* allocator */ = nullptr) {
    ONNX_NAMESPACE::ValueInfoProto* inp = graph_->add_input();
    inp->set_name(name);
    ONNX_NAMESPACE::TypeProto* type_proto = inp->mutable_type();

    type_proto->mutable_tensor_type()->set_elem_type(ToTensorProtoElementType<T>());
    // Set shape even if no dims (for scalar)
    type_proto->mutable_tensor_type()->mutable_shape();
    for (auto& dim : shape) {
      type_proto->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
    }

    Ort::Value input_value;
    CreateMLValue<T>(nullptr,
                     shape,
                     data,
                     input_value);
    feeds_.emplace(name, std::move(input_value));

    return inp;
  }

  template <typename T>
  const ONNX_NAMESPACE::ValueInfoProto* MakeInput(const std::string& name,
                                                  const std::vector<int64_t>& shape, T min, T max,
                                                  AllocatorPtr allocator = nullptr) {
    return MakeInput<T>(name, shape, rand_gen_.Uniform<T>(shape, min, max), allocator);
  }

  const ONNX_NAMESPACE::ValueInfoProto* MakeInputBool(const std::string& name,
                                                      const std::vector<int64_t>& shape, AllocatorPtr allocator = nullptr) {
    std::vector<uint8_t> data_uint8 = rand_gen_.Uniform<uint8_t>(shape, 0, 1);
    std::vector<bool> data;
    for (uint8_t x : data_uint8) {
      data.push_back(x != 0);
    }
    return MakeInput<bool>(name, shape, data, allocator);
  }

  const ONNX_NAMESPACE::ValueInfoProto* MakeOutput(const std::string& name) {
    ONNX_NAMESPACE::ValueInfoProto* out = graph_->add_output();
    out->set_name(name);
    return out;
  }

  template <typename T>
  const ONNX_NAMESPACE::ValueInfoProto* MakeOutput(const std::string& name,
                                                   const std::optional<std::vector<int64_t>>& shape) {
    ONNX_NAMESPACE::ValueInfoProto* out = graph_->add_output();
    out->set_name(name);
    ONNX_NAMESPACE::TypeProto* type_proto = out->mutable_type();
    type_proto->mutable_tensor_type()->set_elem_type(ToTensorProtoElementType<T>());

    if (shape != std::nullopt) {
      ONNX_NAMESPACE::TensorShapeProto* shape_proto = type_proto->mutable_tensor_type()->mutable_shape();
      for (auto& d : *shape) {
        auto dim = shape_proto->add_dim();
        if (d != -1) {
          dim->set_dim_value(d);
        }
      }
    }

    return out;
  }

  /// <summary>
  /// Makes an initializer from the provided shape, element type, and raw data bytes.
  /// </summary>
  /// <param name="shape">Initializer shape</param>
  /// <param name="elem_type">ONNX tensor element data type</param>
  /// <param name="raw_data">Raw data bytes</param>
  /// <returns>ValueInfo pointer for the initializer</returns>
  const ONNX_NAMESPACE::TensorProto* MakeInitializer(const std::string& name,
                                                     gsl::span<const int64_t> shape,
                                                     ONNX_NAMESPACE::TensorProto_DataType elem_type,
                                                     gsl::span<const std::byte> raw_data);

  template <typename T>
  const ONNX_NAMESPACE::TensorProto* MakeInitializer(const std::string& name,
                                                     const std::vector<int64_t>& shape,
                                                     const std::vector<T>& data) {
    gsl::span<const std::byte> raw_data = ReinterpretAsSpan<const std::byte, const T>(data);
    return MakeInitializer(name, shape, ToTensorProtoElementType<T>(), raw_data);
  }

  // Special handle for std::vector<bool>.
  const ONNX_NAMESPACE::TensorProto* MakeInitializerBool(const std::string& name,
                                                         const std::vector<int64_t>& shape, const std::vector<bool>& data) {
    ONNX_NAMESPACE::TensorProto* tensor_proto = graph_->add_initializer();
    tensor_proto->set_name(name);
    tensor_proto->set_data_type(ToTensorProtoElementType<bool>());
    std::unique_ptr<bool[]> data_buffer = std::make_unique<bool[]>(data.size());
    for (size_t i = 0; i < data.size(); ++i) data_buffer[i] = data[i];
    SetRawDataInTensorProto(
        *tensor_proto,
        data_buffer.get(),
        data.size());

    for (auto& dim : shape) {
      tensor_proto->add_dims(dim);
    }

    return tensor_proto;
  }

  const ONNX_NAMESPACE::TensorProto* MakeRandInitializerBool(const std::string& name, const std::vector<int64_t>& shape) {
    std::vector<uint8_t> data_uint8 = rand_gen_.Uniform<uint8_t>(shape, 0, 1);
    std::vector<bool> data;
    for (uint8_t x : data_uint8) {
      data.push_back(x != 0);
    }
    return MakeInitializerBool(name, shape, data);
  }

  template <typename T>
  const ONNX_NAMESPACE::TensorProto* MakeInitializer(const std::string& name,
                                                     const std::vector<int64_t>& shape, T min, T max) {
    return MakeInitializer<T>(name, shape, rand_gen_.Uniform<T>(shape, min, max));
  }

  template <typename T>
  const ONNX_NAMESPACE::TensorProto* MakeScalarInitializer(const std::string& name, T data) {
    return MakeInitializer(name, {}, std::vector<T>{data});
  }

  template <typename T>
  const ONNX_NAMESPACE::TensorProto* Make1DInitializer(const std::string& name, const std::vector<T>& data) {
    return MakeInitializer(name, {static_cast<int64_t>(data.size())}, data);
  }

  const ONNX_NAMESPACE::NodeProto* AddNode(const std::string& node_name,
                                           const std::string& op_type,
                                           const std::vector<std::string>& input_names,
                                           const std::vector<std::string>& output_names,
                                           const std::string& domain = "",
                                           const std::vector<ONNX_NAMESPACE::AttributeProto>& node_attributes = {}) {
    ONNX_NAMESPACE::NodeProto* node = graph_->add_node();
    node->set_op_type(op_type);
    node->set_name(node_name);
    for (const auto& inp_name : input_names) {
      node->add_input(inp_name);
    }
    for (const auto& out_name : output_names) {
      node->add_output(out_name);
    }
    node->set_domain(domain);
    // Add attributes to the node
    for (auto attr : node_attributes) {
      // Copy the attribute to the node
      ONNX_NAMESPACE::AttributeProto* new_attr = node->add_attribute();
      new_attr->CopyFrom(attr);
    }

    return node;
  }

  // Helper functions to create attributes
  ONNX_NAMESPACE::AttributeProto MakeScalarAttribute(const std::string& name, float value) {
    ONNX_NAMESPACE::AttributeProto attr;
    attr.set_name(name);
    attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
    attr.set_f(value);
    return attr;
  }

  ONNX_NAMESPACE::AttributeProto MakeScalarAttribute(const std::string& name, int64_t value) {
    ONNX_NAMESPACE::AttributeProto attr;
    attr.set_name(name);
    attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
    attr.set_i(value);
    return attr;
  }

  ONNX_NAMESPACE::AttributeProto MakeStringAttribute(const std::string& name, const std::string& value) {
    ONNX_NAMESPACE::AttributeProto attr;
    attr.set_name(name);
    attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
    attr.set_s(value);
    return attr;
  }

  ONNX_NAMESPACE::AttributeProto MakeFloatsAttribute(const std::string& name, const std::vector<float>& values) {
    ONNX_NAMESPACE::AttributeProto attr;
    attr.set_name(name);
    attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOATS);
    for (float value : values) {
      attr.add_floats(value);
    }
    return attr;
  }

  ONNX_NAMESPACE::AttributeProto MakeIntsAttribute(const std::string& name, const std::vector<int64_t>& values) {
    ONNX_NAMESPACE::AttributeProto attr;
    attr.set_name(name);
    attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INTS);
    for (int64_t value : values) {
      attr.add_ints(value);
    }
    return attr;
  }

  template <typename ZpType, typename ScaleType = float>
  typename std::enable_if<IsTypeQuantLinearCompatible<ZpType>::value, const ONNX_NAMESPACE::NodeProto*>::type
  AddQuantizeLinearNode(const std::string& node_name,
                        const std::string& input_name,
                        ScaleType input_scale,
                        ZpType input_zero_point,
                        const std::string& output_name,
                        bool use_ms_domain = false) {
    std::vector<std::string> input_names;
    input_names.push_back(input_name);
    auto scale = MakeScalarInitializer<ScaleType>(node_name + "_inp_scale", input_scale);
    auto zp = MakeScalarInitializer<ZpType>(node_name + "_inp_zp", input_zero_point);
    input_names.push_back(scale->name());
    input_names.push_back(zp->name());

    std::string domain = use_ms_domain ? kMSDomain : "";
    return AddNode(node_name, "QuantizeLinear", input_names, {output_name}, domain);
  }

  template <typename T>
  typename std::enable_if<IsTypeQuantLinearCompatible<T>::value, const ONNX_NAMESPACE::NodeProto*>::type
  AddQuantizeLinearNode(const std::string& node_name,
                        const std::string& input_name,
                        const std::vector<float>& input_scales,
                        const std::vector<T>& input_zero_points,
                        const std::string& output_name,
                        const std::vector<ONNX_NAMESPACE::AttributeProto>& attributes = {},
                        bool use_ms_domain = false) {
    std::vector<std::string> input_names;
    input_names.push_back(input_name);

    std::vector<int64_t> qparams_shape = {static_cast<int64_t>(input_scales.size())};
    auto scales = MakeInitializer<float>(node_name + "_inp_scale", qparams_shape, input_scales);
    auto zps = MakeInitializer<T>(node_name + "_inp_zp", qparams_shape, input_zero_points);
    input_names.push_back(scales->name());
    input_names.push_back(zps->name());

    std::string domain = use_ms_domain ? kMSDomain : "";
    return AddNode(node_name, "QuantizeLinear", input_names, {output_name}, domain, attributes);
  }

  const ONNX_NAMESPACE::NodeProto* AddQuantizeLinearNode(const std::string& node_name,
                                                         const std::string& input_name,
                                                         float input_scale,
                                                         const std::string& output_name,
                                                         bool use_ms_domain = false) {
    std::vector<std::string> input_names;
    auto scale = MakeScalarInitializer<float>(
        node_name + "_inp_scale", input_scale);
    input_names.push_back(input_name);
    input_names.push_back(scale->name());

    std::string domain = use_ms_domain ? kMSDomain : "";
    return AddNode(node_name, "QuantizeLinear", input_names, {output_name}, domain);
  }

  const ONNX_NAMESPACE::NodeProto* AddQuantizeLinearNode(const std::string& node_name,
                                                         const std::string& input_name,
                                                         const std::vector<float>& input_scales,
                                                         const std::string& output_name,
                                                         const std::vector<ONNX_NAMESPACE::AttributeProto>& attributes = {},
                                                         bool use_ms_domain = false) {
    std::vector<std::string> input_names;
    auto scale = Make1DInitializer<float>(node_name + "_inp_scale", input_scales);
    input_names.push_back(input_name);
    input_names.push_back(scale->name());

    std::string domain = use_ms_domain ? kMSDomain : "";
    return AddNode(node_name, "QuantizeLinear", input_names, {output_name}, domain, attributes);
  }

  /// <summary>
  /// Adds a Q node with a configurable zero-point type.
  /// Takes in an int64_t zero_point value, which is large enough to represent all ONNX zero-point types.
  /// </summary>
  /// <param name="input_arg">First input to the Q node</param>
  /// <param name="input_scale">Input scale value</param>
  /// <param name="input_zero_point">Input zero point value</param>
  /// <param name="zero_point_type">Input zero point's type</param>
  /// <param name="output_arg">Q node's output node arg</param>
  /// <param name="use_ms_domain">True to use the 'com.microsoft' domain</param>
  /// <returns>Reference to the new Q node</returns>
  const ONNX_NAMESPACE::NodeProto* AddQuantizeLinearNode(const std::string& node_name,
                                                         const std::string& input_name,
                                                         float input_scale,
                                                         int64_t input_zero_point,
                                                         ONNX_NAMESPACE::TensorProto_DataType zero_point_type,
                                                         const std::string& output_name,
                                                         bool use_ms_domain = false);

  template <typename ZpType, typename ScaleType = float>
  typename std::enable_if<IsTypeDequantLinearCompatible<ZpType>::value, const ONNX_NAMESPACE::NodeProto*>::type
  AddDequantizeLinearNode(const std::string& node_name,
                          const std::string& input_name,
                          ScaleType input_scale,
                          ZpType input_zero_point,
                          const std::string& output_name,
                          bool use_ms_domain = false) {
    std::vector<std::string> input_names;
    input_names.push_back(input_name);
    auto scale = MakeScalarInitializer<ScaleType>(node_name + "_inp_scale", input_scale);
    auto zp = MakeScalarInitializer<ZpType>(node_name + "_inp_zp", input_zero_point);
    input_names.push_back(scale->name());
    input_names.push_back(zp->name());

    std::string domain = use_ms_domain ? kMSDomain : "";
    return AddNode(node_name, "DequantizeLinear", input_names, {output_name}, domain);
  }

  template <typename T>
  typename std::enable_if<IsTypeDequantLinearCompatible<T>::value, const ONNX_NAMESPACE::NodeProto*>::type
  AddDequantizeLinearNode(const std::string& node_name,
                          const std::string& input_name,
                          const std::vector<float>& input_scales,
                          const std::vector<T>& input_zero_points,
                          const std::string& output_name,
                          const std::vector<ONNX_NAMESPACE::AttributeProto>& attributes = {},
                          bool use_ms_domain = false) {
    std::vector<std::string> input_names;
    input_names.push_back(input_name);

    std::vector<int64_t> qparams_shape = {static_cast<int64_t>(input_scales.size())};
    auto scales = MakeInitializer<float>(node_name + "_inp_scale", qparams_shape, input_scales);
    auto zps = MakeInitializer<T>(node_name + "_inp_zp", qparams_shape, input_zero_points);
    input_names.push_back(scales->name());
    input_names.push_back(zps->name());

    std::string domain = use_ms_domain ? kMSDomain : "";
    return AddNode(node_name, "DequantizeLinear", input_names, {output_name}, domain, attributes);
  }

  const ONNX_NAMESPACE::NodeProto* AddDequantizeLinearNode(const std::string& node_name,
                                                           const std::string& input_name,
                                                           float input_scale,
                                                           const std::string& output_name,
                                                           bool use_ms_domain = false) {
    std::vector<std::string> input_names;
    auto scale = MakeScalarInitializer<float>(
        node_name + "_inp_scale", input_scale);
    input_names.push_back(input_name);
    input_names.push_back(scale->name());

    std::string domain = use_ms_domain ? kMSDomain : "";
    return AddNode(node_name, "DequantizeLinear", input_names, {output_name}, domain);
  }

  const ONNX_NAMESPACE::NodeProto* AddDequantizeLinearNode(const std::string& node_name,
                                                           const std::string& input_name,
                                                           const std::vector<float>& input_scales,
                                                           const std::string& output_name,
                                                           const std::vector<ONNX_NAMESPACE::AttributeProto>& attributes = {},
                                                           bool use_ms_domain = false) {
    std::vector<std::string> input_names;
    input_names.push_back(input_name);
    auto scale = Make1DInitializer<float>(node_name + "_inp_scale", input_scales);
    input_names.push_back(scale->name());

    std::string domain = use_ms_domain ? kMSDomain : "";
    return AddNode(node_name, "DequantizeLinear", input_names, {output_name}, domain, attributes);
  }

  /// <summary>
  /// Adds a DQ node with a configurable zero-point type.
  /// Takes in an int64_t zero_point value, which is large enough to represent all ONNX zero-point types.
  /// </summary>
  /// <param name="input_arg">First input to the DQ node</param>
  /// <param name="input_scale">Input scale value</param>
  /// <param name="input_zero_point">Input zero point value</param>
  /// <param name="zero_point_type">Input zero point's type</param>
  /// <param name="output_arg">DQ node's output node arg</param>
  /// <param name="use_ms_domain">True to use the 'com.microsoft' domain</param>
  /// <returns>Reference to the new DQ node</returns>
  const ONNX_NAMESPACE::NodeProto* AddDequantizeLinearNode(const std::string& node_name,
                                                           const std::string& input_name,
                                                           float input_scale,
                                                           int64_t input_zero_point,
                                                           ONNX_NAMESPACE::TensorProto_DataType zero_point_type,
                                                           const std::string& output_name,
                                                           bool use_ms_domain = false);

  /**
   * Wrapper function for set_raw_data.
   * First calls the set_raw_data and then calls ConvertRawDataInTensorProto
   * under big endian system.
   * @param tensor_proto given initializer tensor
   * @param raw_data     source raw_data pointer
   * @param raw_data_len  length of raw_data
   * @returns                 None
   */
  template <typename T1, typename T2>
  void SetRawDataInTensorProto(ONNX_NAMESPACE::TensorProto& tensor_proto, T1* raw_data, T2 raw_data_len) {
    using namespace ONNX_NAMESPACE;
    tensor_proto.set_raw_data(raw_data, raw_data_len);
#if defined(__BIG_ENDIAN__) || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
    ConvertRawDataInTensorProto(tensor_proto);
#endif
  }

  ONNX_NAMESPACE::ModelProto model_;
  ONNX_NAMESPACE::GraphProto* graph_ = model_.mutable_graph();
  std::unordered_map<std::string, Ort::Value> feeds_;
  RandomValueGenerator rand_gen_{std::optional<RandomValueGenerator::RandomSeedType>{2345}};

 private:
  /** Gets the TensorProto_DataType corresponding to the template type `T`. */
  template <typename T>
  constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType() {
    if constexpr (std::is_same_v<T, float>) {
      return ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
    } else if constexpr (std::is_same_v<T, uint8_t>) {
      return ONNX_NAMESPACE::TensorProto_DataType_UINT8;
    } else if constexpr (std::is_same_v<T, int8_t>) {
      return ONNX_NAMESPACE::TensorProto_DataType_INT8;
    } else if constexpr (std::is_same_v<T, uint16_t>) {
      return ONNX_NAMESPACE::TensorProto_DataType_UINT16;
    } else if constexpr (std::is_same_v<T, int16_t>) {
      return ONNX_NAMESPACE::TensorProto_DataType_INT16;
    } else if constexpr (std::is_same_v<T, int32_t>) {
      return ONNX_NAMESPACE::TensorProto_DataType_INT32;
    } else if constexpr (std::is_same_v<T, int64_t>) {
      return ONNX_NAMESPACE::TensorProto_DataType_INT64;
    } else if constexpr (std::is_same_v<T, std::string>) {
      return ONNX_NAMESPACE::TensorProto_DataType_STRING;
    } else if constexpr (std::is_same_v<T, bool>) {
      return ONNX_NAMESPACE::TensorProto_DataType_BOOL;
    } else if constexpr (std::is_same_v<T, Ort::Float16_t>) {
      return ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
    } else if constexpr (std::is_same_v<T, double>) {
      return ONNX_NAMESPACE::TensorProto_DataType_DOUBLE;
    } else if constexpr (std::is_same_v<T, uint32_t>) {
      return ONNX_NAMESPACE::TensorProto_DataType_UINT32;
    } else if constexpr (std::is_same_v<T, uint64_t>) {
      return ONNX_NAMESPACE::TensorProto_DataType_UINT64;
    } else if constexpr (std::is_same_v<T, Ort::BFloat16_t>) {
      return ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16;
    } else if constexpr (std::is_same_v<T, UInt4x2>) {
      return ONNX_NAMESPACE::TensorProto_DataType_UINT4;
    } else if constexpr (std::is_same_v<T, Int4x2>) {
      return ONNX_NAMESPACE::TensorProto_DataType_INT4;
    } else {
      return ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
    }
  }

  template <class U, class T>
  [[nodiscard]] inline gsl::span<U> ReinterpretAsSpan(gsl::span<T> src) {
    // adapted from gsl-lite span::as_span():
    // https://github.com/gsl-lite/gsl-lite/blob/4720a2980a30da085b4ddb4a0ea2a71af7351a48/include/gsl/gsl-lite.hpp#L4102-L4108
    Expects(src.size_bytes() % sizeof(U) == 0);
    return gsl::span<U>(reinterpret_cast<U*>(src.data()), src.size_bytes() / sizeof(U));
  }

  void SwapByteOrderInplace(size_t element_size_in_bytes, gsl::span<std::byte> bytes) {
    if (element_size_in_bytes > 1) {
      for (size_t offset = 0, lim = bytes.size_bytes(); offset < lim; offset += element_size_in_bytes) {
        std::reverse(bytes.begin() + offset, bytes.begin() + offset + element_size_in_bytes);
      }
    }
  }

  inline bool HasRawData(const ONNX_NAMESPACE::TensorProto& ten_proto) {
    // Can not be UNDEFINED and can not be STRING but test for STRING is usually performed separately
    // to return an error
    return ten_proto.data_type() != ONNX_NAMESPACE::TensorProto::UNDEFINED &&
           ten_proto.has_raw_data();  // XXX: Figure out how to do in proto3
  }

  // MIRRORED FROM core/framework/tensorprotoutils.h (ConvertRawDataInTensorProto) — keep in sync.
  // NOTE: If ORT mainline adds new tensor element types, this duplicate must be updated to match.
  // Big-endian platforms are not supported by this test framework.
  void ConvertRawDataInTensorProto(ONNX_NAMESPACE::TensorProto& tensor) {
    size_t element_size = 1;
    void* bytes = NULL;
    size_t num_elements = 0;

    // For some data_type, element size differs for raw data vs
    // data set using the add_<data_type>data() API
    if (HasRawData(tensor)) {
      static std::unordered_map<size_t, size_t> tensorproto_data_size{
          {ONNX_NAMESPACE::TensorProto_DataType_FLOAT, sizeof(float)},
          {ONNX_NAMESPACE::TensorProto_DataType_UINT8, sizeof(uint8_t)},
          {ONNX_NAMESPACE::TensorProto_DataType_INT8, sizeof(int8_t)},
          {ONNX_NAMESPACE::TensorProto_DataType_UINT16, sizeof(uint16_t)},
          {ONNX_NAMESPACE::TensorProto_DataType_INT16, sizeof(int16_t)},
          {ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, sizeof(uint16_t)},
          {ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16, sizeof(uint16_t)},
          {ONNX_NAMESPACE::TensorProto_DataType_INT32, sizeof(int32_t)},
          {ONNX_NAMESPACE::TensorProto_DataType_UINT32, sizeof(uint32_t)},
          {ONNX_NAMESPACE::TensorProto_DataType_UINT64, sizeof(uint64_t)},
          {ONNX_NAMESPACE::TensorProto_DataType_INT64, sizeof(int64_t)},
          {ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, sizeof(double)},
          {ONNX_NAMESPACE::TensorProto_DataType_BOOL, sizeof(uint8_t)},
          {ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN, sizeof(uint8_t)},
          {ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FNUZ, sizeof(uint8_t)},
          {ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2, sizeof(uint8_t)},
          {ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2FNUZ, sizeof(uint8_t)},
          {ONNX_NAMESPACE::TensorProto_DataType_UINT4, sizeof(uint8_t)},
          {ONNX_NAMESPACE::TensorProto_DataType_INT4, sizeof(uint8_t)},
          {ONNX_NAMESPACE::TensorProto_DataType_UINT2, sizeof(uint8_t)},
          {ONNX_NAMESPACE::TensorProto_DataType_INT2, sizeof(uint8_t)},
      };
      auto pos = tensorproto_data_size.find(tensor.data_type());
      if (pos == tensorproto_data_size.end()) {
        return;
      }
      element_size = pos->second;
      if (element_size == 1) {
        return;
      }
      num_elements = tensor.raw_data().size() / element_size;
      bytes = tensor.mutable_raw_data()->data();
    } else {  // HasRawData(tensor)

      switch (tensor.data_type()) {
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
          bytes = tensor.mutable_float_data()->mutable_data();
          num_elements = tensor.float_data_size();
          element_size = sizeof(float);
          break;

        case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
        case ONNX_NAMESPACE::TensorProto_DataType_UINT4:
        case ONNX_NAMESPACE::TensorProto_DataType_INT4:
        case ONNX_NAMESPACE::TensorProto_DataType_UINT2:
        case ONNX_NAMESPACE::TensorProto_DataType_INT2:
        case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
        case ONNX_NAMESPACE::TensorProto_DataType_INT8:
        case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
        case ONNX_NAMESPACE::TensorProto_DataType_INT16:
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
        case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN:
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FNUZ:
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2:
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2FNUZ:
        case ONNX_NAMESPACE::TensorProto_DataType_INT32:
          bytes = tensor.mutable_int32_data()->mutable_data();
          num_elements = tensor.int32_data_size();
          // We are setting this to int32_t size because we need to swap all 4 bytes
          // to represent 16 bits within 32 bits correctly on a LE/BE system.
          element_size = sizeof(int32_t);
          break;

        // uint32_t is stored in uint64_t
        case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
        case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
          bytes = tensor.mutable_uint64_data()->mutable_data();
          num_elements = tensor.uint64_data_size();
          element_size = sizeof(uint64_t);
          break;

        case ONNX_NAMESPACE::TensorProto_DataType_INT64:
          bytes = tensor.mutable_int64_data()->mutable_data();
          num_elements = tensor.int64_data_size();
          element_size = sizeof(int64_t);
          break;

        case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
          bytes = tensor.mutable_double_data()->mutable_data();
          num_elements = tensor.double_data_size();
          element_size = sizeof(double);
          break;
      }
    }

    gsl::span<std::byte> span = gsl::make_span(reinterpret_cast<std::byte*>(bytes), num_elements * element_size);
    SwapByteOrderInplace(element_size, span);
  }
};
}  // namespace test
}  // namespace onnxruntime
