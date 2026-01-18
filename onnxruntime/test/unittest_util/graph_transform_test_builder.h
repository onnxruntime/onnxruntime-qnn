// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <functional>
#include <type_traits>
#include <vector>
#include <random>
#include "gtest/gtest.h"

#include "core/common/span_utils.h"
#include "core/common/type_utils.h"
#include "core/graph/graph.h"
#include "core/framework/framework_common.h"
#include "core/framework/int4.h"
#include "core/optimizer/graph_transformer_level.h"
#include "core/graph/onnx_protobuf.h"
#include "core/framework/tensorprotoutils.h"
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/test_random_seed.h"
#include "test/util/include/inference_session_wrapper.h"

#define TEST_RETURN_IF(condition)                                               \
  do {                                                                          \
    if (condition) {                                                            \
      return ::onnxruntime::common::Status(::onnxruntime::common::ONNXRUNTIME,  \
                                           ::onnxruntime::common::FAIL,         \
                                           #condition " is evaluated to true"); \
    }                                                                           \
  } while (false)

#define TEST_RETURN_IF_NOT(condition)                                            \
  do {                                                                           \
    if (!(condition)) {                                                          \
      return ::onnxruntime::common::Status(::onnxruntime::common::ONNXRUNTIME,   \
                                           ::onnxruntime::common::FAIL,          \
                                           #condition " is evaluated to false"); \
    }                                                                            \
  } while (false)

namespace onnxruntime {
namespace test {

class RandomValueGenerator {
 public:
  using RandomEngine = std::default_random_engine;
  using RandomSeedType = RandomEngine::result_type;

  explicit RandomValueGenerator(optional<RandomSeedType> seed = {})
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
      std::is_integral<TInt>::value && !utils::IsByteType<TInt>::value,
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
      utils::IsByteType<TByte>::value,
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

  return narrow<size_t>(size);
}
};

template <typename T>
struct IsTypeQuantLinearCompatible : utils::IsByteType<T> {};

template <>
struct IsTypeQuantLinearCompatible<int16_t> : std::true_type {};

template <>
struct IsTypeQuantLinearCompatible<uint16_t> : std::true_type {};

template <>
struct IsTypeQuantLinearCompatible<Int4x2> : std::true_type {};

template <>
struct IsTypeQuantLinearCompatible<UInt4x2> : std::true_type {};

template <typename T>
struct IsTypeDequantLinearCompatible : utils::IsByteType<T> {};

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

  // const std::unordered_map<std::string, int>& DomainToVersionMap() const noexcept {
  //   return graph_.DomainToVersionMap();
  // }

  template <typename T>
  const ONNX_NAMESPACE::ValueInfoProto* MakeInput(const std::string name,
    const std::vector<int64_t>& shape,
    const std::vector<T>& data,
    AllocatorPtr = nullptr) {
    // if (!allocator) {
    //   allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
    // }
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
  const ONNX_NAMESPACE::ValueInfoProto* MakeInput(const std::string name,
      const std::vector<int64_t>& shape, T min, T max,
      AllocatorPtr allocator = nullptr) {
    return MakeInput<T>(name, shape, rand_gen_.Uniform<T>(shape, min, max), allocator);
  }

  const ONNX_NAMESPACE::ValueInfoProto* MakeInputBool(const std::string name,
      const std::vector<int64_t>& shape, AllocatorPtr allocator = nullptr) {
    std::vector<uint8_t> data_uint8 = rand_gen_.Uniform<uint8_t>(shape, 0, 1);
    std::vector<bool> data;
    for (uint8_t x : data_uint8) {
      data.push_back(x != 0);
    }
    return MakeInput<bool>(name, shape, data, allocator);
  }

  // template <typename T>
  // NodeArg* MakeInput(const std::optional<std::vector<int64_t>>& shape,
  //                    std::optional<std::string> input_name = std::nullopt) {
  //   ONNX_NAMESPACE::TypeProto type_proto;
  //   type_proto.mutable_tensor_type()->set_elem_type(utils::ToTensorProtoElementType<T>());
  //   if (shape != std::nullopt) {
  //     type_proto.mutable_tensor_type()->mutable_shape();
  //     for (auto& d : *shape) {
  //       auto dim = type_proto.mutable_tensor_type()->mutable_shape()->add_dim();
  //       if (d != -1) {
  //         dim->set_dim_value(d);
  //       }
  //     }
  //   }

  //   if (input_name == std::nullopt) {
  //     std::string name = graph_.GenerateNodeArgName("input");
  //     return &graph_.GetOrCreateNodeArg(name, &type_proto);
  //   } else {
  //     ORT_ENFORCE(graph_.GetNodeArg(*input_name) == nullptr, "Input name already exists: ", *input_name);
  //     return &graph_.GetOrCreateNodeArg(*input_name, &type_proto);
  //   }
  // }

  // // Make optional tensor
  // NodeArg* MakeOptionalTensor() {
  //   ONNX_NAMESPACE::TypeProto type_proto;
  //   type_proto.mutable_tensor_type()->set_elem_type(utils::ToTensorProtoElementType<float>());
  //   std::string name;
  //   return &graph_.GetOrCreateNodeArg(name, &type_proto);
  // }

  // template <typename T>
  // NodeArg* MakeSymbolicInput(const std::vector<std::variant<int64_t, std::string>>& shape) {
  //   ONNX_NAMESPACE::TypeProto type_proto;
  //   type_proto.mutable_tensor_type()->set_elem_type(utils::ToTensorProtoElementType<T>());
  //   type_proto.mutable_tensor_type()->mutable_shape();
  //   for (auto& d : shape) {
  //     auto dim = type_proto.mutable_tensor_type()->mutable_shape()->add_dim();
  //     std::visit([&dim](auto&& arg) -> void {
  //       using V = std::decay_t<decltype(arg)>;
  //       if constexpr (std::is_same_v<V, int64_t>) {
  //         ORT_ENFORCE(arg >= 0, "Negative dimension is not allowed in symbolic shape");
  //         dim->set_dim_value(arg);
  //       } else {
  //         dim->set_dim_param(arg);
  //       }
  //     },
  //                d);
  //   }

  //   std::string name = graph_.GenerateNodeArgName("symbolic_input");
  //   return &graph_.GetOrCreateNodeArg(name, &type_proto);
  // }

  const ONNX_NAMESPACE::ValueInfoProto* MakeOutput(const std::string name) {
    ONNX_NAMESPACE::ValueInfoProto* out = graph_->add_output();
    out->set_name(name);
    return out;
  }

  template <typename T>
  const ONNX_NAMESPACE::ValueInfoProto* MakeOutput(const std::string name,
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

  // NodeArg* MakeIntermediate() {
  //   std::string name = graph_.GenerateNodeArgName("node");
  //   return &graph_.GetOrCreateNodeArg(name, nullptr);
  // }

  // template <typename T>
  // NodeArg* MakeIntermediate(const std::optional<std::vector<int64_t>>& shape) {
  //   ONNX_NAMESPACE::TypeProto type_proto;
  //   type_proto.mutable_tensor_type()->set_elem_type(ToTensorProtoElementType<T>());
  //   if (shape != std::nullopt) {
  //     type_proto.mutable_tensor_type()->mutable_shape();
  //     for (auto& d : *shape) {
  //       auto dim = type_proto.mutable_tensor_type()->mutable_shape()->add_dim();
  //       if (d != -1) {
  //         dim->set_dim_value(d);
  //       }
  //     }
  //   }
  //   std::string name = graph_.GenerateNodeArgName("node");
  //   return &graph_.GetOrCreateNodeArg(name, &type_proto);
  // }

  /// <summary>
  /// Makes an initializer from the provided shape, element type, and raw data bytes.
  /// </summary>
  /// <param name="shape">Initializer shape</param>
  /// <param name="elem_type">ONNX tensor element data type</param>
  /// <param name="raw_data">Raw data bytes</param>
  /// <returns>ValueInfo pointer for the initializer</returns>
  const ONNX_NAMESPACE::TensorProto* MakeInitializer(std::string name,
    gsl::span<const int64_t> shape,
    ONNX_NAMESPACE::TensorProto_DataType elem_type,
    gsl::span<const std::byte> raw_data);

  template <typename T>
  const ONNX_NAMESPACE::TensorProto* MakeInitializer(std::string name,
    const std::vector<int64_t>& shape,
    const std::vector<T>& data) {
    gsl::span<const std::byte> raw_data = ReinterpretAsSpan<const std::byte, const T>(data);
    return MakeInitializer(name, shape, ToTensorProtoElementType<T>(), raw_data);
  }

  // Special handle for std::vector<bool>.
  const ONNX_NAMESPACE::TensorProto* MakeInitializerBool(std::string name,
    const std::vector<int64_t>& shape, const std::vector<bool>& data) {
    ONNX_NAMESPACE::TensorProto* tensor_proto = graph_->add_initializer();
    tensor_proto->set_name(name);
    tensor_proto->set_data_type(ToTensorProtoElementType<bool>());
    std::unique_ptr<bool[]> data_buffer = std::make_unique<bool[]>(data.size());
    for (size_t i = 0; i < data.size(); ++i) data_buffer[i] = data[i];
    utils::SetRawDataInTensorProto(
      *tensor_proto,
      data_buffer.get(),
      data.size());

    for (auto& dim : shape) {
      tensor_proto->add_dims(dim);
    }

    return tensor_proto;
  }

  const ONNX_NAMESPACE::TensorProto* MakeRandInitializerBool(std::string name, const std::vector<int64_t>& shape) {
    std::vector<uint8_t> data_uint8 = rand_gen_.Uniform<uint8_t>(shape, 0, 1);
    std::vector<bool> data;
    for (uint8_t x : data_uint8) {
      data.push_back(x != 0);
    }
    return MakeInitializerBool(name, shape, data);
  }

  template <typename T>
  const ONNX_NAMESPACE::TensorProto* MakeInitializer(std::string name,
    const std::vector<int64_t>& shape, T min, T max) {
    return MakeInitializer<T>(name, shape, rand_gen_.Uniform<T>(shape, min, max));
  }

  template <typename T>
  const ONNX_NAMESPACE::TensorProto* MakeScalarInitializer(std::string name, T data) {
    return MakeInitializer(name, {}, std::vector<T>{data});
  }

  template <typename T>
  const ONNX_NAMESPACE::TensorProto* Make1DInitializer(std::string name, const std::vector<T>& data) {
    return MakeInitializer(name, {static_cast<int64_t>(data.size())}, data);
  }

  // NodeArg* MakeEmptyInput() {
  //   NodeArg* empty = &graph_.GetOrCreateNodeArg("", nullptr);
  //   return empty;
  // }

  const ONNX_NAMESPACE::NodeProto* AddNode(const std::string& node_name,
               const std::string& op_type,
               const std::vector<std::string>& input_names,
               const std::vector<std::string>& output_names,
               const std::string& domain = "",
               std::vector<ONNX_NAMESPACE::AttributeProto> node_attributes = {}) {
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

  // Node& AddConvNode(NodeArg* input_arg,
  //                   NodeArg* weights_arg,
  //                   NodeArg* output_arg) {
  //   std::vector<NodeArg*> input_args;
  //   input_args.push_back(input_arg);
  //   input_args.push_back(weights_arg);

  //   return AddNode("Conv", input_args, {output_arg});
  // }

  template <typename ZpType, typename ScaleType = float>
  typename std::enable_if<IsTypeQuantLinearCompatible<ZpType>::value, const ONNX_NAMESPACE::NodeProto*>::type
  AddQuantizeLinearNode(const std::string& node_name,
                        const std::string input_name,
                        ScaleType input_scale,
                        ZpType input_zero_point,
                        const std::string output_name,
                        bool use_ms_domain = false) {
    std::vector<std::string> input_names;
    input_names.push_back(input_name);
    auto scale = MakeScalarInitializer<ScaleType>(node_name+"_inp_scale", input_scale);
    auto zp = MakeScalarInitializer<ZpType>(node_name+"_inp_zp", input_zero_point);
    input_names.push_back(scale->name());
    input_names.push_back(zp->name());

    std::string domain = use_ms_domain ? kMSDomain : "";
    return AddNode(node_name, "QuantizeLinear", input_names, {output_name}, domain);
  }

  template <typename T>
  typename std::enable_if<IsTypeQuantLinearCompatible<T>::value, const ONNX_NAMESPACE::NodeProto*>::type
  AddQuantizeLinearNode(const std::string& node_name,
                        const std::string input_name,
                        const std::vector<float>& input_scales,
                        const std::vector<T>& input_zero_points,
                        const std::string output_name,
                        std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {},
                        bool use_ms_domain = false) {
    std::vector<std::string> input_names;
    input_names.push_back(input_name);

    std::vector<int64_t> qparams_shape = {static_cast<int64_t>(input_scales.size())};
    auto scales = MakeInitializer<float>(node_name+"_inp_scale", qparams_shape, input_scales);
    auto zps = MakeInitializer<T>(node_name+"_inp_zp", qparams_shape, input_zero_points);
    input_names.push_back(scales->name());
    input_names.push_back(zps->name());

    std::string domain = use_ms_domain ? kMSDomain : "";
    return AddNode(node_name, "QuantizeLinear", input_names, {output_name}, domain, attributes);
  }

  const ONNX_NAMESPACE::NodeProto* AddQuantizeLinearNode(const std::string& node_name,
                              const std::string input_name,
                              float input_scale,
                              const std::string output_name,
                              bool use_ms_domain = false) {
    std::vector<std::string> input_names;
    auto scale = MakeScalarInitializer<float>(
      node_name+"_inp_scale", input_scale);
    input_names.push_back(input_name);
    input_names.push_back(scale->name());

    std::string domain = use_ms_domain ? kMSDomain : "";
    return AddNode(node_name, "QuantizeLinear", input_names, {output_name}, domain);
  }

  const ONNX_NAMESPACE::NodeProto* AddQuantizeLinearNode(const std::string& node_name,
                              const std::string input_name,
                              const std::vector<float>& input_scales,
                              const std::string output_name,
                              std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {},
                              bool use_ms_domain = false) {
    std::vector<std::string> input_names;
    auto scale = Make1DInitializer<float>(node_name+"_inp_scale", input_scales);
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
                                                         const std::string input_name,
                                                         float input_scale,
                                                         int64_t input_zero_point,
                                                         ONNX_NAMESPACE::TensorProto_DataType zero_point_type,
                                                         const std::string output_name,
                                                         bool use_ms_domain = false);

  template <typename ZpType, typename ScaleType = float>
  typename std::enable_if<IsTypeDequantLinearCompatible<ZpType>::value, const ONNX_NAMESPACE::NodeProto*>::type
  AddDequantizeLinearNode(const std::string& node_name,
                          const std::string input_name,
                          ScaleType input_scale,
                          ZpType input_zero_point,
                          const std::string output_name,
                          bool use_ms_domain = false) {
    std::vector<std::string> input_names;
    input_names.push_back(input_name);
    auto scale = MakeScalarInitializer<ScaleType>(node_name+"_inp_scale", input_scale);
    auto zp = MakeScalarInitializer<ZpType>(node_name+"_inp_zp", input_zero_point);
    input_names.push_back(scale->name());
    input_names.push_back(zp->name());

    std::string domain = use_ms_domain ? kMSDomain : "";
    return AddNode(node_name, "DequantizeLinear", input_names, {output_name}, domain);
  }

  template <typename T>
  typename std::enable_if<IsTypeDequantLinearCompatible<T>::value, const ONNX_NAMESPACE::NodeProto*>::type
  AddDequantizeLinearNode(const std::string& node_name,
                          const std::string input_name,
                          const std::vector<float>& input_scales,
                          const std::vector<T>& input_zero_points,
                          const std::string output_name,
                          std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {},
                          bool use_ms_domain = false) {
    std::vector<std::string> input_names;
    input_names.push_back(input_name);

    std::vector<int64_t> qparams_shape = {static_cast<int64_t>(input_scales.size())};
    auto scales = MakeInitializer<float>(node_name+"_inp_scale", qparams_shape, input_scales);
    auto zps = MakeInitializer<T>(node_name+"_inp_zp", qparams_shape, input_zero_points);
    input_names.push_back(scales->name());
    input_names.push_back(zps->name());

    std::string domain = use_ms_domain ? kMSDomain : "";
    return AddNode(node_name, "DequantizeLinear", input_names, {output_name}, domain, attributes);
  }

  const ONNX_NAMESPACE::NodeProto* AddDequantizeLinearNode(const std::string& node_name,
                                                           const std::string input_name,
                                                           float input_scale,
                                                           const std::string output_name,
                                                           bool use_ms_domain = false) {
    std::vector<std::string> input_names;
    auto scale = MakeScalarInitializer<float>(
      node_name+"_inp_scale", input_scale);
    input_names.push_back(input_name);
    input_names.push_back(scale->name());

    std::string domain = use_ms_domain ? kMSDomain : "";
    return AddNode(node_name, "DequantizeLinear", input_names, {output_name}, domain);
  }

  const ONNX_NAMESPACE::NodeProto* AddDequantizeLinearNode(const std::string& node_name,
                                const std::string input_name,
                                const std::vector<float>& input_scales,
                                const std::string output_name,
                                std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {},
                                bool use_ms_domain = false) {
    std::vector<std::string> input_names;
    input_names.push_back(input_name);
    auto scale = Make1DInitializer<float>(node_name+"_inp_scale", input_scales);
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
                                                          const std::string input_name,
                                                          float input_scale,
                                                          int64_t input_zero_point,
                                                          ONNX_NAMESPACE::TensorProto_DataType zero_point_type,
                                                          const std::string output_name,
                                                          bool use_ms_domain = false);

  // template <typename TWeight>
  // Node& AddQLinearConvNode(NodeArg* input_arg,
  //                          float input_scale,
  //                          uint8_t input_zero_point,
  //                          NodeArg* weight_arg,
  //                          float weights_scale,
  //                          TWeight weights_zero_point,
  //                          NodeArg* output_arg,
  //                          float output_scale,
  //                          uint8_t output_zero_point) {
  //   std::vector<NodeArg*> input_args{input_arg};
  //   input_args.push_back(MakeScalarInitializer<float>(input_scale));
  //   input_args.push_back(MakeScalarInitializer<uint8_t>(input_zero_point));
  //   input_args.push_back(weight_arg);
  //   input_args.push_back(MakeScalarInitializer<float>(weights_scale));
  //   input_args.push_back(MakeScalarInitializer<TWeight>(weights_zero_point));
  //   input_args.push_back(MakeScalarInitializer<float>(output_scale));
  //   input_args.push_back(MakeScalarInitializer<TWeight>(output_zero_point));

  //   return AddNode("QLinearConv", input_args, {output_arg});
  // }

  // Node& AddQLinearBinaryNode(const std::string& op_type,
  //                            NodeArg* input1_arg,
  //                            float input1_scale,
  //                            uint8_t input1_zero_point,
  //                            NodeArg* input2_arg,
  //                            float input2_scale,
  //                            uint8_t input2_zero_point,
  //                            NodeArg* output_arg,
  //                            float output_scale,
  //                            uint8_t output_zero_point) {
  //   std::vector<NodeArg*> input_args;
  //   input_args.push_back(input1_arg);
  //   input_args.push_back(MakeScalarInitializer<float>(input1_scale));
  //   input_args.push_back(MakeScalarInitializer<uint8_t>(input1_zero_point));
  //   input_args.push_back(input2_arg);
  //   input_args.push_back(MakeScalarInitializer<float>(input2_scale));
  //   input_args.push_back(MakeScalarInitializer<uint8_t>(input2_zero_point));
  //   input_args.push_back(MakeScalarInitializer<float>(output_scale));
  //   input_args.push_back(MakeScalarInitializer<uint8_t>(output_zero_point));

  //   return AddNode(op_type, input_args, {output_arg}, kMSDomain);
  // }

  // Node& AddQLinearConcatLike(const std::string& op_type,
  //                            NodeArg* output_arg,
  //                            float output_scale,
  //                            uint8_t output_zero_point,
  //                            std::vector<std::tuple<NodeArg*, float, uint8_t>> quantized_inputs) {
  //   std::vector<NodeArg*> input_args;
  //   input_args.push_back(MakeScalarInitializer<float>(output_scale));
  //   input_args.push_back(MakeScalarInitializer<uint8_t>(output_zero_point));
  //   for (size_t input_index = 0; input_index < quantized_inputs.size(); ++input_index) {
  //     input_args.push_back(std::get<0>(quantized_inputs[input_index]));
  //     input_args.push_back(MakeScalarInitializer<float>(std::get<1>(quantized_inputs[input_index])));
  //     input_args.push_back(MakeScalarInitializer<uint8_t>(std::get<2>(quantized_inputs[input_index])));
  //   }
  //   return AddNode(op_type, input_args, {output_arg}, kMSDomain);
  // }

  // Node& AddQLinearActivationNode(const std::string& op_type,
  //                                NodeArg* input_arg,
  //                                float input_scale,
  //                                uint8_t input_zero_point,
  //                                NodeArg* output_arg,
  //                                float output_scale,
  //                                uint8_t output_zero_point) {
  //   std::vector<NodeArg*> input_args;
  //   input_args.push_back(input_arg);
  //   input_args.push_back(MakeScalarInitializer<float>(input_scale));
  //   input_args.push_back(MakeScalarInitializer<uint8_t>(input_zero_point));
  //   input_args.push_back(MakeScalarInitializer<float>(output_scale));
  //   input_args.push_back(MakeScalarInitializer<uint8_t>(output_zero_point));

  //   return AddNode(op_type, input_args, {output_arg}, kMSDomain);
  // }

  // void SetGraphOutputs() {
  //   std::vector<const NodeArg*> outputs;
  //   for (auto& output_name : output_names_) {
  //     outputs.push_back(graph_.GetNodeArg(output_name));
  //   }
  //   graph_.SetOutputs(outputs);
  // }

  ONNX_NAMESPACE::ModelProto model_;
  ONNX_NAMESPACE::GraphProto* graph_ = model_.mutable_graph();
  std::unordered_map<std::string, Ort::Value> feeds_;
  // std::vector<std::string> output_names_;
  RandomValueGenerator rand_gen_{optional<RandomValueGenerator::RandomSeedType>{2345}};

private:
  /** Gets the TensorProto_DataType corresponding to the template type `T`. */
  template <typename T>
  constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType() {
    return ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
  }
  template <>
  constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType<float>() {
    return ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
  }
  template <>
  constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType<uint8_t>() {
    return ONNX_NAMESPACE::TensorProto_DataType_UINT8;
  }
  template <>
  constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType<int8_t>() {
    return ONNX_NAMESPACE::TensorProto_DataType_INT8;
  }
  template <>
  constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType<uint16_t>() {
    return ONNX_NAMESPACE::TensorProto_DataType_UINT16;
  }
  template <>
  constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType<int16_t>() {
    return ONNX_NAMESPACE::TensorProto_DataType_INT16;
  }
  template <>
  constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType<int32_t>() {
    return ONNX_NAMESPACE::TensorProto_DataType_INT32;
  }
  template <>
  constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType<int64_t>() {
    return ONNX_NAMESPACE::TensorProto_DataType_INT64;
  }
  template <>
  constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType<std::string>() {
    return ONNX_NAMESPACE::TensorProto_DataType_STRING;
  }
  template <>
  constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType<bool>() {
    return ONNX_NAMESPACE::TensorProto_DataType_BOOL;
  }
  template <>
  constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType<Ort::Float16_t>() {
    return ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
  }
  template <>
  constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType<double>() {
    return ONNX_NAMESPACE::TensorProto_DataType_DOUBLE;
  }
  template <>
  constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType<uint32_t>() {
    return ONNX_NAMESPACE::TensorProto_DataType_UINT32;
  }
  template <>
  constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType<uint64_t>() {
    return ONNX_NAMESPACE::TensorProto_DataType_UINT64;
  }
  template <>
  constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType<Ort::BFloat16_t>() {
    return ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16;
  }
};

void TransformerTester(const std::function<void(ModelTestBuilder& helper)>& build_test_case,
                       const std::function<void(InferenceSessionWrapper& session)>& check_transformed_graph,
                       TransformerLevel baseline_level,
                       TransformerLevel target_level,
                       int opset_version = 12,
                       double per_sample_tolerance = 0.0,
                       double relative_per_sample_tolerance = 0.0,
                       std::unique_ptr<GraphTransformer> transformer = nullptr,
                       const std::function<void(SessionOptions&)>& add_session_options = {},
                       const InlinedHashSet<std::string>& disabled_optimizers = {},
                       std::unique_ptr<IExecutionProvider> ep = nullptr);

void TransformerTester(const std::function<void(ModelTestBuilder& helper)>& build_test_case,
                       const std::function<void(InferenceSessionWrapper& session)>& check_transformed_graph,
                       TransformerLevel baseline_level,
                       TransformerLevel target_level,
                       const std::vector<int>& opset_versions,
                       double per_sample_tolerance = 0.0,
                       double relative_per_sample_tolerance = 0.0,
                       std::unique_ptr<GraphTransformer> transformer = nullptr,  // must be null in this case.
                       const std::function<void(SessionOptions&)>& add_session_options = {},
                       const InlinedHashSet<std::string>& disabled_optimizers = {});
}  // namespace test
}  // namespace onnxruntime
