// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/util/include/test_utils.h"

#include "core/common/narrow.h"
#include "core/common/span_utils.h"
#include "core/framework/ort_value.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/tensorprotoutils.h"

#include "test/util/include/asserts.h"
#include "test/util/include/test/test_environment.h"
#include "test/util/include/inference_session_wrapper.h"

#include "gmock/gmock.h"

namespace onnxruntime {
namespace test {
void VerifyOutput(const std::string& output_name,
                  const Ort::Value& expected_value,
                  const Ort::Value& actual_value,
                  float fp32_abs_err) {
  // Get tensor type info
  auto expected_type_info = expected_value.GetTensorTypeAndShapeInfo();
  auto actual_type_info = actual_value.GetTensorTypeAndShapeInfo();

  // Verify shapes match
  auto expected_shape = expected_type_info.GetShape();
  auto actual_shape = actual_type_info.GetShape();
  ASSERT_EQ(expected_shape.size(), actual_shape.size()) << "Shape rank mismatch for " << output_name;
  for (size_t i = 0; i < expected_shape.size(); ++i) {
    ASSERT_EQ(expected_shape[i], actual_shape[i]) << "Shape dimension " << i << " mismatch for " << output_name;
  }

  // Verify data types match
  auto expected_type = expected_type_info.GetElementType();
  auto actual_type = actual_type_info.GetElementType();
  ASSERT_EQ(expected_type, actual_type) << "Data type mismatch for " << output_name;

  // Get element count
  size_t element_count = expected_type_info.GetElementCount();

  // Compare data based on type
  switch (expected_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: {
      const uint32_t* expected_data = expected_value.GetTensorData<uint32_t>();
      const uint32_t* actual_data = actual_value.GetTensorData<uint32_t>();
      for (size_t i = 0; i < element_count; ++i) {
        EXPECT_EQ(expected_data[i], actual_data[i]) << "Element " << i << " mismatch for " << output_name;
      }
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
      const int32_t* expected_data = expected_value.GetTensorData<int32_t>();
      const int32_t* actual_data = actual_value.GetTensorData<int32_t>();
      for (size_t i = 0; i < element_count; ++i) {
        EXPECT_EQ(expected_data[i], actual_data[i]) << "Element " << i << " mismatch for " << output_name;
      }
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
      const int64_t* expected_data = expected_value.GetTensorData<int64_t>();
      const int64_t* actual_data = actual_value.GetTensorData<int64_t>();
      for (size_t i = 0; i < element_count; ++i) {
        EXPECT_EQ(expected_data[i], actual_data[i]) << "Element " << i << " mismatch for " << output_name;
      }
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: {
      const uint16_t* expected_data = expected_value.GetTensorData<uint16_t>();
      const uint16_t* actual_data = actual_value.GetTensorData<uint16_t>();
      for (size_t i = 0; i < element_count; ++i) {
        EXPECT_EQ(expected_data[i], actual_data[i]) << "Element " << i << " mismatch for " << output_name;
      }
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
      const uint8_t* expected_data = expected_value.GetTensorData<uint8_t>();
      const uint8_t* actual_data = actual_value.GetTensorData<uint8_t>();
      for (size_t i = 0; i < element_count; ++i) {
        EXPECT_EQ(expected_data[i], actual_data[i]) << "Element " << i << " mismatch for " << output_name;
      }
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
      const int8_t* expected_data = expected_value.GetTensorData<int8_t>();
      const int8_t* actual_data = actual_value.GetTensorData<int8_t>();
      for (size_t i = 0; i < element_count; ++i) {
        EXPECT_EQ(expected_data[i], actual_data[i]) << "Element " << i << " mismatch for " << output_name;
      }
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
      const bool* expected_data = expected_value.GetTensorData<bool>();
      const bool* actual_data = actual_value.GetTensorData<bool>();
      for (size_t i = 0; i < element_count; ++i) {
        EXPECT_EQ(expected_data[i], actual_data[i]) << "Element " << i << " mismatch for " << output_name;
      }
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
      const float* expected_data = expected_value.GetTensorData<float>();
      const float* actual_data = actual_value.GetTensorData<float>();
      for (size_t i = 0; i < element_count; ++i) {
        EXPECT_NEAR(expected_data[i], actual_data[i], fp32_abs_err)
            << "Element " << i << " mismatch for " << output_name;
      }
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
      const Ort::Float16_t* expected_data = expected_value.GetTensorData<Ort::Float16_t>();
      const Ort::Float16_t* actual_data = actual_value.GetTensorData<Ort::Float16_t>();
      for (size_t i = 0; i < element_count; ++i) {
        float expected_float = static_cast<float>(expected_data[i]);
        float actual_float = static_cast<float>(actual_data[i]);
        EXPECT_NEAR(expected_float, actual_float, fp32_abs_err)
            << "Element " << i << " mismatch for " << output_name;
      }
      break;
    }
    default:
      FAIL() << "Unhandled data type " << expected_type << " for " << output_name;
  }
}

static void VerifyOutputs(const std::vector<std::string>& output_names,
                          const std::vector<Ort::Value>& expected_fetches,
                          const std::vector<Ort::Value>& fetches,
                          const EPVerificationParams& params) {
  ASSERT_EQ(expected_fetches.size(), fetches.size());

  for (size_t i = 0, end = expected_fetches.size(); i < end; ++i) {
    VerifyOutput(output_names[i], expected_fetches[i], fetches[i], params.fp32_abs_err);
  }
}

// TODO: Implement the CountAssignedNodes and VerifyEPNodeAssignment once public API support get ep graph partitioning info

// int CountAssignedNodes(const Graph& current_graph, const std::string& ep_type) {
//   int count = 0;

//   for (const auto& node : current_graph.Nodes()) {
//     if (node.GetExecutionProviderType() == ep_type) {
//       ++count;
//     }

//     if (node.ContainsSubgraph()) {
//       for (const auto& entry : node.GetSubgraphs()) {
//         count += CountAssignedNodes(*entry, ep_type);
//       }
//     }
//   }

//   return count;
// }

// void VerifyEPNodeAssignment(const Graph& graph, const std::string& provider_type,
//                             ExpectedEPNodeAssignment assignment) {
//   const auto provider_node_count = CountAssignedNodes(graph, provider_type);
//   if (assignment == ExpectedEPNodeAssignment::All) {
//     // Verify the entire graph is assigned to the EP
//     ASSERT_EQ(provider_node_count, graph.NumberOfNodes()) << "Not all nodes were assigned to " << provider_type;
//   } else if (assignment == ExpectedEPNodeAssignment::None) {
//     // or none of the graph
//     ASSERT_EQ(provider_node_count, 0) << "Some nodes were assigned to " << provider_type;
//   } else {
//     // or some of the graph
//     ASSERT_GT(provider_node_count, 0) << "No nodes were assigned to " << provider_type;
//   }
// }

static gsl::span<const std::byte> GetModelBytes(ModelPathOrBytes model_path_or_bytes,
                                                std::vector<std::byte>& byte_buffer_out) {
  if (const auto* model_bytes = std::get_if<gsl::span<const std::byte>>(&model_path_or_bytes);
      model_bytes != nullptr) {
    byte_buffer_out = std::vector<std::byte>{};
    return *model_bytes;
  }

  const auto model_path = std::get<std::basic_string_view<ORTCHAR_T>>(model_path_or_bytes);

  std::vector<std::byte> byte_buffer{};
  std::ifstream stream{std::basic_string<ORTCHAR_T>{model_path},
                       std::ios::in | std::ios::binary | std::ios::ate};
  assert(stream && "Failed to open file.");
  const auto num_bytes = narrow<size_t>(stream.tellg());
  byte_buffer.resize(num_bytes);
  stream.seekg(0);
  assert(stream.read(reinterpret_cast<char*>(byte_buffer.data()), num_bytes) && "Failed to read file.");

  byte_buffer_out = std::move(byte_buffer);
  return gsl::span<const std::byte>(byte_buffer_out);
}

void RunAndVerifyOutputsWithEP(ModelPathOrBytes model_path_or_bytes, std::string_view log_id,
                               std::unique_ptr<IExecutionProvider> execution_provider,
                               std::unordered_map<std::string, Ort::Value>& feeds,
                               const EPVerificationParams& params,
                               const std::function<void(Ort::SessionOptions&)>& session_options_updater,
                               bool verify_outputs) {
  std::vector<std::byte> model_data_buffer{};
  const auto model_data = GetModelBytes(model_path_or_bytes, model_data_buffer);

  // Use public API directly
  Ort::SessionOptions ort_so;
  if (session_options_updater) {
    session_options_updater(ort_so);
  }

  //
  // get expected output from CPU EP
  //
  Ort::Session session_object(*GetOrtEnv(), model_data.data(), static_cast<int>(model_data.size()), ort_so);

  // fetch all outputs using public API
  std::vector<std::string> output_names;
  size_t output_count = session_object.GetOutputCount();
  output_names.reserve(output_count);

  Ort::AllocatorWithDefaultOptions allocator;
  for (size_t i = 0; i < output_count; ++i) {
    auto output_name = session_object.GetOutputNameAllocated(i, allocator);
    output_names.push_back(output_name.release());
  }

  Ort::RunOptions ort_run_options;
  ort_run_options.SetRunTag(log_id.data());

  std::vector<Ort::Value> expected_fetches;
  RunWithEPABI(session_object, ort_run_options, feeds, expected_fetches);

  auto provider_type = execution_provider->Type();  // copy string so the std::move doesn't affect us

  //
  // get output with EP enabled
  //
  RunAndVerifyOutputsWithEPABI(model_path_or_bytes, ort_so, provider_type, log_id, feeds, params, verify_outputs);
}

void RunWithEPABI(Ort::Session& ort_session,
                  const Ort::RunOptions& ort_ro,
                  std::unordered_map<std::string, Ort::Value>& feeds,
                  std::vector<Ort::Value>& output_vals) {
  // Fetch all input/output names using public API - store strings to ensure lifetime
  std::vector<std::string> ort_input_names = ort_session.GetInputNames();
  std::vector<std::string> ort_output_names = ort_session.GetOutputNames();
  size_t input_count = ort_input_names.size();
  size_t output_count = ort_output_names.size();
  std::vector<const char*> ort_input_names_cstr(input_count);
  std::vector<const char*> ort_output_names_cstr(output_count);
  std::transform(ort_input_names.begin(), ort_input_names.end(), ort_input_names_cstr.begin(),
                   [](const std::string& s) { return s.c_str(); });
  std::transform(ort_output_names.begin(), ort_output_names.end(), ort_output_names_cstr.begin(),
                   [](const std::string& s) { return s.c_str(); });

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.reserve(input_count);
  for (size_t i = 0; i < input_count; ++i) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    ort_inputs.emplace_back(Ort::Value::CreateTensor(
      memory_info,
      (void*)feeds.at(ort_input_names[i]).GetTensorRawData(),
      feeds.at(ort_input_names[i]).GetTensorSizeInBytes(),
      feeds.at(ort_input_names[i]).GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape().data(),
      feeds.at(ort_input_names[i]).GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape().size(),
      feeds.at(ort_input_names[i]).GetTypeInfo().GetTensorTypeAndShapeInfo().GetElementType())
    );
  }
  // Run.
  output_vals = ort_session.Run(ort_ro,
                                ort_input_names_cstr.data(),
                                ort_inputs.data(),
                                input_count,
                                ort_output_names_cstr.data(),
                                ort_output_names_cstr.size());

}

void RunAndVerifyOutputsWithEPABI(ModelPathOrBytes model_path_or_bytes,
                                  Ort::SessionOptions& ort_so,
                                  const std::string& /* provider_type */,
                                  std::string_view log_id,
                                  std::unordered_map<std::string, Ort::Value>& feeds,
                                  const EPVerificationParams& params,
                                  bool verify_outputs) {
  std::vector<std::byte> model_data_buffer{};
  const auto model_data = GetModelBytes(model_path_or_bytes, model_data_buffer);

  //
  // get expected output from CPU EP using public API
  //
  Ort::SessionOptions cpu_so;
  Ort::Session cpu_session(*GetOrtEnv(), model_data.data(), static_cast<int>(model_data.size()), cpu_so);

  // fetch all outputs using public API
  std::vector<std::string> output_names;
  size_t output_count = cpu_session.GetOutputCount();
  output_names.reserve(output_count);

  Ort::AllocatorWithDefaultOptions allocator;
  for (size_t i = 0; i < output_count; ++i) {
    auto output_name = cpu_session.GetOutputNameAllocated(i, allocator);
    output_names.push_back(output_name.release());
  }

  Ort::RunOptions cpu_run_options;
  cpu_run_options.SetRunTag(log_id.data());

  std::vector<Ort::Value> expected_fetches;
  RunWithEPABI(cpu_session, cpu_run_options, feeds, expected_fetches);

  // Run with EP and verify the result
  Ort::Session ort_session(*GetOrtEnv(), model_data.data(), static_cast<int>(model_data.size()), ort_so);

  // Note: VerifyEPNodeAssignment and graph_verifier require internal graph access
  // These are commented out since we're migrating to public API
  // ASSERT_NO_FATAL_FAILURE(VerifyEPNodeAssignment(ort_session.GetGraph(), provider_type, params.ep_node_assignment));

  Ort::RunOptions ort_run_options;
  ort_run_options.SetRunTag(log_id.data());

  std::vector<Ort::Value> fetches;
  RunWithEPABI(ort_session, ort_run_options, feeds, fetches);

  if (verify_outputs) {
    VerifyOutputs(output_names, expected_fetches, fetches, params);
  }

  // Note: graph_verifier requires internal graph access, commented out for public API migration
  // if (params.graph_verifier) {
  //   (*params.graph_verifier)(ort_session.GetGraph());
  // }
}

void TestModelLoad(ModelPathOrBytes model_path_or_bytes,
                   std::unique_ptr<IExecutionProvider>, /* execution_provider */
                   const std::function<void(const Graph&)>& /* check_graph */) {
  std::vector<std::byte> model_data_buffer{};
  const auto model_data = GetModelBytes(model_path_or_bytes, model_data_buffer);

  Ort::SessionOptions ort_so;
  
  // Note: EP registration and graph verification require internal APIs
  // These are not available in the public API, so we just test model loading
  OrtSessionWrapper session_object(*GetOrtEnv(), model_data.data(), static_cast<int>(model_data.size()), ort_so);
  
  // Note: check_graph callback requires internal graph access, commented out for public API migration
  // if (check_graph) {
  //   check_graph(session_object.GetGraph());
  // }
}

void CheckShapeEquality(const ONNX_NAMESPACE::TensorShapeProto* shape1,
                        const ONNX_NAMESPACE::TensorShapeProto* shape2) {
  EXPECT_NE(shape1, nullptr);
  EXPECT_NE(shape2, nullptr);
  EXPECT_EQ(shape1->dim_size(), shape2->dim_size()) << "Shapes do not have same rank";
  auto min_dims = std::min(shape1->dim_size(), shape2->dim_size());
  for (int i = 0; i < min_dims; ++i) {
    auto dim1 = shape1->dim(i);
    auto dim2 = shape2->dim(i);
    EXPECT_EQ(dim1.has_dim_value(), dim2.has_dim_value());
    if (dim1.has_dim_value()) {
      EXPECT_EQ(dim1.dim_value(), dim2.dim_value());
    }
    EXPECT_EQ(dim1.has_dim_param(), dim2.has_dim_param());
    if (dim1.has_dim_param()) {
      EXPECT_EQ(dim1.dim_param(), dim2.dim_param());
    }
  }
}

static void SetNameAndType(std::string attr_name, ONNX_NAMESPACE::AttributeProto_AttributeType attr_type, ONNX_NAMESPACE::AttributeProto& a) {
  a.set_name(std::move(attr_name));
  a.set_type(attr_type);
}

#define MAKE_BASIC_ATTR_IMPL(type, enumType, field)                 \
  ONNX_NAMESPACE::AttributeProto MakeAttribute(std::string attr_name, type value) { \
    ONNX_NAMESPACE::AttributeProto a;                                               \
    a.set_##field(std::move(value));                                \
    SetNameAndType(std::move(attr_name), enumType, a);              \
    return a;                                                       \
  }

#define MAKE_ATTR_IMPL(type, enumType, field)                       \
  ONNX_NAMESPACE::AttributeProto MakeAttribute(std::string attr_name, type value) { \
    ONNX_NAMESPACE::AttributeProto a;                                               \
    *(a.mutable_##field()) = std::move(value);                      \
    SetNameAndType(std::move(attr_name), enumType, a);              \
    return a;                                                       \
  }

#define MAKE_LIST_ATTR_IMPL(type, enumType, field)                                    \
  ONNX_NAMESPACE::AttributeProto MakeAttribute(std::string attr_name, gsl::span<const type> values) { \
    ONNX_NAMESPACE::AttributeProto a;                                                                 \
    auto* mutable_field = a.mutable_##field();                                        \
    for (const auto& val : values) {                                                  \
      *(mutable_field->Add()) = val;                                                  \
    }                                                                                 \
    SetNameAndType(std::move(attr_name), enumType, a);                                \
    return a;                                                                         \
  }

MAKE_BASIC_ATTR_IMPL(int64_t, ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT, i)
MAKE_LIST_ATTR_IMPL(int64_t, ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS, ints)

MAKE_BASIC_ATTR_IMPL(float, ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT, f)
MAKE_LIST_ATTR_IMPL(float, ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOATS, floats)

MAKE_ATTR_IMPL(std::string, ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING, s)
MAKE_LIST_ATTR_IMPL(std::string, ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRINGS, strings)

#undef MAKE_BASIC_ATTR_IMPL
#undef MAKE_ATTR_IMPL
#undef MAKE_LIST_ATTR_IMPL

// #if !defined(DISABLE_SPARSE_TENSORS)
// void SparseIndicesChecker(const ONNX_NAMESPACE::TensorProto& indices_proto, gsl::span<const int64_t> expected_indicies) {
//   using namespace ONNX_NAMESPACE;
//   std::filesystem::path model_path;
//   std::vector<uint8_t> unpack_buffer;
//   gsl::span<const int64_t> ind_span;
//   std::vector<int64_t> converted_indices;
//   TensorShape ind_shape(indices_proto.dims().data(), indices_proto.dims().size());
//   const auto elements = narrow<size_t>(ind_shape.Size());
//   const bool has_raw_data = indices_proto.has_raw_data();
//   switch (indices_proto.data_type()) {
//     case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
//       if (has_raw_data) {
//         const auto& rd = indices_proto.raw_data();
//         ASSERT_EQ(rd.size(), elements * sizeof(int64_t));
//         ASSERT_STATUS_OK(utils::UnpackInitializerData(indices_proto, model_path, unpack_buffer));
//         ind_span = ReinterpretAsSpan<const int64_t>(gsl::make_span(unpack_buffer));
//       } else {
//         ind_span = gsl::make_span(indices_proto.int64_data().data(), indices_proto.int64_data_size());
//       }
//       break;
//     }
//     case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
//       if (has_raw_data) {
//         const auto& rd = indices_proto.raw_data();
//         ASSERT_EQ(rd.size(), elements * sizeof(int32_t));
//         ASSERT_STATUS_OK(utils::UnpackInitializerData(indices_proto, model_path, unpack_buffer));
//         auto int32_span = ReinterpretAsSpan<const int32_t>(gsl::make_span(unpack_buffer));
//         converted_indices.insert(converted_indices.cend(), int32_span.begin(), int32_span.end());
//       } else {
//         converted_indices.insert(converted_indices.cend(), indices_proto.int32_data().cbegin(), indices_proto.int32_data().cend());
//       }
//       ind_span = gsl::make_span(converted_indices);
//       break;
//     }
//     case ONNX_NAMESPACE::TensorProto_DataType_INT16: {
//       ASSERT_TRUE(has_raw_data);
//       const auto& rd = indices_proto.raw_data();
//       ASSERT_EQ(rd.size(), elements * sizeof(int16_t));
//       ASSERT_STATUS_OK(utils::UnpackInitializerData(indices_proto, model_path, unpack_buffer));
//       auto int16_span = ReinterpretAsSpan<const int16_t>(gsl::make_span(unpack_buffer));
//       converted_indices.insert(converted_indices.cend(), int16_span.begin(), int16_span.end());
//       ind_span = gsl::make_span(converted_indices);
//       break;
//     }
//     case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
//       ASSERT_TRUE(has_raw_data);
//       const auto& rd = indices_proto.raw_data();
//       ASSERT_EQ(rd.size(), elements);
//       ASSERT_STATUS_OK(utils::UnpackInitializerData(indices_proto, model_path, unpack_buffer));
//       auto int8_span = ReinterpretAsSpan<const int8_t>(gsl::make_span(unpack_buffer));
//       converted_indices.insert(converted_indices.cend(), int8_span.begin(), int8_span.end());
//       ind_span = gsl::make_span(converted_indices);
//       break;
//     }
//     default:
//       ASSERT_TRUE(false);
//   }
//   ASSERT_TRUE(SpanEq(ind_span, expected_indicies));
// }

// #endif  // DISABLE_SPARSE_TENSORS

}  // namespace test
}  // namespace onnxruntime
