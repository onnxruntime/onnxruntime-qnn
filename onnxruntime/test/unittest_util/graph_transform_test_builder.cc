// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/unittest_util/graph_transform_test_builder.h"

#include <functional>
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
      throw std::string("Unhandled zero-point type " + std::to_string(type) + ".");
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
                                                                         const std::string input_name,
                                                                         float input_scale,
                                                                         int64_t input_zero_point,
                                                                         ONNX_NAMESPACE::TensorProto_DataType zero_point_type,
                                                                         const std::string output_name,
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
                                                                           const std::string input_name,
                                                                           float input_scale,
                                                                           int64_t input_zero_point,
                                                                           ONNX_NAMESPACE::TensorProto_DataType zero_point_type,
                                                                           const std::string output_name,
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

// void TransformerTester(const std::function<void(ModelTestBuilder& helper)>& build_test_case,
//                        const std::function<void(InferenceSessionWrapper& session)>& check_transformed_graph,
//                        TransformerLevel baseline_level,
//                        TransformerLevel target_level,
//                        const std::vector<int>& opset_versions,
//                        double per_sample_tolerance,
//                        double relative_per_sample_tolerance,
//                        std::unique_ptr<GraphTransformer> transformer,
//                        const std::function<void(SessionOptions&)>& add_session_options,
//                        const InlinedHashSet<std::string>& disabled_optimizers) {
//   ASSERT_TRUE(transformer == nullptr);
//   for (auto opset_version : opset_versions) {
//     TransformerTester(build_test_case,
//                       check_transformed_graph,
//                       baseline_level,
//                       target_level,
//                       opset_version,
//                       per_sample_tolerance,
//                       relative_per_sample_tolerance,
//                       nullptr,
//                       add_session_options,
//                       disabled_optimizers);
//   }
// }

// void TransformerTester(const std::function<void(ModelTestBuilder& helper)>& build_test_case,
//                        const std::function<void(InferenceSessionWrapper& session)>& check_transformed_graph,
//                        TransformerLevel baseline_level,
//                        TransformerLevel target_level,
//                        int opset_version,
//                        double per_sample_tolerance,
//                        double relative_per_sample_tolerance,
//                        std::unique_ptr<GraphTransformer> transformer,
//                        const std::function<void(SessionOptions&)>& add_session_options,
//                        const InlinedHashSet<std::string>& disabled_optimizers,
//                        std::unique_ptr<IExecutionProvider> ep) {
//   // Build the model for this test.
//   std::unordered_map<std::string, int> domain_to_version;
//   domain_to_version[kOnnxDomain] = opset_version;
//   domain_to_version[kMSDomain] = 1;
//   Model model("TransformerTester", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
//               domain_to_version, {}, DefaultLoggingManager().DefaultLogger());
//   Graph& graph = model.MainGraph();
//   ModelTestBuilder helper(graph);
//   ASSERT_TRUE(build_test_case);
//   build_test_case(helper);
//   helper.SetGraphOutputs();
//   ASSERT_STATUS_OK(model.MainGraph().Resolve());

//   // Serialize the model to a string.
//   std::string model_data;
//   model.ToProto().SerializeToString(&model_data);
//   std::shared_ptr<IExecutionProvider> ep_shared = ep ? std::move(ep) : nullptr;

//   auto run_model = [&](TransformerLevel level, std::vector<OrtValue>& fetches,
//                        std::unique_ptr<GraphTransformer> transformer = nullptr) {
//     // Use public Ort::SessionOptions API
//     Ort::SessionOptions ort_session_options;
//     ort_session_options.SetGraphOptimizationLevel(static_cast<GraphOptimizationLevel>(transformer ? baseline_level : level));
// #if SAVE_TEST_GRAPH
//     ort_session_options.SetOptimizedModelFilePath(ToPathString("model" + std::to_string(static_cast<int>(level)) + ".onnx").c_str());
// #endif
    
//     // Apply custom session options if provided
//     if (add_session_options) {
//       SessionOptions temp_so;
//       add_session_options(temp_so);
//       // Apply relevant settings to Ort::SessionOptions
//       ort_session_options.SetIntraOpNumThreads(temp_so.intra_op_param.thread_pool_size);
//       ort_session_options.SetInterOpNumThreads(temp_so.inter_op_param.thread_pool_size);
//       if (temp_so.enable_profiling) {
//         ort_session_options.EnableProfiling(ORT_TSTR("profile"));
//       }
//       if (temp_so.enable_mem_pattern) {
//         ort_session_options.EnableMemPattern();
//       } else {
//         ort_session_options.DisableMemPattern();
//       }
//       if (temp_so.enable_cpu_mem_arena) {
//         ort_session_options.EnableCpuMemArena();
//       } else {
//         ort_session_options.DisableCpuMemArena();
//       }
//     }
    
//     // Create session using public API
//     OrtSessionWrapper ort_session(*GetOrtEnv(), model_data.data(), static_cast<int>(model_data.size()), ort_session_options);
    
//     // Note: Graph transformer registration and optimizer filtering are not available through public API
//     // These features require internal API access
//     if (transformer || !disabled_optimizers.empty()) {
//       // These operations require internal InferenceSession API
//       // For prebuilt library usage, graph transformations must be done differently
//       ORT_THROW("Graph transformer registration and optimizer filtering require internal API access");
//     }

//     // Run inference using public API
//     Ort::RunOptions ort_run_options;
//     std::vector<OrtValue> temp_fetches;
//     RunWithEPABI(&ort_session, ort_run_options, helper.feeds_, temp_fetches);
//     fetches = std::move(temp_fetches);

//     if (level == target_level) {
//       if (check_transformed_graph) {
//         // Access internal graph through OrtSessionWrapper's GetGraph method
//         // OrtSessionWrapper already provides access to the internal InferenceSession
//         InferenceSessionWrapper* internal_session = reinterpret_cast<InferenceSessionWrapper*>(
//             static_cast<OrtSession*>(ort_session));
//         check_transformed_graph(*internal_session);
//       }
//     }
//   };

//   std::vector<OrtValue> baseline_fetches;
//   ASSERT_NO_FATAL_FAILURE(run_model(baseline_level, baseline_fetches));

//   std::vector<OrtValue> target_fetches;
//   ASSERT_NO_FATAL_FAILURE(run_model(target_level, target_fetches, std::move(transformer)));

//   size_t num_outputs = baseline_fetches.size();
//   ASSERT_EQ(num_outputs, target_fetches.size());

//   for (size_t i = 0; i < num_outputs; i++) {
//     std::pair<COMPARE_RESULT, std::string> ret =
//         CompareOrtValue(target_fetches[i],
//                         baseline_fetches[i],
//                         per_sample_tolerance,
//                         relative_per_sample_tolerance,
//                         false);
//     EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
//   }
// }

}  // namespace test
}  // namespace onnxruntime
