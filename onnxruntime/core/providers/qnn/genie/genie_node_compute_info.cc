// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/genie/genie_node_compute_info.h"

#include <filesystem>
#include <memory>
#include <sstream>

#include "core/providers/qnn/qnn_execution_provider.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {

GenieNodeComputeInfo::GenieNodeComputeInfo(QnnEp& ep,
                                           std::shared_ptr<GenieNodeBuilder> builder)
    : ep(ep), builder(builder) {
  ort_version_supported = ORT_API_VERSION;
  CreateState = CreateStateImpl;
  Compute = ComputeImpl;
  ReleaseState = ReleaseStateImpl;
}

OrtStatus* GenieNodeComputeInfo::CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                 OrtNodeComputeContext* compute_context,
                                                 void** compute_state) {
  auto* node_compute_info = static_cast<GenieNodeComputeInfo*>(this_ptr);
  auto& ep = node_compute_info->ep;
  auto& builder = node_compute_info->builder;
  std::string fused_node_name = ep.ep_api.NodeComputeContext_NodeName(compute_context);
  ORT_CXX_LOG(ep.logger_, ORT_LOGGING_LEVEL_INFO, ("compute_info.create_state_func context->node_name: " + fused_node_name).c_str());

  std::unique_ptr<GenieNodeState, GenieNodeStateDeleter> st(new GenieNodeState(), GenieNodeStateDeleter());
  st->api = builder->api;
  st->num_inputs = builder->num_inputs;
  st->num_outputs = builder->num_outputs;

#if (GENIE_API_VERSION_MAJOR > 1) || (GENIE_API_VERSION_MAJOR == 1 && GENIE_API_VERSION_MINOR >= 17)
  GenieDlcConfig_Handle_t dlc_config_handle = nullptr;
  if (st->api->DlcConfig_create(builder->dlc_path.c_str(), nullptr, &dlc_config_handle) != 0) {
    return ep.ort_api.CreateStatus(ORT_EP_FAIL, "Error creating DLC Config");
  }
  st->dlc_config_handle = dlc_config_handle;

  // Genie DLC Create
  GenieDlc_Handle_t genie_dlc_handle = nullptr;
  if (st->api->Dlc_create(dlc_config_handle, &genie_dlc_handle) != 0) {
    return ep.ort_api.CreateStatus(ORT_EP_FAIL, "Error creating DLC");
  }
  st->dlc_handle = genie_dlc_handle;

  // Generate the Backend Extension json
  auto parent_folder_path = std::filesystem::path(builder->dlc_path).parent_path();

  // Search for the backend extension overrides and pass that as part of the config.
  // The extension file is optional: if the DLC path has no parent directory, or the
  // directory does not exist on disk, we skip the search and proceed without extensions.
  std::string extension_path;
  std::error_code dir_ec;
  const bool parent_is_searchable = !parent_folder_path.empty() &&
                                    std::filesystem::is_directory(parent_folder_path, dir_ec) &&
                                    !dir_ec;
  if (parent_is_searchable) {
    try {
      for (const auto& entry : std::filesystem::directory_iterator(parent_folder_path)) {
        if (entry.is_regular_file()) {
          const std::string filename = entry.path().filename().string();
          if (filename.rfind("tmp", 0) == 0 &&
              filename.size() >= 5 &&
              filename.compare(filename.size() - 5, 5, ".json") == 0) {
            extension_path = entry.path().string();
            break;
          }
        }
      }
    } catch (const std::filesystem::filesystem_error& e) {
      std::string error_msg = std::string("Error searching for extension file: ") + e.what();
      return ep.ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
    }
  } else if (!parent_folder_path.empty()) {
    ORT_CXX_LOG(ep.logger_, ORT_LOGGING_LEVEL_WARNING,
                ("HTP extension directory '" + parent_folder_path.string() +
                 "' is not accessible; continuing without HTP extensions.")
                    .c_str());
  }
  // Replace single backslashes with double backslashes for JSON
  std::string escaped_extension_path;
  for (char c : extension_path) {
    if (c == '\\') {
      escaped_extension_path += "\\\\";
    } else {
      escaped_extension_path += c;
    }
  }
  std::string json_config =
      "{\n"
      "    \"lm-executor\": {\n"
      "        \"version\": 1,\n"
      "        \"engine\": {\n"
      "            \"version\": 1,\n"
      "            \"backend\": {\n"
      "                \"version\": 1,\n"
      "                \"extensions\": \"" +
      escaped_extension_path +
      "\"\n"
      "            }\n"
      "        }\n"
      "    }\n"
      "}";
  GenieNodeConfig_Handle_t cfg = nullptr;
  if (st->api->NodeConfig_createFromDlc(genie_dlc_handle, "default", json_config.c_str(), &cfg) != 0) {
    return ep.ort_api.CreateStatus(ORT_EP_FAIL, "Error creating Node config from dlc");
  }
  st->config = cfg;
#endif  // GENIE_API_VERSION_MAJOR > 1 || GENIE_API_VERSION_MINOR >= 17

  GenieLog_Handle_t g_logger = nullptr;
  const GenieLogConfig_Handle_t cfg_handle = nullptr;
  const GenieLog_Callback_t cb = nullptr;
  const GenieLog_Level_t level = ep.genie_log_level_;

  if (st->api->Log_create(cfg_handle, cb, level, &g_logger) != 0) {
    return ep.ort_api.CreateStatus(ORT_EP_FAIL, "Failed to create Logger");
  }
  st->genie_logger = g_logger;

#if (GENIE_API_VERSION_MAJOR > 1) || (GENIE_API_VERSION_MAJOR == 1 && GENIE_API_VERSION_MINOR >= 17)
  if (st->api->NodeConfig_bindLogger(cfg, g_logger) != 0) {
    return ep.ort_api.CreateStatus(ORT_EP_FAIL, "Failed to bind Logger");
  }

  // 3) Create GenieNode (node)
  GenieNode_Handle_t dlg = nullptr;
  if (st->api->Node_create(cfg, &dlg) != 0) {
    return ep.ort_api.CreateStatus(ORT_EP_FAIL, "Error creating node");
  }
  st->node = dlg;
#endif  // GENIE_API_VERSION_MAJOR > 1 || GENIE_API_VERSION_MINOR >= 17
  *compute_state = static_cast<void*>(st.release());

  return nullptr;
}

OrtStatus* GenieNodeComputeInfo::ComputeImpl(OrtNodeComputeInfo* this_ptr,
                                             void* compute_state,
                                             OrtKernelContext* kernel_context) {
#if (GENIE_API_VERSION_MAJOR > 1) || (GENIE_API_VERSION_MAJOR == 1 && GENIE_API_VERSION_MINOR >= 17)
  auto* node_compute_info = static_cast<GenieNodeComputeInfo*>(this_ptr);
  auto& ep = node_compute_info->ep;
  auto* st = static_cast<GenieNodeState*>(compute_state);
  OrtKernelContext* ctx = kernel_context;
  auto ort_api = &(ep.ort_api);

  RETURN_IF(!st, "Null GenieNodeState");
  RETURN_IF(!st->api, "Null GenieApi");
  RETURN_IF(!st->node, "Null GenieDialog node handle");

  std::lock_guard<std::mutex> guard(st->mu);

  // Reset KV-Cache if required
  const uint64_t rewind_kvcache_value = ep.genie_kv_cache_rewind_.load(std::memory_order_acquire);
  if (rewind_kvcache_value == 0) {
    st->api->Node_reset(st->node);
    // Now reset the value, to prevent repeated rewind
    ep.genie_kv_cache_rewind_.store(1, std::memory_order_release);
  }

  // Set inputs to Genie node
  for (size_t i = 0; i < st->num_inputs; ++i) {
    const OrtValue* in_val = nullptr;
    ORT_CXX_RETURN_ON_API_FAIL(ort_api->KernelContext_GetInput(ctx, i, &in_val));
    RETURN_IF(!in_val, "ORT input is null");

    const void* in_data = nullptr;
    RETURN_IF_NOT_NULL(ort_api->GetTensorData(in_val, &in_data));
    RETURN_IF(!in_data, "ORT input data is null");

    OrtTensorTypeAndShapeInfo* info = nullptr;
    RETURN_IF_NOT_NULL(ort_api->GetTensorTypeAndShape(in_val, &info));
    DeferOrtRelease<OrtTensorTypeAndShapeInfo> defer_info(
        &info, [ort_api](OrtTensorTypeAndShapeInfo* p) {
          ort_api->ReleaseTensorTypeAndShapeInfo(p);
        });
    ONNXTensorElementDataType elem_type;
    RETURN_IF_NOT_NULL(ort_api->GetTensorElementType(info, &elem_type));

    size_t dim_count = 0;
    RETURN_IF_NOT_NULL(ort_api->GetDimensionsCount(info, &dim_count));
    std::vector<int64_t> dims(dim_count);
    RETURN_IF_NOT_NULL(ort_api->GetDimensions(info, dims.data(), dim_count));

    std::ostringstream dim_stream;
    size_t num_elem = 1;
    for (size_t d_idx = 0; d_idx < dims.size(); ++d_idx) {
      RETURN_IF(dims[d_idx] < 0, "Negative tensor dimension");
      num_elem *= static_cast<size_t>(dims[d_idx]);
      dim_stream << dims[d_idx];
      if (d_idx < dims.size() - 1) {
        dim_stream << ",";
      }
    }
    std::string dimString = dim_stream.str();
    std::string input_config = "{\"dimensions\": [" + dimString + "],\"data-type\": \"" + std::string(qnn::utils::GetElementNameByType(elem_type)) + "\"}";
    const char* input_config_ptr = input_config.c_str();
    size_t byte_size = qnn::utils::GetElementSizeByType(elem_type) * num_elem;
    Genie_Status_t rc = st->api->Node_setData(
        st->node,
        GENIE_NODE_LM_EXECUTOR_TOKEN_INPUT,
        in_data,
        byte_size,
        input_config_ptr);

    RETURN_IF(rc != 0, "GenieNode_setData failed");
  }

  // Execute the genie node
  {
    Genie_Status_t rc = st->api->Node_execute(st->node, "{}" /*executionConfig*/, nullptr /*userData*/);
    RETURN_IF(rc != 0, "GenieNode_execute failed");
  }

  // Get output data from the genie node
  for (size_t i = 0; i < st->num_outputs; ++i) {
    struct OutputDataInfo {
      std::vector<std::byte> output_data;
      std::vector<int64_t> output_shape;
    } output_data_info;

    GenieNode_IOCallback_t OutputCallback = [](const void* data,
                                               const size_t dataSize,
                                               const char* outputConfig,
                                               const void* userData) {
      auto* out_data_info = const_cast<OutputDataInfo*>(static_cast<const OutputDataInfo*>(userData));
      out_data_info->output_shape.clear();

      // Parse outputConfig to fetch output shape
      std::string outputInfo = outputConfig;
      size_t firstB = outputInfo.find('[');
      size_t secondB = outputInfo.find(']');
      if (firstB == std::string::npos || secondB == std::string::npos || secondB <= firstB) {
        // Malformed outputConfig — leave output_shape empty so the caller's check catches it.
        return;
      }
      std::string shapeStr = outputInfo.substr(firstB + 1, secondB - firstB - 1);
      std::stringstream ss(shapeStr);
      std::string dim;
      while (std::getline(ss, dim, ',')) {
        out_data_info->output_shape.push_back((int64_t)std::stoi(dim));
      }
      // TODO: Clarify why a dimension of 1 is inserted at index 1 of the output shape.
      out_data_info->output_shape.insert(out_data_info->output_shape.begin() + 1, 1);

      // Set appropriate datasize for output buffer
      out_data_info->output_data.clear();
      out_data_info->output_data.resize(dataSize);
      std::memcpy(out_data_info->output_data.data(), data, dataSize);
    };

    Genie_Status_t rc = st->api->Node_getData(
        st->node,
        GENIE_NODE_LM_EXECUTOR_LOGIT_OUTPUT,
        "{}",
        OutputCallback,
        &output_data_info);

    RETURN_IF(rc != 0, "GenieNode_getData failed");

    OrtValue* out_val = nullptr;
    ORT_CXX_RETURN_ON_API_FAIL(ort_api->KernelContext_GetOutput(
        ctx, i, output_data_info.output_shape.data(), output_data_info.output_shape.size(), &out_val));
    RETURN_IF(!out_val, "ORT output is null");

    void* out_data = nullptr;
    RETURN_IF_NOT_NULL(ort_api->GetTensorMutableData(out_val, &out_data));
    RETURN_IF(!out_data, "ORT output data is null");

    OrtTensorTypeAndShapeInfo* out_info = nullptr;
    RETURN_IF_NOT_NULL(ort_api->GetTensorTypeAndShape(out_val, &out_info));
    ONNXTensorElementDataType out_elem_type;
    RETURN_IF_NOT_NULL(ort_api->GetTensorElementType(out_info, &out_elem_type));
    size_t out_element_count = 0;
    RETURN_IF_NOT_NULL(ort_api->GetTensorShapeElementCount(out_info, &out_element_count));
    ort_api->ReleaseTensorTypeAndShapeInfo(out_info);
    size_t expected_byte_size = qnn::utils::GetElementSizeByType(out_elem_type) * out_element_count;
    RETURN_IF(output_data_info.output_data.size() > expected_byte_size,
              "Genie output data size exceeds ORT tensor buffer");

    std::memcpy(out_data, output_data_info.output_data.data(), output_data_info.output_data.size());
    output_data_info.output_shape.clear();
    output_data_info.output_data.clear();
  }

  return nullptr;
#else
  ORT_UNUSED_PARAMETER(this_ptr);
  ORT_UNUSED_PARAMETER(compute_state);
  ORT_UNUSED_PARAMETER(kernel_context);
  return nullptr;
#endif  // GENIE_API_VERSION_MAJOR > 1 || GENIE_API_VERSION_MINOR >= 17
}

void GenieNodeComputeInfo::ReleaseStateImpl(OrtNodeComputeInfo* this_ptr,
                                            void* compute_state) {
  ORT_UNUSED_PARAMETER(this_ptr);
  std::unique_ptr<GenieNodeState, GenieNodeStateDeleter> state(static_cast<GenieNodeState*>(compute_state));
}

}  // namespace onnxruntime
