// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "nlohmann/json.hpp"
#include "QnnInterface.h"

#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_quant_params_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

// Forward declarations
class BF16ConversionGuard;
class OpTraceCollector;

// Stores information about an ONNX input or output tensor.
// Filled out by QnnModelWrapper::GetTensorInfo()
struct TensorInfo {
  std::vector<uint32_t> shape;
  Qnn_DataType_t qnn_data_type;
  QnnQuantParamsWrapper quant_param;
  bool is_initializer;
  const OrtValueInfo* initializer_tensor;
};

struct ModelSettings {
  bool offload_graph_io_quantization = false;
  bool htp_shared_memory = false;
  bool htp_bf16_enable = false;
  bool enable_block_quant_weight_optimization = false;
};

class QnnModelWrapper {
  // Allow BF16ConversionGuard to access private RestoreFP32AfterValidation method
  friend class BF16ConversionGuard;

 public:
  QnnModelWrapper(const OrtGraph& ort_graph,
                  const ApiPtrs& api_ptrs,
                  const Ort::Logger& logger,
                  const QNN_INTERFACE_VER_TYPE& qnn_interface,
                  const Qnn_BackendHandle_t& backend_handle,
                  const QNN_INTERFACE_VER_TYPE& qnn_validator_interface,
                  const Qnn_BackendHandle_t& validator_backend_handle,
                  const GraphInputOutputInfo& graph_inputs,
                  const GraphInputOutputInfo& graph_outputs,
                  QnnBackendType qnn_backend_type,
                  const ModelSettings& model_settings,
                  std::unordered_map<std::string, std::string>* tensor_name_overrides = nullptr,
                  OpTraceCollector* op_trace_collector = nullptr)
      : ort_graph_(ort_graph),
        logger_(logger),
        qnn_interface_(qnn_interface),
        backend_handle_(backend_handle),
        qnn_validator_interface_(qnn_validator_interface),
        validator_backend_handle_(validator_backend_handle),
        graph_inputs_(graph_inputs),
        graph_outputs_(graph_outputs),
        qnn_backend_type_(qnn_backend_type),
        model_settings_(model_settings),
        api_ptrs_(ApiPtrs{api_ptrs.ort_api, api_ptrs.ep_api, api_ptrs.model_editor_api}),
        tensor_name_overrides_(tensor_name_overrides),
        op_trace_collector_(op_trace_collector) {
    // Invariant: validator interface and handle must both be set or both be null.
    // They are populated together by QnnBackendManager::LoadQnnSerializerBackend() (QnnIr flow).
    assert((validator_backend_handle == nullptr) ==
           (qnn_validator_interface.backendValidateOpConfig == nullptr));
  }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QnnModelWrapper);

  ~QnnModelWrapper() = default;

  const ModelSettings& GetModelSettings() const { return model_settings_; }

  bool CreateQnnGraph(const Qnn_ContextHandle_t& context,
                      const std::string& graph_name,
                      const QnnGraph_Config_t** graph_configs = nullptr);

  // Make a QnnTensorWrapper from an onnx input or output.
  Ort::Status MakeTensorWrapper(const OrtNodeUnitIODef& tensor, QnnTensorWrapper& tensor_wrapper) const;
  Ort::Status MakeTensorWrapper(const TensorInfo& tensor_info,
                                const std::string& tensor_name,
                                QnnTensorWrapper& tensor_wrapper) const;

  // Sets the QNN tensor memory type based on model settings and whether the tensor is a graph input/output.
  void SetTensorMemTypeFromSettings(QnnTensorWrapper& tensor_wrapper, const std::string& tensor_name);

  // Add to internal tensor wrapper table
  bool AddTensorWrapper(QnnTensorWrapper&& tensor_wrapper);

  // Add to internal param wrapper table
  bool AddParamWrapper(QnnParamWrapper&& param_wrapper);

  const QnnTensorWrapper& GetQnnTensorWrapper(const std::string& tensor_name);

  // Utility function to validate a QNN node. Does not modify this object's state.
  Ort::Status ValidateQnnNode(const std::string& node_name,
                              const std::string& package_name,
                              const std::string& qnn_op_type,
                              std::vector<Qnn_Tensor_t>&& input_tensors,
                              std::vector<Qnn_Tensor_t>&& output_tensors,
                              std::vector<Qnn_Param_t>&& params) const;

  bool CreateQnnNode(const std::string& name,
                     const std::string& package_name,
                     const std::string& type,
                     std::vector<std::string>&& input_names,
                     std::vector<std::string>&& output_names,
                     std::vector<std::string>&& param_tensor_names,
                     bool do_op_validation = false);

  bool ComposeQnnGraph(bool build_json_qnn_graph = false);

  Qnn_GraphHandle_t GetQnnGraph() const { return graph_; }

  std::string GetQnnGraphName() const { return graph_name_; }

  Qnn_ContextHandle_t GetQnnGraphContext() const { return graph_context_; }

  std::vector<QnnTensorWrapper>&& GetGraphInputTensorWrappers() {
    GetGraphInputOutputTensorWrapper(graph_inputs_.names, model_input_tensor_wrappers_);
    return std::move(model_input_tensor_wrappers_);
  }

  std::vector<QnnTensorWrapper>&& GetGraphOutputTensorWrappers() {
    GetGraphInputOutputTensorWrapper(graph_outputs_.names, model_output_tensor_wrappers_);
    return std::move(model_output_tensor_wrappers_);
  }

  Ort::Status GetInitializerTensors(gsl::span<const OrtValueInfo*> initializers) const {
    ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.Graph_GetInitializers(&ort_graph_,
                                                                       initializers.data(),
                                                                       initializers.size()));
    return Ort::Status();
  }

  // Find an initializer by name
  Ort::Status FindInitializer(const std::string& tensor_name,
                              const OrtValueInfo** found_value_info = nullptr) const {
    // Walk the branch graph scope stack from innermost to outermost so that initializers
    // defined inside an `If`/`Loop` branch shadow same-named initializers in the parent graph.
    for (auto it = branch_graph_scope_stack_.rbegin(); it != branch_graph_scope_stack_.rend(); ++it) {
      if (FindInitializerInGraph(*it, tensor_name, found_value_info).IsOK()) {
        return Ort::Status();
      }
    }
    return FindInitializerInGraph(&ort_graph_, tensor_name, found_value_info);
  }

  // Push the given branch graph onto the scope stack. While pushed, FindInitializer (and
  // therefore IsConstantInput / GetConstantTensor) will also consider this graph's
  // initializers. Used by IfOpBuilder when recursively translating branch nodes so that
  // ORT-folded branch-internal Constants (which live as initializers on the branch graph)
  // resolve as STATIC tensors during op-builder dispatch.
  void PushBranchGraphScope(const OrtGraph* branch_graph) {
    branch_graph_scope_stack_.push_back(branch_graph);
  }

  void PopBranchGraphScope() {
    if (!branch_graph_scope_stack_.empty()) {
      branch_graph_scope_stack_.pop_back();
    }
  }

  const OrtValueInfo* GetConstantTensor(const std::string& tensor_name) const {
    const OrtValueInfo* value_info = nullptr;
    Ort::Status status = FindInitializer(tensor_name, &value_info);
    if (!status.IsOK() || value_info == nullptr) {
      return nullptr;
    }

    bool is_constant_initializer = false;
    OrtStatus* ort_status = api_ptrs_.ort_api.ValueInfo_IsConstantInitializer(value_info, &is_constant_initializer);
    if (ort_status != nullptr) {
      api_ptrs_.ort_api.ReleaseStatus(ort_status);
      return nullptr;
    }

    if (!is_constant_initializer) {
      return nullptr;
    }

    return value_info;
  }

  // This function aims to check if the input to a node is an initializer.
  // Note: the `input` here refers to the input of the node, not the input of the graph.
  bool IsConstantInput(const std::string& input_name) const {
    const OrtValueInfo* value_info = nullptr;
    Ort::Status status = FindInitializer(input_name, &value_info);
    if (!status.IsOK() || value_info == nullptr) {
      return false;
    }

    bool is_constant_initializer = false;
    OrtStatus* ort_status = api_ptrs_.ort_api.ValueInfo_IsConstantInitializer(value_info, &is_constant_initializer);
    if (ort_status != nullptr) {
      api_ptrs_.ort_api.ReleaseStatus(ort_status);
      return false;
    }

    return is_constant_initializer;
  }

  void MarkTensorAsFoldedConstant(const std::string& tensor_name) {
    folded_constant_tensors_.insert(tensor_name);
  }

  bool IsFoldedConstant(const std::string& tensor_name) const {
    return folded_constant_tensors_.count(tensor_name) > 0;
  }

  // Real graph initializer OR a tensor produced by a previous compile-time fold.
  bool IsEffectivelyConstantInput(const std::string& tensor_name) const {
    return IsConstantInput(tensor_name) || IsFoldedConstant(tensor_name);
  }

  // static bool GetOnnxShape(const NodeArg& node_arg, std::vector<uint32_t>& shape);
  static bool GetOnnxShape(const std::optional<std::vector<int64_t>>& onnx_shape, std::vector<uint32_t>& shape);

  bool IsQnnTensorWrapperExist(const std::string& tensor_name) const;

  bool IsGraphOutput(const std::string& tensor_name) const {
    return graph_outputs_.indices.find(tensor_name) != graph_outputs_.indices.end();
  }

  bool IsGraphInput(const std::string& tensor_name) const {
    return graph_inputs_.indices.find(tensor_name) != graph_inputs_.indices.end();
  }

  const nlohmann::json& GetQnnJSONGraph() {
    return json_qnn_graph_.Finalize();
  }

  Qnn_TensorType_t GetTensorType(const std::string& tensor_name) const {
    if (IsConstantInput(tensor_name) || IsFoldedConstant(tensor_name)) {
      return QNN_TENSOR_TYPE_STATIC;
    } else if (IsGraphInput(tensor_name)) {
      return QNN_TENSOR_TYPE_APP_WRITE;
    } else if (IsGraphOutput(tensor_name)) {
      return QNN_TENSOR_TYPE_APP_READ;
    } else {
      return QNN_TENSOR_TYPE_NATIVE;
    }
  }

  Ort::Status GetTensorInfo(const OrtNodeUnitIODef& tensor, TensorInfo& tensor_info) const;

  Ort::Status AddCastNode(const std::string& cast_node_name,
                          const std::string& input_name,
                          const std::string& output_name,
                          Qnn_TensorType_t output_tensor_type,
                          Qnn_DataType_t output_data_type,
                          QnnQuantParamsWrapper&& output_quant_params,
                          std::vector<uint32_t>&& output_shape,
                          bool do_op_validation);

  // Adds a QNN_OP_DEQUANTIZE node: input (quantized) → output (float).
  // output_data_type must be FLOAT_16 or FLOAT_32.
  // Output tensor type is always QNN_TENSOR_TYPE_NATIVE.
  Ort::Status AddDequantizeNode(const std::string& input_name,
                                const std::string& output_name,
                                Qnn_DataType_t output_data_type,
                                std::vector<uint32_t> output_shape,
                                bool do_op_validation);

  // Adds a QNN_OP_QUANTIZE node: input (float) → output (fixed-point).
  // output_data_type must be a SFIXED_POINT or UFIXED_POINT type.
  Ort::Status AddQuantizeNode(const std::string& input_name,
                              const std::string& output_name,
                              Qnn_TensorType_t output_tensor_type,
                              Qnn_DataType_t output_data_type,
                              QnnQuantParamsWrapper output_quant_param,
                              std::vector<uint32_t> output_shape,
                              bool do_op_validation);

  Ort::Status AddReshapeNode(const std::string& input_name,
                             const std::string& output_name,
                             const std::vector<uint32_t>& input_shape,
                             const std::vector<uint32_t>& output_shape,
                             const Qnn_DataType_t& tensor_data_type,
                             const QnnQuantParamsWrapper& input_quantize_param,
                             const QnnQuantParamsWrapper& output_quantize_param,
                             bool do_op_validation,
                             bool is_for_input = true,
                             bool is_for_output = false);

  Ort::Status AddReshapeNode(const std::string& input_name,
                             const std::string& output_name,
                             const std::vector<uint32_t>& input_shape,
                             const std::vector<uint32_t>& output_shape,
                             const Qnn_DataType_t& tensor_data_type,
                             const QnnQuantParamsWrapper& quantize_param,
                             bool do_op_validation,
                             bool is_for_input = true,
                             bool is_for_output = false);

  Ort::Status AddTransposeNode(size_t node_index,
                               const std::string& input_name,
                               const std::string& output_name,
                               const std::vector<uint32_t>& input_shape,
                               const std::vector<uint32_t>& transpose_perm,
                               const std::vector<uint32_t>& output_shape,
                               const Qnn_DataType_t& tensor_data_type,
                               const QnnQuantParamsWrapper& quantize_param,
                               bool do_op_validation,
                               bool is_for_input = true,
                               bool is_for_output = false);

  // Add Reshape node with identical input/output shapes to handle no-op node.
  // This is a workaround for no-op nodes as there is no mechanism to reroute wrappers inside op builder currently.
  Ort::Status AddNoopReshapeNode(const std::string& node_name,
                                 const std::string& input_name,
                                 const OrtNodeUnitIODef& output,
                                 bool do_op_validation);

  // Transpose NCHW->HWCN for QNN weight
  Ort::Status AddNchwToHwcnTranspose(size_t node_index,
                                     const std::string& input_name,
                                     const std::string& output_name,
                                     const std::vector<uint32_t>& input_shape,
                                     const std::vector<uint32_t>& output_shape,
                                     const Qnn_DataType_t& tensor_data_type,
                                     const QnnQuantParamsWrapper& quantize_param,
                                     bool do_op_validation,
                                     bool is_for_input = true,
                                     bool is_for_output = false,
                                     bool is_3d = false) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Add NCHW->HWCN Transpose node after Conv weight input: " +
                 input_name + " -> " + output_name)
                    .c_str());
    auto perm = is_3d ? nchw2hwcn_perm_3d : nchw2hwcn_perm;
    std::vector<uint32_t> transpose_perm;
    transpose_perm.resize(perm.size());
    std::transform(perm.begin(), perm.end(),
                   transpose_perm.begin(), [](size_t item) -> uint32_t {
                     return gsl::narrow<uint32_t>(item);
                   });
    return AddTransposeNode(node_index, input_name, output_name, input_shape, transpose_perm, output_shape,
                            tensor_data_type, quantize_param, do_op_validation, is_for_input, is_for_output);
  }

  // Tranpose CNHW->HWCN for QNN weight
  Ort::Status AddCnhwToHwcnTranspose(size_t node_index,
                                     const std::string& input_name,
                                     const std::string& output_name,
                                     const std::vector<uint32_t>& input_shape,
                                     const std::vector<uint32_t>& output_shape,
                                     const Qnn_DataType_t& tensor_data_type,
                                     const QnnQuantParamsWrapper& quantize_param,
                                     bool do_op_validation,
                                     bool is_for_input = true,
                                     bool is_for_output = false,
                                     bool is_3d = false) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Add CNHW->HWCN Transpose node after ConvTranspose weight input: " +
                 input_name + " -> " + output_name)
                    .c_str());
    auto perm = is_3d ? cnhw2hwcn_perm_3d : cnhw2hwcn_perm;
    std::vector<uint32_t> transpose_perm;
    transpose_perm.resize(perm.size());
    std::transform(perm.begin(), perm.end(),
                   transpose_perm.begin(), [](size_t item) -> uint32_t {
                     return gsl::narrow<uint32_t>(item);
                   });
    return AddTransposeNode(node_index, input_name, output_name, input_shape, transpose_perm, output_shape,
                            tensor_data_type, quantize_param, do_op_validation, is_for_input, is_for_output);
  }

  Ort::Status UnpackInitializerData(const OrtValueInfo* initializer,
                                    std::vector<uint8_t>& unpacked_tensor,
                                    const bool unpack_sub_byte_to_8_bit = true) const;

  QnnBackendType GetQnnBackendType() const { return qnn_backend_type_; }

  const OrtGraph& GetOrtGraph() const { return ort_graph_; }

  const Ort::Logger& GetLogger() const { return logger_; }

  const std::unordered_map<std::string, QnnTensorWrapper>& GetModelTensorsMap() const {
    return model_tensors_map_;
  }

  const OrtApi& GetOrtApi() const { return api_ptrs_.ort_api; }

  // Unpack scales from initializer (1 scale for per-tensor, > 1 for per-axis or per-block).
  // Template parameter T allows handling both float and uint8_t scale types.
  template <typename T = float>
  Ort::Status UnpackScales(const OrtValueInfo* scale_tensor, std::vector<T>& scales) const {
    RETURN_IF(scale_tensor == nullptr, "Given scale(s) to be unpacked is null.");

    const OrtTypeInfo* type_info = nullptr;
    ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.GetValueInfoTypeInfo(scale_tensor, &type_info));

    const OrtTensorTypeAndShapeInfo* tensor_type_and_shape_info = nullptr;
    ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.CastTypeInfoToTensorInfo(type_info, &tensor_type_and_shape_info));
    ONNXTensorElementDataType onnx_data_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.GetTensorElementType(tensor_type_and_shape_info, &onnx_data_type));

    // Handle float scales
    if constexpr (std::is_same_v<T, float>) {
      // Validate dtype up front so an unsupported type returns before any (potentially external)
      // initializer read, mirroring the early RETURN_IF_NOT guard in the uint8_t branch below.
      RETURN_IF_NOT(onnx_data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                        onnx_data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
                        onnx_data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
                    "Expected scale initializer to be of type FLOAT, FLOAT16, or BFLOAT16");

      std::vector<uint8_t> initializer_bytes;
      RETURN_IF_ERROR(UnpackInitializerData(scale_tensor, initializer_bytes));

      // Reinterpret the raw bytes as SrcT and append each element as a float. static_cast covers
      // float (identity), Ort::Float16_t, and Ort::BFloat16_t (both have operator float()). fp16/bf16
      // scale initializers are produced by quantization tools that match the scale dtype to the
      // activation dtype (e.g. fp16 models); QNN quantization structs use float, so decode here.
      auto append_scales = [&scales, &initializer_bytes](auto src_type_tag) {
        using SrcT = decltype(src_type_tag);
        gsl::span<const SrcT> src = gsl::make_span(reinterpret_cast<const SrcT*>(initializer_bytes.data()),
                                                   initializer_bytes.size() / sizeof(SrcT));
        scales.reserve(scales.size() + src.size());
        for (const auto& val : src) {
          scales.push_back(static_cast<float>(val));
        }
      };

      if (onnx_data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        append_scales(float{});
      } else if (onnx_data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        append_scales(Ort::Float16_t{});
      } else if (onnx_data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
        // Spelled out (rather than a bare else) so a future dtype added to the guard above
        // without a matching branch here fails to compile instead of silently aliasing BFLOAT16.
        append_scales(Ort::BFloat16_t{});
      }
    }
    // Handle uint8_t scales (for block quantization)
    else if constexpr (std::is_same_v<T, uint8_t>) {
      // Verify data type for uint8_t scales
      RETURN_IF_NOT(onnx_data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
                    "Expected uint8_t scale initializer to be of type UINT8");

      RETURN_IF_ERROR(UnpackInitializerData(scale_tensor, scales));
    } else {
      return MAKE_EP_FAIL(("Scale ONNX data type `" +
                           std::string(typeid(T).name()) +
                           "` is not supported for unpacking.")
                              .c_str());
    }
    return Ort::Status();
  }

  // Unpack zero-points from initializer and convert to int32_t (1 zero-point for per-tensor, > 1 for per-channel).
  Ort::Status UnpackZeroPoints(const OrtValueInfo* zp_tensor,
                               /*out*/ std::vector<int32_t>& zero_points,
                               /*out*/ ONNXTensorElementDataType& onnx_data_type) const;

  // // Checks if a tensor in the ONNX graph is per-channel quantized.
  Ort::Status IsPerChannelQuantized(const OrtNodeUnitIODef& io_def,
                                    /*out*/ bool& is_per_channel,
                                    /*out*/ int64_t& axis) const;

  // Maps internal QNN tensor name to original ONNX name. Used by offload_graph_io_quantization.
  void SetTensorNameOverride(const std::string& internal, const std::string& original) const;

  // Returns true if 'external' is already the target of any tensor name override.
  // Used to prevent two Q-node outputs from mapping to the same external graph-input name,
  // which would cause RegisterGraphInputOutputInOrder to create only one APP_WRITE tensor
  // while graph_inputs_ still tracks two entries, leaving a null slot in qnn_tensor_infos.
  bool IsExternalOverrideTarget(const std::string& external) const {
    if (tensor_name_overrides_ == nullptr) return false;
    for (const auto& [internal, ext] : *tensor_name_overrides_) {
      if (ext == external) return true;
    }
    return false;
  }

 private:
  // Searches a single OrtGraph's initializer table for an entry with `tensor_name`.
  // Returns OK on hit (with `found_value_info` populated) and an EP_FAIL on miss.
  Ort::Status FindInitializerInGraph(const OrtGraph* graph,
                                     const std::string& tensor_name,
                                     const OrtValueInfo** found_value_info) const {
    size_t num_initializers = 0;
    ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.Graph_GetNumInitializers(graph, &num_initializers));

    std::vector<const OrtValueInfo*> initializers(num_initializers);
    if (num_initializers > 0) {
      ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.Graph_GetInitializers(graph,
                                                                         initializers.data(),
                                                                         initializers.size()));
    }

    for (const OrtValueInfo* value_info : initializers) {
      const char* value_info_name = nullptr;
      ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.GetValueInfoName(value_info, &value_info_name));
      if (std::string(value_info_name) == tensor_name) {
        if (found_value_info != nullptr) {
          *found_value_info = value_info;
        }
        return Ort::Status();
      }
    }

    return MAKE_EP_FAIL("Initializer not found");
  }

  Ort::Status ValidateQnnNode(QnnOpConfigWrapper& op_config_wrapper,
                              std::string& error_msg) const;

  bool CreateQnnInputOutputTensors(const std::string& qnn_node_name,
                                   const std::vector<std::string>& names,
                                   std::vector<Qnn_Tensor_t>& tensor_wrappers,
                                   bool do_op_validation = false);

  bool QnnParamExists(const std::string& param_tensor_name) const;

  bool CreateQnnParamTensors(const std::string& qnn_node_name,
                             const std::vector<std::string>& param_tensor_names,
                             std::vector<Qnn_Param_t>& qnn_params,
                             bool do_op_validation = false);

  void GetGraphInputOutputTensorWrapper(const std::vector<std::string>& names,
                                        std::vector<QnnTensorWrapper>& wrappers_list);

  // Register graph inputs and outputs in ONNX declaration order. Must be called before
  // processing ops, as tensorCreateGraphTensor() order determines tensor IDs in DLC.
  bool RegisterGraphInputOutputInOrder();

  // BF16 conversion helper methods
  bool IsBF16ConversionEnabled() const {
    return model_settings_.htp_bf16_enable &&
           (qnn_backend_type_ == QnnBackendType::HTP || qnn_backend_type_ == QnnBackendType::SERIALIZER);
  }

  bool ProcessBF16InputConversion(const std::string& qnn_node_name,
                                  const std::vector<std::string>& input_names,
                                  std::vector<std::string>& converted_input_names,
                                  std::vector<QnnOpProperty>& cast_ops_to_add);

  bool ProcessBF16OutputConversion(const std::string& qnn_node_name,
                                   const std::vector<std::string>& output_names,
                                   std::vector<std::string>& converted_output_names,
                                   std::vector<std::pair<std::string, std::string>>& graph_output_cast_ops);

  bool ApplyBF16ConversionForValidation(const std::vector<std::string>& input_names,
                                        const std::vector<std::string>& output_names,
                                        std::vector<std::string>& validation_input_names,
                                        std::vector<std::string>& validation_output_names);

  void RestoreFP32AfterValidation(const std::vector<std::string>& input_names,
                                  const std::vector<std::string>& output_names);

  bool CreateBF16CastTensor(const std::string& tensor_name,
                            const std::vector<uint32_t>& shape,
                            Qnn_TensorType_t tensor_type);

  bool ProcessBF16Conversions(std::vector<QnnOpProperty>& final_ops);

  const std::string* GetTensorNameOverride(const std::string& internal) const;

  const OrtGraph& ort_graph_;
  const Ort::Logger& logger_;
  const QNN_INTERFACE_VER_TYPE& qnn_interface_;
  const Qnn_BackendHandle_t& backend_handle_;
  const QNN_INTERFACE_VER_TYPE& qnn_validator_interface_;
  const Qnn_BackendHandle_t& validator_backend_handle_;
  Qnn_GraphHandle_t graph_ = nullptr;
  std::string graph_name_ = "";
  // QNN context that holds the QNN graph referenced by `graph_`
  Qnn_ContextHandle_t graph_context_ = nullptr;

  std::vector<QnnTensorWrapper> model_input_tensor_wrappers_;
  std::vector<QnnTensorWrapper> model_output_tensor_wrappers_;
  // All QnnTensorWrapper for the graph
  std::unordered_map<std::string, QnnTensorWrapper> model_tensors_map_;
  // All QnnParamWrapper for the graph
  std::unordered_map<std::string, QnnParamWrapper> model_params_map_;
  std::vector<QnnOpProperty> qnn_op_property_list_;
  // <tensor_name, qnn_tensor_id> -- stores the QNN-assigned ID once the tensor is created
  // it includes normal qnn_tensors and qnn_tensors inside param_tensors
  std::unordered_map<std::string, uint32_t> qnn_tensor_id_map_;
  const GraphInputOutputInfo& graph_inputs_;
  const GraphInputOutputInfo& graph_outputs_;
  QnnBackendType qnn_backend_type_ = QnnBackendType::CPU;
  ModelSettings model_settings_ = {};
  utils::QnnJSONGraph json_qnn_graph_;
  const ApiPtrs api_ptrs_;

  std::unordered_map<std::string, std::string>* tensor_name_overrides_ = nullptr;

  // Tensor names produced by compile-time Q/DQ folds; chained across hops.
  std::unordered_set<std::string> folded_constant_tensors_;

  // Stack of branch graphs (e.g., If::then_branch / else_branch) currently being translated.
  // FindInitializer walks this from top to bottom before falling back to ort_graph_, which
  // lets op-builders dispatched on branch nodes resolve branch-internal initializers
  // (notably ORT-folded Constants) as STATIC tensors.
  std::vector<const OrtGraph*> branch_graph_scope_stack_;

  // Non-owning pointer to the trace collector. Lifetime is managed by
  // QnnModel::ComposeGraph (stack-allocated unique_ptr).
  // Null when tracing is disabled.
  OpTraceCollector* op_trace_collector_ = nullptr;
};  // QnnModelWrapper

template <typename T>
inline Ort::Status AddQnnScalar(QnnModelWrapper& qnn_model_wrapper,
                                const size_t& node_index,
                                const std::string& node_name,
                                const T& scalar,
                                const std::string& qnn_scalar_param_name,
                                std::vector<std::string>& param_names) {
  Qnn_Scalar_t qnn_scalar = QNN_SCALAR_INIT;
  if (std::is_same<T, float>::value) {
    qnn_scalar.dataType = QNN_DATATYPE_FLOAT_32;
    qnn_scalar.floatValue = static_cast<float>(scalar);
  } else if (std::is_same<T, uint32_t>::value) {
    qnn_scalar.dataType = QNN_DATATYPE_UINT_32;
    qnn_scalar.uint32Value = static_cast<uint32_t>(scalar);
  } else if (std::is_same<T, int32_t>::value) {
    qnn_scalar.dataType = QNN_DATATYPE_INT_32;
    qnn_scalar.int32Value = static_cast<int32_t>(scalar);
  } else if (std::is_same<T, int64_t>::value) {
    qnn_scalar.dataType = QNN_DATATYPE_INT_64;
    qnn_scalar.int64Value = static_cast<int64_t>(scalar);
  } else if (std::is_same<T, bool>::value) {
    qnn_scalar.dataType = QNN_DATATYPE_BOOL_8;
    qnn_scalar.bool8Value = static_cast<uint8_t>(scalar);
  } else {
    RETURN_IF(true, "QNN EP: Unsupported scalar dtype");
  }
  QnnParamWrapper qnn_param_wrapper(node_index, node_name, qnn_scalar_param_name, qnn_scalar);
  std::string param_tensor_name = qnn_param_wrapper.GetParamTensorName();
  RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(qnn_param_wrapper)),
                ("QNN EP: Failed to add scalar param " + qnn_scalar_param_name).c_str());
  param_names.push_back(std::move(param_tensor_name));
  return Ort::Status();
}

inline Ort::Status AddQnnScalar(QnnModelWrapper& qnn_model_wrapper,
                                const size_t& node_index,
                                const std::string& node_name,
                                const std::string& scalar,
                                const std::string& qnn_scalar_param_name,
                                std::vector<std::string>& param_names) {
  Qnn_Scalar_t qnn_scalar = QNN_SCALAR_INIT;
  qnn_scalar.dataType = QNN_DATATYPE_STRING;
  qnn_scalar.stringValue = scalar.c_str();
  QnnParamWrapper qnn_param_wrapper(node_index, node_name, qnn_scalar_param_name, qnn_scalar);
  std::string param_tensor_name = qnn_param_wrapper.GetParamTensorName();
  RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(qnn_param_wrapper)),
                ("QNN EP: Failed to add scalar param " + qnn_scalar_param_name).c_str());
  param_names.push_back(std::move(param_tensor_name));
  return Ort::Status();
}

// RAII guard to ensure FP32 restoration after BF16 conversion for validation
class BF16ConversionGuard {
 public:
  BF16ConversionGuard(QnnModelWrapper* wrapper,
                      const std::vector<std::string>& input_names,
                      const std::vector<std::string>& output_names)
      : wrapper_(wrapper),
        input_names_(input_names),
        output_names_(output_names) {}

  ~BF16ConversionGuard() {
    if (wrapper_) {
      try {
        wrapper_->RestoreFP32AfterValidation(input_names_, output_names_);
      } catch (...) {
        // Destructors must not throw exceptions
        // Silently catch any exceptions during cleanup
      }
    }
  }

  // Prevent copying
  BF16ConversionGuard(const BF16ConversionGuard&) = delete;
  BF16ConversionGuard& operator=(const BF16ConversionGuard&) = delete;

  // Prevent moving to avoid double-cleanup issues
  BF16ConversionGuard(BF16ConversionGuard&&) = delete;
  BF16ConversionGuard& operator=(BF16ConversionGuard&&) = delete;

 private:
  QnnModelWrapper* wrapper_;
  std::vector<std::string> input_names_;   // Store by value, not reference
  std::vector<std::string> output_names_;  // Store by value, not reference
};

// Adds ElementWiseNeuron operation=HARD_SWISH param to the model wrapper
// alpha/beta are not accepted by HTP and hence are not explicitly set here
inline void AddHardSwishNeuronParams(QnnModelWrapper& qnn_model_wrapper,
                                     size_t node_index,
                                     const std::string& node_name,
                                     std::vector<std::string>& param_tensor_names) {
  Qnn_Scalar_t neuron_operation = QNN_SCALAR_INIT;
  neuron_operation.dataType = QNN_DATATYPE_UINT_32;
  neuron_operation.uint32Value = QNN_OP_ELEMENT_WISE_NEURON_OPERATION_HARD_SWISH;
  QnnParamWrapper operation_param(node_index, node_name,
                                  QNN_OP_ELEMENT_WISE_NEURON_PARAM_OPERATION, neuron_operation);
  param_tensor_names.push_back(operation_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(operation_param));
}

}  // namespace qnn
}  // namespace onnxruntime
