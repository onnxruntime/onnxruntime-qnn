// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/util/qmath.h"
#include "core/providers/shared/utils/utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/op_builder_factory.h"

#include "base_op_builder.h"

#include <limits>
#include <cmath>
#include <utility>

namespace onnxruntime {
namespace qnn {
class BatchNormOpBuilder : public BaseOpBuilder {
 public:
  BatchNormOpBuilder() : BaseOpBuilder("BatchNormOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(BatchNormOpBuilder);

  Status ProcessQDQInputs(QnnModelWrapper& qnn_model_wrapper,
                          const NodeUnit& node_unit,
                          const logging::Logger& logger,
                          std::vector<std::string>& input_names,
                          bool do_op_validation) const ORT_MUST_USE_RESULT;

  Status ProcessFPInputs(QnnModelWrapper& qnn_model_wrapper,
                          const NodeUnit& node_unit,
                          const logging::Logger& logger,
                          std::vector<std::string>& input_names,
                          bool do_op_validation) const ORT_MUST_USE_RESULT;

  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                        const NodeUnit& node_unit,
                        const logging::Logger& logger,
                        std::vector<std::string>& input_names,
                        bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger) const override final ORT_MUST_USE_RESULT;
  std::pair<float, int> GetQuantParams(float rmin, float rmax) const {
    // Ensure a minimum range of 0.0001 (required by QNN)
    rmax = std::max(rmax, rmin + 0.0001f);

    // Both QNN and ORT require the range to include 0.0f
    rmin = std::min(rmin, 0.0f);
    rmax = std::max(rmax, 0.0f);

    constexpr float qmin = static_cast<float>(std::numeric_limits<uint8_t>::min());
    constexpr float qmax = static_cast<float>(std::numeric_limits<uint8_t>::max());

    const float scale = rmax == rmin ? 1.0f : (rmax - rmin) / (qmax - qmin);
    const float initial_zero_point = qmin - (rmin / scale);
    const int zero_point = static_cast<int>(RoundHalfToEven(std::max(qmin, std::min(qmax, initial_zero_point))));

    return std::make_pair(scale, zero_point);
  }
};

// BatchNorm is sensitive with data layout, no special validation so far
// The nodes from 1st call of GetCapability do not get layout transformer applied, it's still NCHW
// The nodes from 2nd call of GetCapability get layout transformer applied, it's NHWC
Status BatchNormOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                         const NodeUnit& node_unit,
                                         const logging::Logger& logger) const {
  if (node_unit.Domain() == kMSInternalNHWCDomain) {
    // It's useless to fallback the node after layout transformation because CPU EP can't support it anyway
    // Still do it here so hopefully QNN Op validation API can tell us some details why it's not supported
    return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
  } else {
    NodeAttrHelper node_helper(node_unit);
    const float default_epsilon = 1e-05f;
    const float epsilon = node_helper.Get("epsilon", 1e-05f);  // Default is 1e-05 according to ONNX spec.
    ORT_RETURN_IF(abs(epsilon - default_epsilon) > default_epsilon, "QNN BatchNorm doesn't support epsilon.");

    const auto& inputs = node_unit.Inputs();
    ORT_ENFORCE(inputs.size() == 5, "5 input expected per BatchNorm Onnx Spec.");

    // Check input type is float for CPU. Can't use Qnn Op validation API since it's before layout transformation
    ORT_RETURN_IF_ERROR(DataTypeCheckForCpuBackend(qnn_model_wrapper, inputs[0].node_arg.Type()));

    std::vector<uint32_t> input_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].node_arg, input_shape), "Cannot get shape of input 0.");
    const size_t input_rank = input_shape.size();

    ORT_RETURN_IF(input_rank <= 2 || input_rank > 4,
                  "QNN BatchNorm only supports input ranks of size 3 or 4.");

    const uint32_t num_channels = input_shape[1];

    std::vector<uint32_t> scale_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[1].node_arg, scale_shape), "Cannot get shape of input 1 (scale).");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.IsInitializerInput(inputs[1].node_arg.Name()), "QNN BatchNorm doesn't support dynamic scale.");
    ORT_RETURN_IF(scale_shape.size() != 1 || scale_shape[0] != num_channels,
                  "QNN BatchNorm input 1 (scale) must have 1D shape [channel].");

    std::vector<uint32_t> bias_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[2].node_arg, bias_shape), "Cannot get shape of input 2 (bias).");
        ORT_RETURN_IF_NOT(qnn_model_wrapper.IsInitializerInput(inputs[2].node_arg.Name()), "QNN BatchNorm doesn't support dynamic bias.");

    ORT_RETURN_IF(bias_shape.size() != 1 || bias_shape[0] != num_channels,
                  "QNN BatchNorm input 2 (bias) must have 1D shape [channel].");

    std::vector<uint32_t> mean_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[3].node_arg, mean_shape), "Cannot get shape of input 3 (mean).");
    ORT_RETURN_IF(mean_shape.size() != 1 || mean_shape[0] != num_channels,
                  "QNN BatchNorm input 3 (mean) must have 1D shape [channel].");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.IsInitializerInput(inputs[3].node_arg.Name()), "QNN BatchNorm doesn't support dynamic mean.");

    std::vector<uint32_t> var_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[4].node_arg, var_shape), "Cannot get shape of input 4 (var).");
    ORT_RETURN_IF(var_shape.size() != 1 || var_shape[0] != num_channels,
                  "QNN BatchNorm input 4 (var) must have 1D shape [channel].");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.IsInitializerInput(inputs[4].node_arg.Name()), "QNN BatchNorm doesn't support dynamic var.");

    ORT_RETURN_IF(node_unit.Outputs().size() > 1, "QNN BatchNorm only support 1 output.");
  }

  return Status::OK();
}

Status BatchNormOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                        const NodeUnit& node_unit,
                        const logging::Logger& logger,
                        std::vector<std::string>& input_names,
                        bool do_op_validation) const {
  bool is_npu_backend = IsNpuBackend(qnn_model_wrapper.GetQnnBackendType());
  if (is_npu_backend) {
    // Inputs & output handling mostly same for most of the Ops, just node attributes are different
    ORT_RETURN_IF_ERROR(ProcessQDQInputs(qnn_model_wrapper, node_unit, logger,
                                      input_names, do_op_validation));
  } else {
    // Inputs & output handling mostly same for most of the Ops, just node attributes are different
    ORT_RETURN_IF_ERROR(ProcessFPInputs(qnn_model_wrapper, node_unit, logger,
                                      input_names, do_op_validation));
  }
  return Status::OK();
}

Status BatchNormOpBuilder::ProcessQDQInputs(QnnModelWrapper& qnn_model_wrapper,
                                            const NodeUnit& node_unit,
                                            const logging::Logger& logger,
                                            std::vector<std::string>& input_names,
                                            bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);

  const auto& inputs = node_unit.Inputs();
  const size_t num_inputs = inputs.size();
  //
  // Input 0
  //
  {
    const std::string& input0_name = inputs[0].node_arg.Name();
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(input0_name)) {
      LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << input0_name;
    } else {
      OnnxInputInfo input0_info = {};
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetOnnxInputInfo(inputs[0], input0_info));
      std::vector<uint8_t> unpacked_tensor;
      if (input0_info.is_initializer) {
        ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input0_info.initializer_tensor, unpacked_tensor));
      }

      Qnn_TensorType_t tensor_type = GetInputTensorType(qnn_model_wrapper, input0_name);
      QnnTensorWrapper input_tensorwrapper(input0_name, tensor_type, input0_info.qnn_data_type, input0_info.quant_param,
                                          std::move(input0_info.shape), std::move(unpacked_tensor));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
    }
    input_names.push_back(input0_name);
  }

  //
  // Input 1: scale
  // Input 2: bias
  // QNN only accept 3 input. We need to first combine mean and variance into scale and bias.
  //
  {
    const std::string& scale_name = inputs[1].node_arg.Name();
    const std::string& bias_name = inputs[2].node_arg.Name();
    const std::string& mean_name = inputs[3].node_arg.Name();
    const std::string& var_name = inputs[4].node_arg.Name();
    OnnxInputInfo var_info = {};
    OnnxInputInfo mean_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetOnnxInputInfo(inputs[3], mean_info));
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetOnnxInputInfo(inputs[4], var_info));
    // scale, bias, mean, and var must be initializers
    ORT_RETURN_IF_NOT(mean_info.is_initializer, "scale, bias, mean, and var must be initializers");
    ORT_RETURN_IF_NOT(var_info.is_initializer, "scale, bias, mean, and var must be initializers");
    std::vector<uint8_t> var_unpacked_tensor;
    std::vector<uint8_t> mean_unpacked_tensor;
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*var_info.initializer_tensor, var_unpacked_tensor));
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*mean_info.initializer_tensor, mean_unpacked_tensor));
    std::vector<double> std_float_tensor(var_unpacked_tensor.size());
    std::vector<double> mean_float_tensor(mean_unpacked_tensor.size());
    auto var_offset = static_cast<double>(var_info.quant_param.scaleOffsetEncoding.offset);
    auto var_scale = static_cast<double>(var_info.quant_param.scaleOffsetEncoding.scale);
    auto mean_offset = static_cast<double>(mean_info.quant_param.scaleOffsetEncoding.offset);
    auto mean_scale = static_cast<double>(mean_info.quant_param.scaleOffsetEncoding.scale);
    for(int i = 0;i<var_unpacked_tensor.size();i++){
      auto var_quant = static_cast<double>(var_unpacked_tensor[i]);
      auto mean_quant = static_cast<double>(mean_unpacked_tensor[i]);
      std_float_tensor[i] = std::sqrt(static_cast<double>((var_quant + var_offset) * var_scale) + 1e-5);
      mean_float_tensor[i] = static_cast<double>((mean_quant + mean_offset) * mean_scale);
    }
    OnnxInputInfo scale_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetOnnxInputInfo(inputs[1], scale_info));
    
    std::vector<uint8_t> scale_unpacked_tensor;
    // scale, bias, mean, and var must be initializers
    ORT_RETURN_IF_NOT(scale_info.is_initializer, "scale, bias, mean, and var must be initializers");
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*scale_info.initializer_tensor, scale_unpacked_tensor));
    // experiment, only implement uint8 now
    std::vector<uint8_t> new_scale_unpacked_tensor(scale_unpacked_tensor.size());
    std::vector<float> new_scale_float_tensor(scale_unpacked_tensor.size());
    auto scale_offset = static_cast<double>(scale_info.quant_param.scaleOffsetEncoding.offset);
    auto scale_scale = static_cast<double>(scale_info.quant_param.scaleOffsetEncoding.scale);
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(scale_name)) {
      LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << scale_name;
    } else {
      float rmax = std::numeric_limits<float>::min();
      float rmin = std::numeric_limits<float>::max();
      for (int i = 0;i < scale_unpacked_tensor.size();++i){
        auto scale_quant = static_cast<double>(scale_unpacked_tensor[i]);
        new_scale_float_tensor[i] = static_cast<double>((scale_quant + scale_offset) * scale_scale) / std_float_tensor[i];
        rmax = std::max(rmax, new_scale_float_tensor[i]);
        rmin = std::min(rmin, new_scale_float_tensor[i]);
      }
      auto [new_scale_scale, new_scale_zero_point] = GetQuantParams(rmin, rmax);
      new_scale_zero_point = 0 - new_scale_zero_point;
      Qnn_QuantizeParams_t new_scale_quant_param = QNN_QUANTIZE_PARAMS_INIT;
      utils::InitializeQuantizeParam(new_scale_quant_param, true, new_scale_scale, new_scale_zero_point);
      for (int i = 0;i < new_scale_float_tensor.size();++i){
        int out_quant = std::round((new_scale_float_tensor[i] / new_scale_scale) - new_scale_zero_point);
        new_scale_unpacked_tensor[i] = (out_quant < 0)? 0 :
                                                        (out_quant > 255)? 255: out_quant;
      }
      Qnn_TensorType_t scale_tensor_type = GetInputTensorType(qnn_model_wrapper, scale_name);
      QnnTensorWrapper input_tensorwrapper(scale_name, scale_tensor_type, scale_info.qnn_data_type, new_scale_quant_param,
                                          std::move(scale_info.shape), std::move(new_scale_unpacked_tensor));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
    }
    input_names.push_back(scale_name);
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(bias_name)) {
      LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << bias_name;
    } else {
      OnnxInputInfo bias_info = {};
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetOnnxInputInfo(inputs[2], bias_info));
      
      std::vector<uint8_t> bias_unpacked_tensor;
      // scale, bias, mean, and var must be initializers
      ORT_RETURN_IF_NOT(bias_info.is_initializer, "scale, bias, mean, and var must be initializers");
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*bias_info.initializer_tensor, bias_unpacked_tensor));
      // experiment, only implement uint8 now
      std::vector<uint8_t> new_bias_unpacked_tensor(bias_unpacked_tensor.size());
      std::vector<float> new_bias_float_tensor(bias_unpacked_tensor.size());
      float rmax = std::numeric_limits<float>::min();
      float rmin = std::numeric_limits<float>::max();
      auto bias_offset = static_cast<double>(bias_info.quant_param.scaleOffsetEncoding.offset);
      auto bias_scale = static_cast<double>(bias_info.quant_param.scaleOffsetEncoding.scale);
      for (int i = 0;i < bias_unpacked_tensor.size();i++){
        auto bias_quant = static_cast<double>(bias_unpacked_tensor[i]);
        auto scale_quant = static_cast<double>(scale_unpacked_tensor[i]);
        new_bias_float_tensor[i] = static_cast<double>((bias_quant + bias_offset) * bias_scale) - (mean_float_tensor[i] * static_cast<double>((scale_quant + scale_offset) * scale_scale) / std_float_tensor[i]);
        rmax = std::max(rmax, new_bias_float_tensor[i]);
        rmin = std::min(rmin, new_bias_float_tensor[i]);
      }
      auto [new_bias_scale, new_bias_zero_point] = GetQuantParams(rmin, rmax);
      new_bias_zero_point = 0 - new_bias_zero_point;
      Qnn_QuantizeParams_t new_bias_quant_param = QNN_QUANTIZE_PARAMS_INIT;
      utils::InitializeQuantizeParam(new_bias_quant_param, true, new_bias_scale, new_bias_zero_point);
      for (int i = 0;i < new_bias_float_tensor.size();++i){
        int out_quant = std::round((new_bias_float_tensor[i] / new_bias_scale) - new_bias_zero_point);
        new_bias_unpacked_tensor[i] = (out_quant < 0)? 0 :
                                                        (out_quant > 255)? 255: out_quant;
      }
      Qnn_TensorType_t bias_tensor_type = GetInputTensorType(qnn_model_wrapper, bias_name);
      QnnTensorWrapper input_tensorwrapper(bias_name, bias_tensor_type, bias_info.qnn_data_type, new_bias_quant_param,
                                          std::move(bias_info.shape), std::move(new_bias_unpacked_tensor));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
    }
    input_names.push_back(bias_name);
  }
  
  return Status::OK();
}

Status BatchNormOpBuilder::ProcessFPInputs(QnnModelWrapper& qnn_model_wrapper,
                                          const NodeUnit& node_unit,
                                          const logging::Logger& logger,
                                          std::vector<std::string>& input_names,
                                          bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);

  const auto& inputs = node_unit.Inputs();
  const size_t num_inputs = inputs.size();
  //
  // Input 0
  //
  {
    const std::string& input0_name = inputs[0].node_arg.Name();
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(input0_name)) {
      LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << input0_name;
    } else {
      OnnxInputInfo input0_info = {};
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetOnnxInputInfo(inputs[0], input0_info));
      std::vector<uint8_t> unpacked_tensor;
      if (input0_info.is_initializer) {
        ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input0_info.initializer_tensor, unpacked_tensor));
      }

      Qnn_TensorType_t tensor_type = GetInputTensorType(qnn_model_wrapper, input0_name);
      QnnTensorWrapper input_tensorwrapper(input0_name, tensor_type, input0_info.qnn_data_type, input0_info.quant_param,
                                          std::move(input0_info.shape), std::move(unpacked_tensor));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
    }
    input_names.push_back(input0_name);
  }

  //
  // Input 1: scale
  // Input 2: bias
  // QNN only accept 3 input. We need to first combine mean and variance into scale and bias.
  //
  {
    const std::string& scale_name = inputs[1].node_arg.Name();
    const std::string& bias_name = inputs[2].node_arg.Name();
    const std::string& mean_name = inputs[3].node_arg.Name();
    const std::string& var_name = inputs[4].node_arg.Name();
    OnnxInputInfo var_info = {};
    OnnxInputInfo mean_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetOnnxInputInfo(inputs[3], mean_info));
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetOnnxInputInfo(inputs[4], var_info));
    // scale, bias, mean, and var must be initializers
    ORT_RETURN_IF_NOT(mean_info.is_initializer, "scale, bias, mean, and var must be initializers");
    ORT_RETURN_IF_NOT(var_info.is_initializer, "scale, bias, mean, and var must be initializers");
    std::vector<uint8_t> var_unpacked_tensor;
    std::vector<uint8_t> mean_unpacked_tensor;
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*var_info.initializer_tensor, var_unpacked_tensor));
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*mean_info.initializer_tensor, mean_unpacked_tensor));
    const float* mean_float_tensor = reinterpret_cast<const float*>(mean_unpacked_tensor.data());
    size_t mean_length = mean_unpacked_tensor.size()/4;
    const float* var_float_tensor = reinterpret_cast<const float*>(var_unpacked_tensor.data());
    size_t var_length = var_unpacked_tensor.size()/4;
    std::vector<float> std_float_tensor(var_unpacked_tensor.size()/4);
    for(int i = 0;i < var_length;i++){
      std_float_tensor[i] = std::sqrt(var_float_tensor[i] + 1e-5);
    }
    OnnxInputInfo scale_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetOnnxInputInfo(inputs[1], scale_info));
    
    std::vector<uint8_t> scale_unpacked_tensor;
    // scale, bias, mean, and var must be initializers
    ORT_RETURN_IF_NOT(scale_info.is_initializer, "scale, bias, mean, and var must be initializers");
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*scale_info.initializer_tensor, scale_unpacked_tensor));
    // experiment, only implement uint8 now
    const float* scale_float_tensor = reinterpret_cast<const float*>(scale_unpacked_tensor.data());
    size_t scale_lenght = scale_unpacked_tensor.size()/4;
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(scale_name)) {
      LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << scale_name;
    } else {
      std::vector<float> new_scale_float_tensor(scale_lenght);
      for (int i = 0;i < scale_lenght;++i){
        new_scale_float_tensor[i] = scale_float_tensor[i] / std_float_tensor[i];
      }
      uint8_t* new_scale_ptr = reinterpret_cast<uint8_t*>(new_scale_float_tensor.data());
      std::vector<uint8_t> new_scale_unpacked_tensor(new_scale_ptr, new_scale_ptr + (scale_lenght*4));
      Qnn_TensorType_t scale_tensor_type = GetInputTensorType(qnn_model_wrapper, scale_name);
      QnnTensorWrapper input_tensorwrapper(scale_name, scale_tensor_type, scale_info.qnn_data_type, scale_info.quant_param,
                                          std::move(scale_info.shape), std::move(new_scale_unpacked_tensor));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
    }
    input_names.push_back(scale_name);
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(bias_name)) {
      LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << bias_name;
    } else {
      OnnxInputInfo bias_info = {};
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetOnnxInputInfo(inputs[2], bias_info));
      
      std::vector<uint8_t> bias_unpacked_tensor;
      // scale, bias, mean, and var must be initializers
      ORT_RETURN_IF_NOT(bias_info.is_initializer, "scale, bias, mean, and var must be initializers");
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*bias_info.initializer_tensor, bias_unpacked_tensor));
      // experiment, only implement uint8 now
      const float* bias_float_tensor = reinterpret_cast<const float*>(bias_unpacked_tensor.data());
      size_t bias_lenght = bias_unpacked_tensor.size()/4;
      std::vector<float> new_bias_float_tensor(bias_lenght);
      for (int i = 0;i < bias_lenght;i++){
        new_bias_float_tensor[i] = bias_float_tensor[i] - (mean_float_tensor[i] * scale_float_tensor[i] / std_float_tensor[i]);
      }
      uint8_t* new_bias_ptr = reinterpret_cast<uint8_t*>(new_bias_float_tensor.data());
      std::vector<uint8_t> new_bias_unpacked_tensor(new_bias_ptr, new_bias_ptr + (bias_lenght * 4));
      Qnn_TensorType_t bias_tensor_type = GetInputTensorType(qnn_model_wrapper, bias_name);
      QnnTensorWrapper input_tensorwrapper(bias_name, bias_tensor_type, bias_info.qnn_data_type, bias_info.quant_param,
                                          std::move(bias_info.shape), std::move(new_bias_unpacked_tensor));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
    }
    input_names.push_back(bias_name);
  }
  
  return Status::OK();
}

void CreateBatchNormOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<BatchNormOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
