// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder.h"
#include "core/providers/qnn/builder/qnn_quant_params_wrapper.h"

#include "QnnOpDef.h"

namespace onnxruntime {
namespace qnn {
class OpBuilderHelper {
 public:
  OpBuilderHelper(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit);

  Status AddSingleNode(
      const std::string& qnn_node_type,
      std::vector<std::string>&& input_names,
      std::vector<std::string>&& param_names,
      std::vector<uint32_t>&& output_shape,
      std::vector<std::string>&& output_name,
      bool do_op_validation);

  Status AddSequentialNode(
      const std::vector<std::string>& qnn_node_type,
      std::vector<std::string>&& input_names,
      std::vector<std::string>&& param_names,
      std::vector<uint32_t>&& output_shape,
      std::vector<std::string>&& output_name,
      bool do_op_validation);

  Status AddSequentialNode(
      const std::vector<std::string>& qnn_node_type,
      std::vector<std::string>&& input_names,
      std::vector<std::string>&& param_names,
      std::vector<std::vector<uint32_t>>&& output_shape_and_perm,
      std::vector<std::string>&& output_name,
      bool do_op_validation);

  Status AddQnnConstant(const std::vector<uint32_t>& constant_value, Qnn_DataType_t qnn_data_type);

 protected:
  std::unordered_map<std::string, std::vector<uint32_t>> tensor_to_shape_dict;

 private:
  Status RunQnnNodePreLowerValidation(const std::string& qnn_node_type,
                                      std::vector<std::string>& input_names,
                                      std::vector<std::string>& param_names,
                                      std::vector<uint32_t>& output_shape,
                                      std::vector<std::string>& output_name);

  QnnModelWrapper& qnn_model_wrapper_;
  const NodeUnit& node_unit_;

  TensorInfo input_info_;
  TensorInfo output_info_;
  const std::string& org_output_name;
  const bool is_graph_output;
  const Qnn_TensorType_t op_output_tensor_type;
};

}  // namespace qnn
}  // namespace onnxruntime
