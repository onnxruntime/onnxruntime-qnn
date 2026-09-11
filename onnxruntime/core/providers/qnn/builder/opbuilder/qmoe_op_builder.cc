#include <string>
#include <vector>
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_quant_params_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

namespace {
constexpr char kQMoEOp[] = "QMoE";
constexpr char kK[] = "k";
constexpr char kActivationType[] = "activation_type";
constexpr char kSwigluFusion[] = "swiglu_fusion";
constexpr char kSwigluLimit[] = "swiglu_limit";
constexpr char kNormalizeRoutingWeights[] = "normalize_routing_weights";
constexpr char kUseSparseMixer[] = "use_sparse_mixer";
constexpr char kActivationAlpha[] = "activation_alpha";
constexpr char kActivationBeta[] = "activation_beta";

Ort::Status AddQMoEWeight(QnnModelWrapper& qnn_model_wrapper,
                          const OrtNodeUnitIODef& weight,
                          const OrtNodeUnitIODef& scale,
                          uint32_t block_size,
                          uint32_t bits,
                          const Ort::Logger& logger) {
  if (qnn_model_wrapper.IsQnnTensorWrapperExist(weight.name)) {
    return Ort::Status();
  }
  const OrtValueInfo* weight_proto = qnn_model_wrapper.GetConstantTensor(weight.name);
  const OrtValueInfo* scale_proto = qnn_model_wrapper.GetConstantTensor(scale.name);
  RETURN_IF_NOT(weight_proto != nullptr && scale_proto != nullptr, "QMoE weights and scales must be initializers.");

  std::vector<uint8_t> weight_data;
  RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(weight_proto, weight_data, false));
  std::vector<uint8_t> scale_data;
  RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(scale_proto, scale_data));

  const size_t scale_element_size = utils::GetElementSizeByType(scale.type);
  RETURN_IF_NOT(scale_element_size != 0 && scale_data.size() % scale_element_size == 0,
                "Invalid QMoE scale initializer size.");
  const size_t scale_count = scale_data.size() / scale_element_size;
  std::vector<float> scales;
  scales.reserve(scale_count);
  if (scale.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    const float* p = reinterpret_cast<const float*>(scale_data.data());
    scales.assign(p, p + scale_count);
  } else {
    const Ort::Float16_t* p = reinterpret_cast<const Ort::Float16_t*>(scale_data.data());
    for (size_t i = 0; i < scale_count; ++i) scales.push_back(static_cast<float>(p[i]));
  }
  std::vector<int32_t> offsets(scale_count, 0);
  const std::vector<uint32_t> block_sizes = {1, 1, block_size / (8 / bits)};
  QnnQuantParamsWrapper quant_params = QnnQuantParamsWrapper::BwBlockMapped(scales, offsets, bits, block_sizes);

  std::vector<uint32_t> shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(weight.shape, shape), "Invalid QMoE weight shape.");
  QnnTensorWrapper wrapper(weight.name,
                           qnn_model_wrapper.GetTensorType(weight.name),
                           QNN_DATATYPE_UINT_8,
                           std::move(quant_params),
                           std::move(shape),
                           std::move(weight_data));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(wrapper)), "Failed to add QMoE weight tensor.");
  ORT_UNUSED_PARAMETER(logger);
  return Ort::Status();
}

}  // namespace

class QMoEOpBuilder final : public BaseOpBuilder {
 public:
  QMoEOpBuilder() : BaseOpBuilder("QMoEOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QMoEOpBuilder);

  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override ORT_MUST_USE_RESULT {
    RETURN_IF_NOT(IsGpuBackend(qnn_model_wrapper.GetQnnBackendType()), "QMoE is supported only on QNN GPU.");
    const auto& inputs = node_unit.Inputs();
    RETURN_IF_NOT(inputs.size() >= 8, "QMoE requires the input, router, fc1/fc2 weights, scales, and biases.");
    OrtNodeAttrHelper h(node_unit);
    RETURN_IF_NOT(h.Get("expert_weight_bits", int64_t(0)) == 4, "QMoE GPU requires expert_weight_bits=4.");
    RETURN_IF_NOT(h.Get("block_size", int64_t(0)) == 32, "QMoE GPU requires block_size=32.");
    RETURN_IF_NOT(h.Get("k", int64_t(0)) > 0, "QMoE requires k > 0.");
    RETURN_IF_NOT(h.Get("activation_type", std::string()) == "swiglu", "QMoE GPU requires swiglu activation.");
    RETURN_IF_NOT(h.Get("swiglu_fusion", int64_t(0)) == 1, "QMoE GPU requires swiglu_fusion=1.");
    RETURN_IF_NOT(h.Get("use_sparse_mixer", int64_t(0)) == 0, "QMoE GPU does not support sparse mixer.");
    RETURN_IF_NOT(inputs.size() < 14 || !inputs[8].Exists(), "Fused SwiGLU QMoE must not provide fc3 inputs.");
    return BaseOpBuilder::IsOpSupported(qnn_model_wrapper, node_unit, logger);
  }

 protected:
  Ort::Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger,
                            std::vector<std::string>& input_names,
                            bool do_op_validation) const override ORT_MUST_USE_RESULT {
    ORT_UNUSED_PARAMETER(do_op_validation);
    const auto& in = node_unit.Inputs();
    OrtNodeAttrHelper h(node_unit);
    const uint32_t block_size = static_cast<uint32_t>(h.Get("block_size", int64_t(32)));
    const uint32_t bits = static_cast<uint32_t>(h.Get("expert_weight_bits", int64_t(4)));
    const size_t map[14] = {0, 1, 2, 5, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13};
    const size_t scale_for_weight[14] = {0, 0, 3, 6, 0, 0, 6, 0, 9, 0, 0, 0, 0, 0};
    for (size_t slot = 0; slot < 14; ++slot) {
      const size_t idx = map[slot];
      if (idx >= in.size() || !in[idx].Exists()) {
        input_names.emplace_back("");
        continue;
      }
      if (slot == 2 || slot == 3 || slot == 8) {
        RETURN_IF_ERROR(AddQMoEWeight(qnn_model_wrapper, in[idx], in[scale_for_weight[slot]], block_size, bits, logger));
        input_names.push_back(in[idx].name);
      } else {
        RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, in[idx], logger, input_names));
      }
    }
    while (!input_names.empty() && input_names.back().empty()) {
      input_names.pop_back();
    }
    return Ort::Status();
  }

  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT {
    OrtNodeAttrHelper h(node_unit);
    std::vector<std::string> params;
    const std::string activation = h.Get("activation_type", std::string("relu"));
    uint32_t activation_type = activation == "relu" ? 0u : activation == "gelu" ? 1u
                                                       : activation == "silu"   ? 2u
                                                       : activation == "swiglu" ? 3u
                                                                                : 4u;
    RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), static_cast<uint32_t>(h.Get("k", int64_t(0))), kK, params));
    RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), activation_type, kActivationType, params));
    RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), static_cast<uint32_t>(h.Get("swiglu_fusion", int64_t(0))), kSwigluFusion, params));
    RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), h.Get("swiglu_limit", 0.0f), kSwigluLimit, params));
    RETURN_IF_ERROR(AddQnnScalar<bool>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), h.Get("normalize_routing_weights", false), kNormalizeRoutingWeights, params));
    RETURN_IF_ERROR(AddQnnScalar<bool>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), h.Get("use_sparse_mixer", false), kUseSparseMixer, params));
    RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), h.Get("activation_alpha", 1.0f), kActivationAlpha, params));
    RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), h.Get("activation_beta", 0.0f), kActivationBeta, params));

    QnnTensorWrapper output;
    RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(node_unit.Outputs()[0], output));
    const std::string output_name = node_unit.Outputs()[0].name;
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output)), "Failed to add QMoE output tensor.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_unit.Name(), QNN_OP_PACKAGE_NAME_QTI_AISW, kQMoEOp,
                                                  std::move(input_names), {output_name}, std::move(params), do_op_validation),
                  "Failed to create QMoE QNN node.");
    ORT_UNUSED_PARAMETER(logger);
    return Ort::Status();
  }
};

void CreateQMoEOpBuilder(const std::string& op_type, OpBuilderRegistrations& registrations) {
  registrations.AddOpBuilder(op_type, std::make_unique<QMoEOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
