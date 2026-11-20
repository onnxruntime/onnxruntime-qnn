#include <string>
#include <vector>
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"

namespace onnxruntime {
namespace qnn {

namespace {
constexpr char kQMoEOp[] = "QMoE";
constexpr char kK[] = "k";
constexpr char kExpertWeightBits[] = "expert_weight_bits";
constexpr char kBlockSize[] = "block_size";
constexpr char kActivationType[] = "activation_type";
constexpr char kSwigluFusion[] = "swiglu_fusion";
constexpr char kSwigluLimit[] = "swiglu_limit";
constexpr char kNormalizeRoutingWeights[] = "normalize_routing_weights";
constexpr char kUseSparseMixer[] = "use_sparse_mixer";
constexpr char kActivationAlpha[] = "activation_alpha";
constexpr char kActivationBeta[] = "activation_beta";

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
    // QMoE has fixed positional inputs. Preserve null optional slots through router_weights
    // (slot 14), as QNN validation does not accept a truncated input array.
    const size_t map[15] = {0, 1, 2, 5, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    for (size_t slot = 0; slot < 15; ++slot) {
      const size_t idx = map[slot];
      if (idx >= in.size() || !in[idx].Exists()) {
        input_names.emplace_back("");
        continue;
      }
      RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, in[idx], logger, input_names));
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
    RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), static_cast<uint32_t>(h.Get("expert_weight_bits", int64_t(4))), kExpertWeightBits, params));
    RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), static_cast<uint32_t>(h.Get("block_size", int64_t(0))), kBlockSize, params));
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
