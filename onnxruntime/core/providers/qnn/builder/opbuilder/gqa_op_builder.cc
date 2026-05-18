// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// =============================================================================
// GroupQueryAttention (GQA) op-builder skeleton.
//
// This is a *minimal stub*. It only registers the op type with the EP and
// returns success from IsOpSupported / ProcessInputs / ProcessAttributesAndOutputs
// without actually emitting any QNN nodes. Real implementation is gated on:
//
//   - QNN OpDef proposal (AISW-180612, fixVersion QAIRT-2.48.0).
//     Local diff: AISW-183377-gqa/8d69f96.diff, Gerrit qctaisw/qnn-api/+/292838.
//   - HTP backend implementation: AISW-180475 (fp16), AISW-180474 (u8).
//
// Tracking ticket for ORT EP enablement: AISW-183377.
//
// Reference specs:
//   ORT contrib op  : com.microsoft.GroupQueryAttention
//                     https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftgroupqueryattention
//   QNN OpDef       : "GroupQueryAttention" (per AISW-183377-gqa/8d69f96.diff)
//
// HTP backend constraints noted in the OpDef proposal:
//   - Computation always in float (uint8/uint16 dequantized before exec).
//   - Native KV cache NOT supported; backend always processes full max context.
//   - Packed QKV NOT supported; ORT must unpack before dispatch.
//
// ORT contrib op input order (for reference when ProcessInputs is implemented):
//   0  query                  (mandatory)        BSH or packed-QKV
//   1  key                    (optional, BSH)
//   2  value                  (optional, BSH)
//   3  past_key               (optional, BNSH)
//   4  past_value             (optional, BNSH)
//   5  seqlens_k              (mandatory, INT32 [B])
//   6  total_sequence_length  (mandatory, INT32 scalar)
//   7  cos_cache              (optional, when do_rotary=1)
//   8  sin_cache              (optional, when do_rotary=1)
//   9  position_ids           (optional, INT64 [B,S])
// Outputs:
//   0  output                 (mandatory, BSH)
//   1  present_key            (optional, BNSH)
//   2  present_value          (optional, BNSH)
//
// Attributes (ORT contrib):
//   num_heads (int64, mandatory), kv_num_heads (int64, mandatory),
//   scale (float, default 0.0 -> 1/sqrt(head_size)),
//   do_rotary (int64, default 0),
//   plus several MS-specific extensions (rotary_interleaved, softcap,
//   smooth_softmax, local_window_size, qk_output) -- not part of QNN OpDef.
// =============================================================================

#include <string>
#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

namespace {

// QNN op-type and parameter names per the OpDef proposal in
// AISW-183377-gqa/8d69f96.diff. Replace these with the official
// QNN_OP_GROUP_QUERY_ATTENTION* macros once the SDK ships them
// (target: QAIRT-2.48.0).
constexpr const char* kQnnOpGroupQueryAttention = "GroupQueryAttention";
constexpr const char* kQnnParamNumHeads = "num_heads";
constexpr const char* kQnnParamKvNumHeads = "kv_num_heads";
constexpr const char* kQnnParamDoRotary = "do_rotary";
constexpr const char* kQnnParamScale = "scale";

}  // namespace

class GroupQueryAttentionOpBuilder : public BaseOpBuilder {
 public:
  GroupQueryAttentionOpBuilder() : BaseOpBuilder("GroupQueryAttentionOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GroupQueryAttentionOpBuilder);

  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override ORT_MUST_USE_RESULT;

 protected:
  Ort::Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger,
                            std::vector<std::string>& input_names,
                            bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Ort::Status GroupQueryAttentionOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                                        const OrtNodeUnit& node_unit,
                                                        const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(qnn_model_wrapper);
  ORT_UNUSED_PARAMETER(node_unit);
  ORT_UNUSED_PARAMETER(logger);


  // TODO(AISW-183377): real support gating.
  //   1. Reject CPU backend -- not on the QNN GQA roadmap.
  //   2. For HTP: detect packed-QKV (key & value missing while query has hidden
  //      = (num_heads + 2*kv_num_heads) * head_size) and reject; future work
  //      should insert a Split. See OpDef HTP note.
  //   3. Validate dtypes via CheckHtpDataTypes / CheckGpuDataTypes helpers.



  return Ort::Status();
}

Ort::Status GroupQueryAttentionOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                                        const OrtNodeUnit& node_unit,
                                                        const Ort::Logger& logger,
                                                        std::vector<std::string>& input_names,
                                                        bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(qnn_model_wrapper);
  ORT_UNUSED_PARAMETER(node_unit);
  ORT_UNUSED_PARAMETER(logger);
  ORT_UNUSED_PARAMETER(input_names);
  ORT_UNUSED_PARAMETER(do_op_validation);

  // TODO(AISW-183377): walk node_unit.Inputs() in ORT contrib order and call
  // ProcessInput() for each present input. Optional inputs use the pattern
  //   if (inputs.size() > i && inputs[i].Exists()) { ... }
  // (cf. lstm_op_builder.cc:255, matmulnbits_op_builder.cc:227-261).
  // Append names to input_names in the order the QNN OpDef expects (confirm
  // exact order from QnnOpDef.h once QAIRT-2.48.0 ships).
  return Ort::Status();
}

Ort::Status GroupQueryAttentionOpBuilder::ProcessAttributesAndOutputs(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& node_unit,
    std::vector<std::string>&& input_names,
    const Ort::Logger& logger,
    bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(qnn_model_wrapper);
  ORT_UNUSED_PARAMETER(node_unit);
  ORT_UNUSED_PARAMETER(input_names);
  ORT_UNUSED_PARAMETER(logger);
  ORT_UNUSED_PARAMETER(do_op_validation);

  // TODO(AISW-183377):
  //   - Read attributes: num_heads (int64), kv_num_heads (int64),
  //     do_rotary (int64, default 0), scale (float, default 0.0 -> backend
  //     computes 1/sqrt(head_size)).
  //   - Build QnnParamWrapper for each scalar param. Pattern:
  //       Qnn_Scalar_t s = QNN_SCALAR_INIT;
  //       s.uint32Value = static_cast<uint32_t>(num_heads);
  //       QnnParamWrapper p(node_unit.Index(), node_unit.Name(),
  //                         kQnnParamNumHeads, s);
  //     (cf. softmax_op_builder.cc:180.)
  //     Use UINT32 for heads/do_rotary, FLOAT32 for scale.
  //   - Process outputs: output (mandatory), present_key, present_value
  //     (both optional).
  //   - Call qnn_model_wrapper.CreateQnnNode() with op type
  //     kQnnOpGroupQueryAttention, package QNN_OP_PACKAGE_NAME_QTI_AISW.
  return Ort::Status();
}

void CreateGroupQueryAttentionOpBuilder(const std::string& op_type,
                                        OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<GroupQueryAttentionOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
