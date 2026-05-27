// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/op_builder_factory.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace onnxruntime {
namespace qnn {

OpBuilderRegistrations::OpBuilderRegistrations() {
  CreateArgMaxMinOpBuilder("ArgMax", *this);
  CreateArgMaxMinOpBuilder("ArgMin", *this);
  CreateBatchNormalizationOpBuilder("BatchNormalization", *this);
  CreateCastOpBuilder("Cast", *this);
  CreateClipOpBuilder("Clip", *this);
  CreateConcatOpBuilder("Concat", *this);
  CreateConvOpBuilder("Conv", *this);
  CreateConvOpBuilder("ConvTranspose", *this);
  CreateCumSumOpBuilder("CumSum", *this);
  CreateEinsumOpBuilder("Einsum", *this);
  CreateExpandOpBuilder("Expand", *this);
  CreateFusedMatMulOpBuilder("FusedMatMul", *this);
  CreateGatherOpBuilder("Gather", *this);
  CreateGatherOpBuilder("GatherElements", *this);
  CreateGatherNDOpBuilder("GatherND", *this);
  CreateGemmOpBuilder("Gemm", *this);
  CreateGroupNormalizationOpBuilder("GroupNormalization", *this);
  CreateGRUOpBuilder("GRU", *this);
  CreateIdentityOpBuilder("Identity", *this);
  CreateInstanceNormalizationOpBuilder("InstanceNormalization", *this);
  CreateInverseOpBuilder("Inverse", *this);
  CreateIsNaNOpBuilder("IsNaN", *this);
  CreateLayerNormalizationOpBuilder("LayerNormalization", *this);
  CreateLRNOpBuilder("LRN", *this);
  CreateLSTMOpBuilder("LSTM", *this);
  CreateMatMulOpBuilder("MatMul", *this);
  CreateMatMulNBitsOpBuilder("MatMulNBits", *this);
  CreateMeanOpBuilder("Mean", *this);
  CreateModOpBuilder("Mod", *this);
  CreateNonZeroOpBuilder("NonZero", *this);
  CreatePadOpBuilder("Pad", *this);
  CreatePoolOpBuilder("AveragePool", *this);
  CreatePoolOpBuilder("GlobalAveragePool", *this);
  CreatePoolOpBuilder("GlobalMaxPool", *this);
  CreatePoolOpBuilder("MaxPool", *this);
  CreateQuickGeluOpBuilder("QuickGelu", *this);
  CreateRandomNormalLikeOpBuilder("RandomNormalLike", *this);
  CreateRandomUniformLikeOpBuilder("RandomUniformLike", *this);
  CreateReciprocalOpBuilder("Reciprocal", *this);
  CreateReduceOpBuilder("ReduceL2", *this);
  CreateReduceOpBuilder("ReduceMax", *this);
  CreateReduceOpBuilder("ReduceMean", *this);
  CreateReduceOpBuilder("ReduceMin", *this);
  CreateReduceOpBuilder("ReduceProd", *this);
  CreateReduceOpBuilder("ReduceSum", *this);
  CreateReshapeOpBuilder("Flatten", *this);
  CreateReshapeOpBuilder("Reshape", *this);
  CreateReshapeOpBuilder("Squeeze", *this);
  CreateReshapeOpBuilder("Unsqueeze", *this);
  CreateResizeOpBuilder("Resize", *this);
  CreateRMSNormalizationOpBuilder("RMSNormalization", *this);
  CreateRMSNormalizationOpBuilder("SimplifiedLayerNormalization", *this);
  CreateRoiAlignOpBuilder("RoiAlign", *this);
  CreateRotaryEmbeddingOpBuilder("RotaryEmbedding", *this);
  CreateSimpleOpBuilder("Abs", *this);
  CreateSimpleOpBuilder("Add", *this);
  CreateSimpleOpBuilder("And", *this);
  CreateSimpleOpBuilder("Asin", *this);
  CreateSimpleOpBuilder("Atan", *this);
  CreateSimpleOpBuilder("Ceil", *this);
  CreateSimpleOpBuilder("Cos", *this);
  CreateSimpleOpBuilder("DequantizeLinear", *this);
  CreateSimpleOpBuilder("DepthToSpace", *this);
  CreateSimpleOpBuilder("Div", *this);
  CreateSimpleOpBuilder("Elu", *this);
  CreateSimpleOpBuilder("Equal", *this);
  CreateSimpleOpBuilder("Exp", *this);
  CreateSimpleOpBuilder("Floor", *this);
  CreateSimpleOpBuilder("Gelu", *this);
  CreateSimpleOpBuilder("Greater", *this);
  CreateSimpleOpBuilder("GreaterOrEqual", *this);
  CreateSimpleOpBuilder("GridSample", *this);
  CreateSimpleOpBuilder("HardSigmoid", *this);
  CreateSimpleOpBuilder("HardSwish", *this);
  CreateSimpleOpBuilder("LeakyRelu", *this);
  CreateSimpleOpBuilder("Less", *this);
  CreateSimpleOpBuilder("LessOrEqual", *this);
  CreateSimpleOpBuilder("Log", *this);
  CreateSimpleOpBuilder("LpNormalization", *this);
  CreateSimpleOpBuilder("Max", *this);
  CreateSimpleOpBuilder("Min", *this);
  CreateSimpleOpBuilder("Mul", *this);
  CreateSimpleOpBuilder("Neg", *this);
  CreateSimpleOpBuilder("Not", *this);
  CreateSimpleOpBuilder("Or", *this);
  CreateSimpleOpBuilder("Pow", *this);
  CreateSimpleOpBuilder("PRelu", *this);
  CreateSimpleOpBuilder("QuantizeLinear", *this);
  CreateSimpleOpBuilder("Relu", *this);
  CreateSimpleOpBuilder("Round", *this);
  CreateScatterElementsOpBuilder("ScatterElements", *this);
  CreateScatterNDOpBuilder("ScatterND", *this);
  CreateSeluOpBuilder("Selu", *this);
  CreateSimpleOpBuilder("Sigmoid", *this);
  CreateSimpleOpBuilder("Sign", *this);
  CreateSimpleOpBuilder("Sin", *this);
  CreateSimpleOpBuilder("Softplus", *this);
  CreateSimpleOpBuilder("SpaceToDepth", *this);
  CreateSimpleOpBuilder("Sqrt", *this);
  CreateSimpleOpBuilder("Sub", *this);
  CreateSimpleOpBuilder("Sum", *this);
  CreateSimpleOpBuilder("Tanh", *this);
  CreateSimpleOpBuilder("Where", *this);
  CreateSimpleOpBuilder("Xor", *this);
  CreateSliceOpBuilder("Slice", *this);
  CreateSoftmaxOpBuilder("LogSoftmax", *this);
  CreateSoftmaxOpBuilder("Softmax", *this);
  CreateSplitOpBuilder("Split", *this);
  CreateSTFTOpBuilder("STFT", *this);
  CreateTanOpBuilder("Tan", *this);
  CreateThresholdedReluOpBuilder("ThresholdedRelu", *this);
  CreateTileOpBuilder("Tile", *this);
  CreateTopKOpBuilder("TopK", *this);
  CreateTransposeOpBuilder("Transpose", *this);
  CreateUpsampleOpBuilder("Upsample", *this);
}

const IOpBuilder* GetOpBuilder(const std::string& onnx_op_type) {
  static const OpBuilderRegistrations op_registrations;
  return op_registrations.GetOpBuilderByOnnxOpType(onnx_op_type);
}

}  // namespace qnn
}  // namespace onnxruntime
