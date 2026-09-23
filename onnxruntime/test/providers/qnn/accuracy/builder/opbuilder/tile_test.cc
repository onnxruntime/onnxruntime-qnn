// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Op-builder-paired accuracy tests for TileOpBuilder.

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS && QNN_EP_ACCURACY_UT

#include <cstdint>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "test/providers/qnn/infra/specs/builder/opbuilder/tile_specs.h"
#include "test/providers/qnn/qnn_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

ProviderOptions MakeAccuracyProviderOptions([[maybe_unused]] TileBackend backend) {
  ProviderOptions po;
  po["backend_type"] = "htp";
  po["offload_graph_io_quantization"] = "0";
  return po;
}

GetTestModelFn BuildTileOnnxF32(const TileQDQSpec& spec) {
  auto input_def = TestInputDef<float>(spec.input_shape, false, spec.input_data);
  auto repeats_def = TestInputDef<int64_t>(
      {static_cast<int64_t>(spec.repeats.size())}, true, spec.repeats);
  return BuildOpTestCase<float, int64_t>("Tile_node", "Tile", {input_def}, {repeats_def}, {});
}

template <typename QuantType>
GetTestQDQModelFn<QuantType> BuildQDQTileOnnx(const TileQDQSpec& spec) {
  return [spec](ModelTestBuilder& builder, std::vector<QuantParams<QuantType>>& output_qparams) {
    auto input_def = TestInputDef<float>(spec.input_shape, false, spec.input_data);
    MakeTestInput(builder, "input", input_def);

    const QuantParams<QuantType> input_qparams{
        spec.scale, static_cast<QuantType>(spec.zero_point)};
    const std::string input_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq_in", "input",
                                  input_qparams.scale, input_qparams.zero_point,
                                  spec.use_contrib_qdq);

    auto repeats_def = TestInputDef<int64_t>(
        {static_cast<int64_t>(spec.repeats.size())}, true, spec.repeats);
    MakeTestInput(builder, "repeats", repeats_def);

    builder.AddNode("Tile", "Tile", {input_qdq, "repeats"}, {"tile_out"});

    output_qparams[0] = input_qparams;
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(
        builder, "qdq_out", "tile_out", input_qparams.scale, input_qparams.zero_point,
        spec.use_contrib_qdq);
  };
}

void RunTileQDQAccuracy(const TileQDQSpec& spec) {
  ProviderOptions po = MakeAccuracyProviderOptions(spec.accuracy_backend);
  switch (spec.qdq_dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      TestQDQModelAccuracy(BuildTileOnnxF32(spec),
                           BuildQDQTileOnnx<uint8_t>(spec),
                           po, spec.opset, ExpectedEPNodeAssignment::All);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      TestQDQModelAccuracy(BuildTileOnnxF32(spec),
                           BuildQDQTileOnnx<uint16_t>(spec),
                           po, spec.opset, ExpectedEPNodeAssignment::All);
      break;
    default:
      ADD_FAILURE() << "Unsupported Tile QDQ dtype: " << spec.qdq_dtype;
  }
}

}  // namespace

class QnnAcc_Tile_Accuracy_QDQTest : public ::testing::TestWithParam<TileQDQSpec> {};

TEST_P(QnnAcc_Tile_Accuracy_QDQTest, Case) {
  RunTileQDQAccuracy(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    , QnnAcc_Tile_Accuracy_QDQTest,
    ::testing::ValuesIn(kTileQDQSpecs),
    [](const ::testing::TestParamInfo<TileQDQSpec>& i) { return std::string(i.param.name); });

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS && QNN_EP_ACCURACY_UT
