// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Op-builder snapshot tests for TileOpBuilder.

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"
#include "test/providers/qnn/infra/specs/builder/opbuilder/tile_specs.h"
#include "test/providers/qnn/snapshot/snapshot.h"

using namespace onnxruntime;
using namespace onnxruntime::qnn;

namespace onnxruntime {
namespace test {

namespace {

const OrtValueInfo* AddTensorInt64(const std::string& name,
                                   std::vector<int64_t> dims,
                                   const std::vector<int64_t>& data) {
  MockInitSpec spec;
  spec.elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  spec.dims = std::move(dims);
  spec.raw_bytes.resize(data.size() * sizeof(int64_t));
  if (!data.empty()) {
    std::memcpy(spec.raw_bytes.data(), data.data(), spec.raw_bytes.size());
  }
  return g_mock_init_reg.Add(name, std::move(spec));
}

std::vector<int64_t> TileOutputShape(const std::vector<int64_t>& input_shape,
                                     const std::vector<int64_t>& repeats) {
  std::vector<int64_t> output_shape;
  output_shape.reserve(input_shape.size());
  for (size_t i = 0; i < input_shape.size(); ++i) {
    output_shape.push_back(input_shape[i] * repeats[i]);
  }
  return output_shape;
}

std::pair<const OrtValueInfo*, const OrtValueInfo*> RegisterQDQScaleZp(const TileQDQSpec& spec) {
  auto scale_vi = g_mock_init_reg.AddScalarFloat("data_scale", spec.scale);
  const OrtValueInfo* zp_vi = nullptr;
  if (spec.qdq_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
    zp_vi = g_mock_init_reg.AddScalarUint8("data_zp", static_cast<uint8_t>(spec.zero_point));
  } else if (spec.qdq_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) {
    zp_vi = g_mock_init_reg.AddScalarUint16("data_zp", static_cast<uint16_t>(spec.zero_point));
  } else {
    ADD_FAILURE() << "Unsupported Tile QDQ dtype: " << spec.qdq_dtype;
  }
  return {scale_vi, zp_vi};
}

void RunTileSnapshotQDQ(const TileQDQSpec& spec) {
  const IOpBuilder* builder = GetOpBuilder("Tile");
  ASSERT_NE(builder, nullptr);

  g_mock_init_reg.clear();
  auto [scale_vi, zp_vi] = RegisterQDQScaleZp(spec);
  AddTensorInt64("repeats", {static_cast<int64_t>(spec.repeats.size())}, spec.repeats);

  OpBuilderTestContext ctx;
  SetupMockInitRegistryStubs(ctx);
  QnnRealHtpBackendManagerContext htp;
  if (!htp.IsValid()) GTEST_SKIP() << "libQnnHtp.so not available";
  std::unique_ptr<QnnModelWrapper> wrapper =
      MakeSnapshotWrapperHtpJson(ctx, htp, {"data"}, {"output"});
  ASSERT_NE(wrapper, nullptr) << "Failed to initialize QNN graph for snapshot test";

  auto data = MakeMockQDQIODef("data", spec.qdq_dtype, spec.input_shape, scale_vi, zp_vi);
  auto repeats = MakeMockIODef("repeats", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                               std::vector<int64_t>{static_cast<int64_t>(spec.repeats.size())});
  auto output = MakeMockQDQIODef("output", spec.qdq_dtype,
                                 TileOutputShape(spec.input_shape, spec.repeats),
                                 scale_vi, zp_vi);
  auto node_unit = MakeMockQDQNodeUnit("Tile", {data, repeats}, {output}, "tile_node");

  auto status = builder->AddToModelBuilder(*wrapper, node_unit, ctx.ort_logger, false);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  ASSERT_TRUE(wrapper->ComposeQnnGraph(/*build_json_qnn_graph=*/true)) << "ComposeQnnGraph failed";
  AssertSnapshotJson(*wrapper, spec.name);
}

}  // namespace

class QnnSnapshot_Tile_OpBuilder_QDQTest : public ::testing::TestWithParam<TileQDQSpec> {};

TEST_P(QnnSnapshot_Tile_OpBuilder_QDQTest, Case) {
  RunTileSnapshotQDQ(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    , QnnSnapshot_Tile_OpBuilder_QDQTest,
    ::testing::ValuesIn(kTileQDQSpecs),
    [](const ::testing::TestParamInfo<TileQDQSpec>& i) { return std::string(i.param.name); });

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
