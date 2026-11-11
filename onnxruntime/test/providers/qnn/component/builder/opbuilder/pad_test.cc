// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {
namespace {

std::string CheckPad(const std::vector<int64_t>& input_shape, const std::vector<int64_t>& pads,
                     const std::string& mode, bool initializer, bool validate = false,
                     const std::vector<int64_t>& output_shape = {4, 4}) {
  g_mock_init_reg.clear();
  OpBuilderTestContext ctx;
  SetupMockInitRegistryStubs(ctx);
  std::vector<FakeOpAttr> attrs{FakeOpAttr::MakeString("mode", mode)};
  auto input = MakeMockIODef("input", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, input_shape);
  std::vector<OrtNodeUnitIODef> inputs{input};
  if (initializer) {
    MockInitSpec spec;
    spec.elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    spec.dims = {static_cast<int64_t>(pads.size())};
    // The registry's GetTensorData stub requires backing storage even for a zero-element tensor.
    // Keep dims == {0} for empty pads so unpacking still produces an empty vector and reaches the guard.
    spec.raw_bytes.resize(pads.empty() ? sizeof(int64_t) : pads.size() * sizeof(int64_t));
    if (!pads.empty()) {
      std::memcpy(spec.raw_bytes.data(), pads.data(), spec.raw_bytes.size());
    }
    g_mock_init_reg.Add("pads", std::move(spec));
    inputs.push_back(MakeMockIODef("pads", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                                   std::vector<int64_t>{static_cast<int64_t>(pads.size())}));
  } else {
    attrs.push_back(FakeOpAttr::MakeInt64s("pads", pads));
  }
  auto wrapper = ctx.CreateWrapper(qnn::ModelSettings{}, qnn::QnnBackendType::CPU);
  auto output = MakeMockIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, output_shape);
  auto node = MakeMockNodeUnit("Pad", std::move(inputs), {output}, "pad", "", initializer ? 11 : 10, 0,
                               std::move(attrs));
  auto status = qnn::GetOpBuilder("Pad")->AddToModelBuilder(*wrapper, node, ctx.ort_logger, validate);
  return status.IsOK() ? std::string{} : status.GetErrorMessage();
}

}  // namespace

TEST(QnnUnit_Pad_ComponentTest, InvalidPadsLength_RejectsAttributeAndInitializerBeforeRankIndexing) {
  ASSERT_NE(qnn::GetOpBuilder("Pad"), nullptr);
  for (bool initializer : {false, true}) {
    for (bool validate : {false, true}) {
      for (const std::string mode : {"constant", "reflect", "edge"}) {
        for (const std::vector<int64_t> pads : {std::vector<int64_t>{}, {1}, {1, 0, 0}, {-1, 0, 0}, {1, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0}}) {
          SCOPED_TRACE("initializer=" + std::to_string(initializer) + " validate=" + std::to_string(validate) +
                       " mode=" + mode + " pads=" + ::testing::PrintToString(pads));
          auto error = CheckPad({4, 4}, pads, mode, initializer, validate);
          ASSERT_FALSE(error.empty());
          EXPECT_NE(error.find("twice the rank of input 0"), std::string::npos) << error;
        }
      }
    }
  }
}

TEST(QnnUnit_Pad_ComponentTest, ReflectZeroDimension_RejectsBeforeUnsignedSubtraction) {
  for (bool initializer : {false, true}) {
    auto error = CheckPad({0, 4}, {0, 0, 0, 0}, "reflect", initializer);
    ASSERT_FALSE(error.empty());
    EXPECT_NE(error.find("Pad amount should not be greater"), std::string::npos);
  }
}

TEST(QnnUnit_Pad_ComponentTest, ReflectPaddingAtOrAboveDimension_RejectsBothEnds) {
  for (bool initializer : {false, true}) {
    for (const std::vector<int64_t> pads : {std::vector<int64_t>{4, 0, 0, 0}, {0, 0, 4, 0}, {5, 0, 0, 0}, {0, 0, 5, 0}}) {
      auto error = CheckPad({4, 4}, pads, "reflect", initializer);
      ASSERT_FALSE(error.empty());
      EXPECT_NE(error.find("Pad amount should not be greater"), std::string::npos);
    }
  }
}

TEST(QnnUnit_Pad_ComponentTest, MixedPadsDimensionOverflow_RejectsBeforeAddition) {
  const int64_t max_dimension = std::numeric_limits<uint32_t>::max();
  for (bool initializer : {false, true}) {
    // Exercise overflow at the beginning pad and at the end pad separately.
    for (const std::vector<int64_t> pads : {std::vector<int64_t>{1, -1, 0, 0}, {0, -1, 1, 0}, {1, -1, 1, 0}}) {
      auto error = CheckPad({max_dimension, 4}, pads, "constant", initializer);
      ASSERT_FALSE(error.empty());
      EXPECT_NE(error.find("padded dimension exceeds uint32 range"), std::string::npos);
    }
  }
}

TEST(QnnUnit_Pad_ComponentTest, ValidPads_AttributeAndInitializerTranslationSucceeds) {
  for (bool initializer : {false, true}) {
    for (const std::string mode : {"constant", "reflect", "edge"}) {
      SCOPED_TRACE(mode);
      EXPECT_TRUE(CheckPad({4, 4}, {1, 1, 1, 1}, mode, initializer, false, {6, 6}).empty());
    }
    // Reflect allows exactly dimension - 1 values at either end.
    EXPECT_TRUE(CheckPad({4, 4}, {3, 3, 3, 3}, "reflect", initializer, false, {10, 10}).empty());
    EXPECT_TRUE(CheckPad({4, 4}, {-1, -1, -1, -1}, "constant", initializer, false, {2, 2}).empty());
    EXPECT_TRUE(CheckPad({4, 4}, {1, -1, -1, 1}, "constant", initializer).empty());
    EXPECT_TRUE(CheckPad({4, 4}, {0, 0, 0, 0}, "constant", initializer).empty());
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
