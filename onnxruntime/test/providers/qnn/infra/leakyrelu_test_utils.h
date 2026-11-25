// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include "test/providers/qnn/infra/qnn_unit_test_utils.h"
#include "test/providers/qnn/infra/specs/builder/opbuilder/leakyrelu_specs.h"

namespace onnxruntime {
namespace test {
namespace {

// Models the operator boundary after QDQ grouping; no numerical inference.
// Keep this helper local to the TU that owns its initializer registry.
inline MockNodeUnit MakeLeakyReluNode(const LeakyReluSpec& spec, bool explicit_alpha = true) {
  const int since_version = spec.opset < 16 ? 6 : 16;
  std::vector<FakeOpAttr> attrs;
  if (explicit_alpha) attrs.push_back(FakeOpAttr::MakeFloat("alpha", spec.alpha));
  g_mock_init_reg.clear();
  if (spec.dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
    auto input_scale = g_mock_init_reg.AddScalarFloat("input_scale", kLeakyReluInputScale);
    auto input_zp = g_mock_init_reg.AddScalarUint8("input_zp", kLeakyReluInputZeroPoint);
    auto output_scale = g_mock_init_reg.AddScalarFloat("output_scale", kLeakyReluOutputScale);
    auto output_zp = g_mock_init_reg.AddScalarUint8("output_zp", kLeakyReluOutputZeroPoint);
    auto input = MakeMockQDQIODef("data", spec.dtype, spec.shape, input_scale, input_zp);
    auto output = MakeMockQDQIODef("output", spec.dtype, spec.shape, output_scale, output_zp);
    return MakeMockQDQNodeUnit("LeakyRelu", {input}, {output}, "LeakyRelu_node", "", since_version, 0,
                               std::move(attrs));
  }
  return MakeMockNodeUnit("LeakyRelu", {MakeMockIODef("data", spec.dtype, spec.shape)},
                          {MakeMockIODef("output", spec.dtype, spec.shape)}, "LeakyRelu_node", "",
                          since_version, 0, std::move(attrs));
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime

#endif
