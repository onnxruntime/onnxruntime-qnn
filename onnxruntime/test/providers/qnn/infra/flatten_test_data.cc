// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include "test/providers/qnn/infra/flatten_test_data.h"
#include "test/providers/qnn/qnn_test_utils.h"

namespace onnxruntime {
namespace test {
namespace {

template <typename QType>
FlattenQdqData MakeFlattenQdqData(const std::vector<int64_t>& shape) {
  const TestInputDef<float> definition(shape, false, -10.0f, 10.0f);
  ModelTestBuilder model;
  MakeTestInput<float>(model, "input", definition);
  const auto& tensor = model.feeds_.at("input");
  const auto count = tensor.GetTensorTypeAndShapeInfo().GetElementCount();
  const auto* values = tensor.GetTensorData<float>();
  const auto input_params = GetTestInputQuantParams<QType>(definition);
  // Flatten preserves every value: its CPU output range is the actual input
  // range, whereas the random input definition advertises [-10,10]. Use the
  // existing helper for both rounding and range calculation, without copying
  // the quantization formula into the fake-ORT translation unit.
  const auto output_params = GetDataQuantParams<QType>(gsl::span<const float>(values, count));
  return {{values, values + count},
          {input_params.scale, static_cast<uint32_t>(input_params.zero_point)},
          {output_params.scale, static_cast<uint32_t>(output_params.zero_point)}};
}

}  // namespace

FlattenQdqData GetFlattenQdqData(const std::vector<int64_t>& shape, bool use_uint16) {
  return use_uint16 ? MakeFlattenQdqData<uint16_t>(shape) : MakeFlattenQdqData<uint8_t>(shape);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
