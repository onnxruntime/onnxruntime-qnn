// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Runs a model with a Upsample operator on the QNN HTP backend. Checks the graph node assignment
// and that inference outputs for QNN EP and CPU EP match.
template <typename DataType>
static void RunUpsampleTestOnCPU(const TestInputDef<DataType>& input_def,
                                 const TestInputDef<float>& scales_def,
                                 std::vector<ONNX_NAMESPACE::AttributeProto>&& attrs,
                                 ExpectedEPNodeAssignment expected_ep_assignment,
                                 int opset = 9) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  if (opset <= 7) {
    const std::vector<float>& scales = scales_def.GetRawData();
    attrs.push_back(test::MakeAttribute("scales", scales));

    RunQnnModelTest(BuildOpTestCase<DataType>("Upsample_node", "Upsample", {input_def}, {}, attrs),
                    provider_options,
                    opset,
                    expected_ep_assignment);
  } else {
    RunQnnModelTest(BuildOpTestCase<DataType, float>("Upsample_node", "Upsample", {input_def}, {scales_def}, attrs),
                    provider_options,
                    opset,
                    expected_ep_assignment);
  }
}

/*
QNN HTP backend tests for the QDQ Upsample model is bypassed and can not be enabled.

ONNX Upsample is deprecated in domain version 10. However, ONNX QuantizeLinear and DequantizeLinear are enabled in
domain version 10. Their conditions are mutually exclusive, so it is not possible for these ops to coexist in the
same domain version.
*/

}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
