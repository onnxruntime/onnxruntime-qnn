// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

/*
QNN HTP backend tests for the QDQ Upsample model is bypassed and can not be enabled.

ONNX Upsample is deprecated in domain version 10. However, ONNX QuantizeLinear and DequantizeLinear are enabled in
domain version 10. Their conditions are mutually exclusive, so it is not possible for these ops to coexist in the
same domain version.
*/

}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
