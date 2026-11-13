// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "QnnInterface.h"
#include "QnnTypes.h"

#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace qnn {
namespace utils {
// The scalar overload is defined in qnn_utils.cc, alongside the public datatype overload.
std::ostream& operator<<(std::ostream& out, const Qnn_Scalar_t& scalar);
}  // namespace utils
}  // namespace qnn

namespace test {
namespace {

using qnn::utils::operator<<;

TEST(QnnUnit_StringDiagnosticsTest, ScalarStringPrintsValue) {
  Qnn_Scalar_t scalar = QNN_SCALAR_INIT;
  scalar.dataType = QNN_DATATYPE_STRING;
  scalar.stringValue = "string attribute";
  std::ostringstream stream;
  stream << scalar;
  EXPECT_EQ(stream.str(), "string attribute");
}

TEST(QnnUnit_StringDiagnosticsTest, ScalarEmptyStringPrintsEmptyValue) {
  Qnn_Scalar_t scalar = QNN_SCALAR_INIT;
  scalar.dataType = QNN_DATATYPE_STRING;
  scalar.stringValue = "";
  std::ostringstream stream;
  stream << scalar;
  EXPECT_TRUE(stream.good());
  EXPECT_TRUE(stream.str().empty());
}

TEST(QnnUnit_StringDiagnosticsTest, ScalarNullStringPrintsPlaceholder) {
  Qnn_Scalar_t scalar = QNN_SCALAR_INIT;
  scalar.dataType = QNN_DATATYPE_STRING;
  scalar.stringValue = nullptr;
  std::ostringstream stream;
  stream << scalar;
  EXPECT_TRUE(stream.good());
  EXPECT_EQ(stream.str(), "<null>");
}

TEST(QnnUnit_StringDiagnosticsTest, StringDatatypePrintsSymbolicName) {
  std::ostringstream stream;
  stream << QNN_DATATYPE_STRING;
  EXPECT_EQ(stream.str(), "QNN_DATATYPE_STRING");
}

std::string captured_string_value;
size_t captured_string_calls = 0;

void CaptureStringParameter(const Qnn_OpConfig_t& config) {
  ++captured_string_calls;
  ASSERT_EQ(config.version, QNN_OPCONFIG_VERSION_1);
  ASSERT_EQ(config.v1.numOfParams, 1u);
  ASSERT_NE(config.v1.params, nullptr);
  const Qnn_Param_t& param = config.v1.params[0];
  ASSERT_EQ(param.paramType, QNN_PARAMTYPE_SCALAR);
  ASSERT_EQ(param.scalarParam.dataType, QNN_DATATYPE_STRING);
  ASSERT_NE(param.scalarParam.stringValue, nullptr);
  captured_string_value = param.scalarParam.stringValue;
}

Qnn_ErrorHandle_t ValidateStringParameter(Qnn_BackendHandle_t, Qnn_OpConfig_t config) {
  CaptureStringParameter(config);
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t AddStringParameterNode(Qnn_GraphHandle_t, Qnn_OpConfig_t config) {
  CaptureStringParameter(config);
  return QNN_SUCCESS;
}

class QnnStringDiagnosticsTest : public testing::TestWithParam<std::string> {
 public:
  QnnStringDiagnosticsTest() = default;
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(QnnStringDiagnosticsTest);

 protected:
  void SetUp() override {
    captured_string_value.clear();
    captured_string_calls = 0;
    backend_manager_.QnnInterface().backendValidateOpConfig = ValidateStringParameter;
    backend_manager_.QnnInterface().graphAddNode = AddStringParameterNode;
  }

  std::unique_ptr<qnn::QnnModelWrapper> MakeWrapper() {
    const OrtGraph& graph = *reinterpret_cast<const OrtGraph*>(&graph_sentinel_);
    return std::make_unique<qnn::QnnModelWrapper>(
        graph, api_stub_.MakeApiPtrs(), logger_, *backend_manager_.Get(),
        input_info_, output_info_, qnn::ModelSettings{});
  }

  std::string AddStringParameter(qnn::QnnModelWrapper& wrapper) {
    // UDO attributes are temporary strings. Destroy the source before either
    // diagnostic path runs, exercising QnnParamWrapper's existing ownership.
    std::string param_key;
    {
      const std::string attribute = GetParam();
      Qnn_Scalar_t scalar = QNN_SCALAR_INIT;
      scalar.dataType = QNN_DATATYPE_STRING;
      scalar.stringValue = attribute.c_str();
      qnn::QnnParamWrapper param(0, "string_node", "mode", scalar);
      param_key = param.GetParamTensorName();
      EXPECT_TRUE(wrapper.AddParamWrapper(std::move(param)));
    }
    return param_key;
  }

  OrtApiStubContext api_stub_;
  Ort::Logger logger_{MakeNullLogger()};
  StubBackendManager backend_manager_{api_stub_.MakeApiPtrs(), logger_};
  qnn::GraphInputOutputInfo input_info_;
  qnn::GraphInputOutputInfo output_info_;
  int graph_sentinel_{};
};

TEST_P(QnnStringDiagnosticsTest, ValidationReachesBackendWithOwnedString) {
  auto wrapper = MakeWrapper();
  const std::string param_key = AddStringParameter(*wrapper);

  // No tensors are needed to exercise the scalar diagnostic before validation.
  ASSERT_TRUE(wrapper->CreateQnnNode("string_node", "test.package", "StringOp",
                                     {}, {}, {param_key}, /*do_op_validation=*/true));
  EXPECT_EQ(captured_string_calls, 1u);
  EXPECT_EQ(captured_string_value, GetParam());
}

TEST_P(QnnStringDiagnosticsTest, CompositionReachesBackendWithOwnedString) {
  auto wrapper = MakeWrapper();
  const std::string param_key = AddStringParameter(*wrapper);

  ASSERT_TRUE(wrapper->CreateQnnNode("string_node", "test.package", "StringOp",
                                     {}, {}, {param_key}, /*do_op_validation=*/false));
  ASSERT_TRUE(wrapper->ComposeQnnGraph());
  EXPECT_EQ(captured_string_calls, 1u);
  EXPECT_EQ(captured_string_value, GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    StringAttributes, QnnStringDiagnosticsTest,
    testing::Values(std::string{}, std::string{"mode"}, std::string(256, 's')));

}  // namespace
}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
