// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//==============================================================================
// Auto Generated Code for MyAddOpPackage
//==============================================================================
#include <iostream>
#include <string>

#include "CpuBackendUtils.hpp"
#include "CustomOpPackage.hpp"

using namespace qnn::custom;
using namespace qnn::custom::utils;

namespace increment {

Qnn_ErrorHandle_t execute(CustomOp* operation) {
  auto in = operation->getInput(0);
  auto out = operation->getOutput(0);
  float constant = static_cast<float>(backend_utils::getScalarParam(operation->getParam("constant")));
  const float* inData = (const float*)in->data;
  float* outData = (float*)out->data;

  // Calculate number of element of input
  size_t numInputs = 1;
  for (size_t i = 0; i < (int)in->rank; ++i) {
    numInputs *= in->currentDimensions[i];
  }
  for (size_t i = 0; i < numInputs; i++) {
    outData[i] = inData[i] + constant;
  }
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t finalize(const CustomOp* operation) {
  QNN_CUSTOM_BE_ENSURE_EQ(operation->numInput(), 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
  QNN_CUSTOM_BE_ENSURE_EQ(operation->numOutput(), 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
  const auto input = operation->getInput(0);
  const auto output = operation->getOutput(0);

  // for simplicity, only support int32
  QNN_CUSTOM_BE_ENSURE_EQ(
      input->dataType, QNN_CPU_DATATYPE_FLOAT_32, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE);

  QNN_CUSTOM_BE_ENSURE_EQ(
      input->dataType, output->dataType, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE);

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t populateFromNode(const QnnOpPackage_Node_t node,
                                   QnnOpPackage_GraphInfrastructure_t graphInfrastructure,
                                   CustomOp* operation) {
  // Add input
  for (uint32_t i = 0; i < numInputs(node); i++) {
    operation->addInput(getInput(node, i));
  }

  // Add output
  for (uint32_t i = 0; i < numOutputs(node); i++) {
    operation->addOutput(getOutput(node, i));
  }

  // Add param
  auto p = getParam(node, "constant");
  if (p.first) {
    operation->addParam("constant", p.second);
  } else {
    CustomOpParamPtr_t opPkgParam = new CustomOpParam;
    opPkgParam->name = "constant";
    opPkgParam->type = QNN_CPU_PARAMTYPE_SCALAR;
    opPkgParam->scalarParam = 1.0;
    operation->addParam("constant", opPkgParam);
  }
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t validateOpConfig(Qnn_OpConfig_t opConfig) {
  QNN_CUSTOM_BE_ENSURE_EQ(
      strcmp(opConfig.v1.typeName, "MyAdd"), 0, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)

  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfInputs, 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfOutputs, 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)

  return QNN_SUCCESS;
}
}  // namespace increment

CustomOpRegistration_t* register_MyAddCustomOp() {
  using namespace increment;
  static CustomOpRegistration_t MyAddRegister = {execute, finalize, nullptr, validateOpConfig, populateFromNode};
  return &MyAddRegister;
}

REGISTER_OP(MyAdd, register_MyAddCustomOp);
