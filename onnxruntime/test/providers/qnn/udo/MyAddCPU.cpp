// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

//==============================================================================
// Auto Generated Code for MyAddOpPackage
//==============================================================================
#include <iostream>
#include <string>

#include "CpuBackendUtils.hpp"
#include "CustomOpPackage.hpp"

using namespace qnn::custom;
using namespace qnn::custom::utils;

namespace myadd {

Qnn_ErrorHandle_t execute(CustomOp* operation) {
  auto in = operation->getInput(0);
  auto out = operation->getOutput(0);
  float constant = static_cast<float>(backend_utils::getScalarParam(operation->getParam("constant")));
  const float* inData = (const float*)in->data;
  float* outData = (float*)out->data;

  // Calculate number of element of input
  size_t numInputs = 1;
  for (uint32_t i = 0; i < in->rank; ++i) {
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

  // for simplicity, only support float32
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
  // The getParam function returns a pair -> hasParam, paramValue
  // Check that parameter has be retrieved. Pair.first is false if it was not found and the paramValue is nullptr

  auto constantPair = getParam(node, "constant");

  QNN_CUSTOM_BE_ENSURE(constantPair.first, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)
  operation->addParam("constant", constantPair.second);

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t validateOpConfig(Qnn_OpConfig_t opConfig) {
  QNN_CUSTOM_BE_ENSURE_EQ(
      strcmp(opConfig.v1.typeName, "MyAdd"), 0, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)

  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfInputs, 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfOutputs, 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)

  return QNN_SUCCESS;
}
}  // namespace myadd

CustomOpRegistration_t* register_MyaddCustomOp() {
  using namespace myadd;
  static CustomOpRegistration_t MyaddRegister = {execute, finalize, nullptr, validateOpConfig, populateFromNode};
  return &MyaddRegister;
}

REGISTER_OP(MyAdd, register_MyaddCustomOp);
