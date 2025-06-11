//<!--
// Copyright (c) 2022 - 2024 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//-->
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <math.h>

#include "CpuBackendUtils.hpp"
#include "CustomOpPackage.hpp"

using namespace qnn::custom;
using namespace qnn::custom::utils;

namespace softmax {

void QuantizeMultiplier(float realMultiplier, int64_t* outputMultiplier, int32_t* outputShift)
{
    if (realMultiplier == 0.) {
        *outputMultiplier = 0;
        *outputShift      = 0;
    } else {
        const double q = std::frexp(realMultiplier, outputShift);
        auto qFixed    = static_cast<int64_t>(::round(q * (1LL << 31)));
        if (qFixed == (1LL << 31)) {
            qFixed /= 2;
            *outputShift += 1;
        }
        if (*outputShift < -31) {
            *outputShift = 0;
            qFixed      = 0;
        }
        if (*outputShift > 30) {
            *outputShift = 30;
            qFixed      = (1LL << 31) - 1;
        }
        *outputMultiplier = static_cast<int32_t>(qFixed);
    }
}

Qnn_ErrorHandle_t evaluateQuantized(CustomOp* operation) {

  /**
   * Add code here
   */
  /**
   * To have good performance and stability, it is required to avoid heap memory
   * allocation in this function. The heap memory allocation includes but not
   * limited to calling malloc, operator new, constructing STL container objects
   * like std::vector with default allocator, and adding items like calling
   * std::vector::push_back to STL container objects with default allocator.
   *
   * Please check in SDK documentation for more information.
   */

  auto m_Input  = operation->getInput(0);
  auto m_Outputs = operation->getOutput(0);

  const uint32_t rank   = m_Outputs->rank;
  const size_t depth    = m_Outputs->currentDimensions[rank - 1];
  uint32_t tensorLength = 1;
  for (uint32_t j = 0; j < rank; ++j) {
    tensorLength *= (uint32_t)(m_Outputs->currentDimensions[j]);
  }

  const size_t numPixels = tensorLength / depth;
  uint8_t lookupTable[512];
  //Changes here for look up table approach
  const uint8_t maxVal = std::numeric_limits<uint8_t>::max();
  const int32_t max_uint16 = std::numeric_limits<uint16_t>::max();
  float scale = m_Input->quantizeParams.scaleOffsetEncoding.scale;
  for (int32_t val = 0; val <= maxVal; val++) {
      float input_to_exp = scale * (val - maxVal);
      int32_t temp = static_cast<int>(expf(input_to_exp) * max_uint16 + 0.5);
      temp = std::min(max_uint16, temp);
      uint8_t part1 = temp >> 8;
      uint8_t part2 = temp & 0xff;
      lookupTable[val] = part1;
      lookupTable[val + 256] = part2;
  }

  for (size_t pix = 0; pix < numPixels; ++pix) {
    const uint8_t* inData = (uint8_t*)m_Input->data + pix * depth;
    uint8_t* outData      = (uint8_t*)m_Outputs->data + pix * depth;
    uint8_t max_val = inData[0];
    uint8_t min_val = inData[0];

    // find the max element for max subtraction
    for (int j = 1; j < depth; j++) {
      max_val = std::max(max_val, static_cast<uint8_t>(inData[j]));
      min_val = std::min(min_val, static_cast<uint8_t>(inData[j]));
    }

   // compute exponentiations
    int32_t sum = 0;
    uint8_t* table_offset = &lookupTable[maxVal - max_val];
    for (int j = 0; j < depth; j++) {
        int part1 = table_offset[inData[j]];
        int part2 = table_offset[inData[j] + 256];
        int temp = part1;
        sum += ((temp << 8) + part2);
    }

    float inv_sum_exp = 1.0f / (sum * 0.003906);
    int64_t multiplier = 0;
    int shift = 0;
    QuantizeMultiplier(inv_sum_exp, &multiplier, &shift);
    shift = 31 - shift;
    const int64_t round = static_cast<int64_t>(1) << (shift - 1);
    int32_t prob_rescaled;
    for (int j = 0; j < depth; j++) {
        prob_rescaled = (table_offset[inData[j]] << 8) + table_offset[inData[j] + 256];
        int64_t result = prob_rescaled * multiplier + round;
        result = result >> shift;
        result = std::max(std::min(static_cast<int64_t>(255), result), static_cast<int64_t>(0));
        outData[j] =  static_cast<uint8_t>(result);
    }
  }

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t evaluateFloat(CustomOp* operation) {

  /**
   * To have good performance and stability, it is required to avoid heap memory
   * allocation in this function. The heap memory allocation includes but not
   * limited to calling malloc, operator new, constructing STL container objects
   * like std::vector with default allocator, and adding items like calling
   * std::vector::push_back to STL container objects with default allocator.
   *
   * Please check in SDK documentation for more information.
   */

  auto m_Input  = operation->getInput(0);
  auto m_Outputs = operation->getOutput(0);

  const uint32_t rank   = m_Outputs->rank;
  const size_t depth    = m_Outputs->currentDimensions[rank - 1];
  uint32_t tensorLength = 1;
  for (uint32_t j = 0; j < rank; ++j) {
    tensorLength *= (uint32_t)(m_Outputs->currentDimensions[j]);
  }
  const size_t numPixels = tensorLength / depth;
  for (size_t pix = 0; pix < numPixels; ++pix) {
    const float* inData = (float*)m_Input->data + pix * depth;
    float* outData      = (float*)m_Outputs->data + pix * depth;
    float max_val = inData[0];
    float min_val = inData[0];

    // find the max element for max subtraction
    for (int j = 1; j < depth; j++) {
      max_val = std::max(max_val, static_cast<float>(inData[j]));
      min_val = std::min(min_val, static_cast<float>(inData[j]));
    }
   // compute exponentiations
    float sum = 0.0f;
    float* table_offset = nullptr;
    for (int j = 0; j < depth; j++) {
        sum += expf(inData[j] - max_val);
    }
    float inv_sum_exp = 1.0f / sum;
    for (int j = 0; j < depth; j++) {
        outData[j] = expf(inData[j] - max_val) * inv_sum_exp;
    }
  }

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t finalize(const CustomOp* operation) {
  QNN_CUSTOM_BE_ENSURE_EQ(operation->numInput(), 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
  QNN_CUSTOM_BE_ENSURE_EQ(operation->numOutput(), 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)

  /**
   * Add code here
   **/

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t free(CustomOp& operation) {
  /**
   * Add code here
   **/

    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t execute(CustomOp* operation) {
  auto input  = operation->getInput(0);

  if (input->dataType == QNN_CPU_DATATYPE_FLOAT_32) {
    evaluateFloat(operation);
  } else if (input->dataType == QNN_CPU_DATATYPE_UINT_8) {
    evaluateQuantized(operation);
  }

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

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t validateOpConfig(Qnn_OpConfig_t opConfig) {
  QNN_CUSTOM_BE_ENSURE_EQ(
      strcmp(opConfig.v1.typeName, "Softmax"), 0, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)

  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfInputs, 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfOutputs, 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)

  return QNN_SUCCESS;
}
}  // namespace softmax

CustomOpRegistration_t* register_SoftmaxCustomOp() {
  using namespace softmax;
  static CustomOpRegistration_t SoftmaxRegister = {
      execute, finalize, free, validateOpConfig, populateFromNode};
  return &SoftmaxRegister;
}

REGISTER_OP(Softmax, register_SoftmaxCustomOp);
