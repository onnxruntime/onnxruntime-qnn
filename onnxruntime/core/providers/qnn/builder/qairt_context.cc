// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#ifdef USE_QAIRT_API
#include "core/providers/qnn/builder/qairt_backend_manager.h"
#include "QairtCpp/QairtBackend.hpp"
#include "QairtCpp/QairtContext.hpp"
#include "QairtCpp/QairtInfo.hpp"
#include "QairtCpp/System/QairtSystemApi.hpp"
#include "QairtCpp/System/QairtSystemContext.hpp"
#include <filesystem>
#include <limits>

namespace onnxruntime::qnn {
namespace {
Ort::Status ReadTensors(std::vector<qairt::SystemContextTensorInfo>& tensors,
                        Qnn_TensorType_t type, std::vector<QnnTensorWrapper>& result) {
  for (auto& tensor : tensors) {
    QnnQuantParamsWrapper quant;
    const auto& info = tensor.getQuantInfo();
    switch (info.getQuantizationType()) {
      case qairt::SystemContextQuantInfoType::QAIRT_QUANTIZATION_INFO_UNDEFINED:
        break;
      case qairt::SystemContextQuantInfoType::QAIRT_QUANTIZATION_INFO_SCALE_OFFSET: {
        const auto& value = info.getScaleOffset();
        quant = QnnQuantParamsWrapper::PerTensor(value.getScale(), value.getOffset());
        break;
      }
      case qairt::SystemContextQuantInfoType::QAIRT_QUANTIZATION_INFO_BW_SCALE_OFFSET: {
        const auto& value = info.getBwScaleOffset();
        quant = QnnQuantParamsWrapper::PerTensorBw(value.getScale(), value.getOffset(), value.getBw());
        break;
      }
      case qairt::SystemContextQuantInfoType::QAIRT_QUANTIZATION_INFO_AXIS_SCALE_OFFSET: {
        const auto& value = info.getAxisScaleOffset();
        std::vector<float> scales;
        std::vector<int32_t> offsets;
        for (const auto& entry : value.getScaleOffsets()) {
          scales.push_back(entry.getScale());
          offsets.push_back(entry.getOffset());
        }
        quant = QnnQuantParamsWrapper::PerChannel(scales, offsets, value.getAxis());
        break;
      }
      case qairt::SystemContextQuantInfoType::QAIRT_QUANTIZATION_INFO_BW_AXIS_SCALE_OFFSET: {
        const auto& value = info.getBwAxisScaleOffset();
        quant = QnnQuantParamsWrapper::PerChannelBw(value.getScales(), value.getOffsets(),
                                                    value.getAxis(), value.getBitwidth());
        break;
      }
      default:
        return MAKE_EP_FAIL("Unsupported QAIRT cached tensor quantization encoding.");
    }
    std::vector<uint32_t> shape;
    if (tensor.getRank()) {
      const auto* dims = tensor.getDimensions();
      RETURN_IF(!dims, "QAIRT cached tensor has missing dimensions.");
      shape.assign(dims, dims + tensor.getRank());
    }
    RETURN_IF(tensor.getId() > std::numeric_limits<uint32_t>::max(), "QAIRT tensor ID exceeds descriptor range.");
    QnnTensorWrapper wrapper(tensor.getName(), type, static_cast<Qnn_DataType_t>(tensor.getDataType()),
                             std::move(quant), std::move(shape));
    SetQnnTensorID(wrapper.GetQnnTensor(), static_cast<uint32_t>(tensor.getId()));
    result.push_back(std::move(wrapper));
  }
  return Ort::Status();
}
}  // namespace
std::string QairtBackendManager::GetSystemLibraryPath() const {
#ifdef _WIN32
  constexpr const char* name = "QairtSystem.dll";
#else
  constexpr const char* name = "libQairtSystem.so";
#endif
  return (std::filesystem::path(config_.backend_lib_path).parent_path() / name).string();
}
std::string QairtBackendManager::GetSdkVersion() {
  return api_->make<qairt::Info>().getBuildId();
}
Ort::Status QairtBackendManager::LoadContext(std::vector<char>& buffer, std::vector<CachedGraph>& graphs, qairt::Profile* profile) {
  try {
    RETURN_IF(buffer.empty(), "Empty QAIRT context binary.");
    const auto system_path = GetSystemLibraryPath();
    qairt::SystemApi system(system_path.c_str());
    auto system_context = system.make<qairt::SystemContext>();
    auto binary_info = system_context.getBinaryInfo(buffer.data(), buffer.size());
    auto config = api_->make<qairt::ContextConfiguration>();
    auto binary = api_->make<qairt::ContextBinaryBuffer>();
    binary.setData(buffer.data());
    binary.setSize(buffer.size());
    auto context = std::make_shared<qairt::Context>(profile
                                                        ? backend_->createContextFromBinary({}, config, binary, {}, *profile)
                                                        : backend_->createContextFromBinary({}, config, binary));
    std::vector<CachedGraph> loaded;
    for (auto& graph : binary_info.getGraphs()) {
      std::vector<QnnTensorWrapper> inputs, outputs;
      RETURN_IF_ERROR(ReadTensors(graph.getGraphInputs(), QNN_TENSOR_TYPE_APP_WRITE, inputs));
      RETURN_IF_ERROR(ReadTensors(graph.getGraphOutputs(), QNN_TENSOR_TYPE_APP_READ, outputs));
      loaded.push_back({context, std::make_unique<GraphInfo>(nullptr, graph.getGraphName(), nullptr,
                                                             std::move(inputs), std::move(outputs))});
    }
    graphs = std::move(loaded);
    return Ort::Status();
  } catch (const std::exception& e) {
    return MAKE_EP_FAIL(e.what());
  }
}
}  // namespace onnxruntime::qnn
#endif
