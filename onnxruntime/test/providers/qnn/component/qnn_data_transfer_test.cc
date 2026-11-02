// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"

#include "core/providers/qnn/qnn_data_transfer.h"

namespace onnxruntime {
namespace test {
namespace {

struct DataTransferStatusRecord {
  OrtErrorCode code;
  std::string message;
};

struct MemoryDeviceRecord {
  OrtMemoryInfoDeviceType device_type;
  OrtDeviceMemoryType memory_type;
  uint32_t vendor_id;
};

struct TensorRecord {
  void* data;
  size_t size;
  ONNXTensorElementDataType element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  std::vector<int64_t> shape;
};

class DataTransferStubContext {
 public:
  OrtApi ort_api{};
  OrtEpApi ep_api{};
  std::unordered_map<const OrtMemoryDevice*, MemoryDeviceRecord> memory_devices;
  std::unordered_map<const OrtValue*, TensorRecord> tensors;

  DataTransferStubContext() {
    ort_api.CreateStatus = [](OrtErrorCode code, const char* message) noexcept -> OrtStatus* {
      return reinterpret_cast<OrtStatus*>(
          new DataTransferStatusRecord{code, message == nullptr ? "" : message});
    };
    ort_api.ReleaseStatus = [](OrtStatus* status) noexcept {
      delete reinterpret_cast<DataTransferStatusRecord*>(status);
    };
    ort_api.GetErrorMessage = [](const OrtStatus* status) noexcept -> const char* {
      return reinterpret_cast<const DataTransferStatusRecord*>(status)->message.c_str();
    };
    ort_api.GetErrorCode = [](const OrtStatus* status) noexcept -> OrtErrorCode {
      return reinterpret_cast<const DataTransferStatusRecord*>(status)->code;
    };
    ort_api.GetTensorElementTypeAndShapeDataReference =
        [](const OrtValue* value, ONNXTensorElementDataType* element_type,
           const int64_t** shape, size_t* shape_len) noexcept -> OrtStatus* {
      auto* context = current_;
      auto it = context->tensors.find(value);
      if (it == context->tensors.end()) {
        return context->ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "unknown tensor");
      }

      *element_type = it->second.element_type;
      *shape = it->second.shape.data();
      *shape_len = it->second.shape.size();
      return nullptr;
    };
    ort_api.GetTensorSizeInBytes = [](const OrtValue* value, size_t* size) noexcept -> OrtStatus* {
      auto* context = current_;
      auto it = context->tensors.find(value);
      if (it == context->tensors.end()) {
        return context->ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "unknown tensor");
      }

      *size = it->second.size;
      return nullptr;
    };
    ort_api.GetTensorData = [](const OrtValue* value, const void** data) noexcept -> OrtStatus* {
      auto* context = current_;
      auto it = context->tensors.find(value);
      if (it == context->tensors.end()) {
        return context->ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "unknown tensor");
      }

      *data = it->second.data;
      return nullptr;
    };
    ort_api.GetTensorMutableData = [](OrtValue* value, void** data) noexcept -> OrtStatus* {
      auto* context = current_;
      auto it = context->tensors.find(value);
      if (it == context->tensors.end()) {
        return context->ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "unknown tensor");
      }

      *data = it->second.data;
      return nullptr;
    };

    ep_api.MemoryDevice_GetDeviceType =
        [](const OrtMemoryDevice* memory_device) noexcept -> OrtMemoryInfoDeviceType {
      auto* context = current_;
      auto it = context->memory_devices.find(memory_device);
      return it == context->memory_devices.end() ? OrtMemoryInfoDeviceType_CPU : it->second.device_type;
    };
    ep_api.MemoryDevice_GetMemoryType =
        [](const OrtMemoryDevice* memory_device) noexcept -> OrtDeviceMemoryType {
      auto* context = current_;
      auto it = context->memory_devices.find(memory_device);
      return it == context->memory_devices.end() ? OrtDeviceMemoryType_DEFAULT : it->second.memory_type;
    };
    ep_api.MemoryDevice_GetVendorId = [](const OrtMemoryDevice* memory_device) noexcept -> uint32_t {
      auto* context = current_;
      auto it = context->memory_devices.find(memory_device);
      return it == context->memory_devices.end() ? 0 : it->second.vendor_id;
    };
  }

  static thread_local DataTransferStubContext* current_;
};

thread_local DataTransferStubContext* DataTransferStubContext::current_ = nullptr;

class UseDataTransferStubs {
 public:
  explicit UseDataTransferStubs(DataTransferStubContext& context) noexcept
      : previous_{DataTransferStubContext::current_} {
    DataTransferStubContext::current_ = &context;
  }

  ~UseDataTransferStubs() {
    DataTransferStubContext::current_ = previous_;
  }

  UseDataTransferStubs(const UseDataTransferStubs&) = delete;
  UseDataTransferStubs& operator=(const UseDataTransferStubs&) = delete;

 private:
  DataTransferStubContext* previous_;
};

static OrtMemoryDevice* MakeFakeMemoryDevice(uintptr_t tag) {
  return reinterpret_cast<OrtMemoryDevice*>(tag);
}

static OrtValue* MakeFakeOrtValue(uintptr_t tag) {
  return reinterpret_cast<OrtValue*>(tag);
}

static OrtDataTransferImpl* CreateDataTransfer(DataTransferStubContext& context) {
  return new qnn::QnnHostAccessibleDataTransfer(context.ort_api, context.ep_api);
}

TEST(QnnUnit_DataTransferTest, CanCopyOnlyQnnHostAccessibleCpuMemory) {
  DataTransferStubContext context;
  UseDataTransferStubs use(context);
  OrtDataTransferImpl* transfer = CreateDataTransfer(context);

  const auto* cpu = MakeFakeMemoryDevice(10);
  const auto* qnn_shared = MakeFakeMemoryDevice(11);
  const auto* other_shared = MakeFakeMemoryDevice(12);
  const auto* gpu = MakeFakeMemoryDevice(13);
  context.memory_devices[cpu] = {
      OrtMemoryInfoDeviceType_CPU, OrtDeviceMemoryType_DEFAULT, 0};
  context.memory_devices[qnn_shared] = {
      OrtMemoryInfoDeviceType_CPU, OrtDeviceMemoryType_HOST_ACCESSIBLE, qnn::kQualcommPciVendorId};
  context.memory_devices[other_shared] = {
      OrtMemoryInfoDeviceType_CPU, OrtDeviceMemoryType_HOST_ACCESSIBLE, 0x1234};
  context.memory_devices[gpu] = {
      OrtMemoryInfoDeviceType_GPU, OrtDeviceMemoryType_DEFAULT, qnn::kQualcommPciVendorId};

  EXPECT_TRUE(transfer->CanCopy(transfer, cpu, qnn_shared));
  EXPECT_TRUE(transfer->CanCopy(transfer, qnn_shared, cpu));
  EXPECT_FALSE(transfer->CanCopy(transfer, qnn_shared, qnn_shared));
  EXPECT_FALSE(transfer->CanCopy(transfer, qnn_shared, other_shared));
  EXPECT_FALSE(transfer->CanCopy(transfer, cpu, cpu));
  EXPECT_FALSE(transfer->CanCopy(transfer, cpu, other_shared));
  EXPECT_FALSE(transfer->CanCopy(transfer, gpu, qnn_shared));
  EXPECT_FALSE(transfer->CanCopy(transfer, nullptr, qnn_shared));
  EXPECT_FALSE(transfer->CanCopy(transfer, qnn_shared, nullptr));
  EXPECT_FALSE(transfer->CanCopy(transfer, nullptr, nullptr));
  transfer->Release(transfer);
}

TEST(QnnUnit_DataTransferTest, ZeroTensorsIsNoOp) {
  DataTransferStubContext context;
  UseDataTransferStubs use(context);
  OrtDataTransferImpl* transfer = CreateDataTransfer(context);

  EXPECT_EQ(transfer->CopyTensors(transfer, nullptr, nullptr, nullptr, 0), nullptr);
  transfer->Release(transfer);
}

TEST(QnnUnit_DataTransferTest, RejectsNullTensorInputs) {
  DataTransferStubContext context;
  UseDataTransferStubs use(context);
  OrtDataTransferImpl* transfer = CreateDataTransfer(context);

  const auto expect_invalid_argument = [&](const OrtValue** sources, OrtValue** destinations) {
    OrtStatus* status = transfer->CopyTensors(transfer, sources, destinations, nullptr, 1);
    ASSERT_NE(status, nullptr);
    EXPECT_EQ(context.ort_api.GetErrorCode(status), ORT_INVALID_ARGUMENT);
    context.ort_api.ReleaseStatus(status);
  };

  auto* value = MakeFakeOrtValue(20);
  const OrtValue* sources[] = {value};
  OrtValue* destinations[] = {value};
  expect_invalid_argument(nullptr, destinations);
  expect_invalid_argument(sources, nullptr);

  sources[0] = nullptr;
  expect_invalid_argument(sources, destinations);
  sources[0] = value;
  destinations[0] = nullptr;
  expect_invalid_argument(sources, destinations);

  transfer->Release(transfer);
}

TEST(QnnUnit_DataTransferTest, RejectsNullDataForNonEmptyTensor) {
  DataTransferStubContext context;
  UseDataTransferStubs use(context);
  OrtDataTransferImpl* transfer = CreateDataTransfer(context);

  std::vector<uint8_t> data(4, 0);
  auto* source = MakeFakeOrtValue(20);
  auto* destination = MakeFakeOrtValue(21);
  context.tensors[source] = {
      nullptr, data.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {4}};
  context.tensors[destination] = {
      data.data(), data.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {4}};
  const OrtValue* sources[] = {source};
  OrtValue* destinations[] = {destination};

  OrtStatus* status = transfer->CopyTensors(transfer, sources, destinations, nullptr, 1);
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(context.ort_api.GetErrorCode(status), ORT_INVALID_ARGUMENT);
  EXPECT_NE(std::string(context.ort_api.GetErrorMessage(status)).find("null tensor data"),
            std::string::npos);
  context.ort_api.ReleaseStatus(status);

  transfer->Release(transfer);
}

TEST(QnnUnit_DataTransferTest, PropagatesOrtApiErrors) {
  DataTransferStubContext context;
  UseDataTransferStubs use(context);
  OrtDataTransferImpl* transfer = CreateDataTransfer(context);

  std::vector<uint8_t> data(4, 0);
  auto* source = MakeFakeOrtValue(20);
  auto* destination = MakeFakeOrtValue(21);
  context.tensors[destination] = {
      data.data(), data.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {4}};
  const OrtValue* sources[] = {source};
  OrtValue* destinations[] = {destination};

  OrtStatus* status = transfer->CopyTensors(transfer, sources, destinations, nullptr, 1);
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(context.ort_api.GetErrorCode(status), ORT_INVALID_ARGUMENT);
  EXPECT_STREQ(context.ort_api.GetErrorMessage(status), "unknown tensor");
  context.ort_api.ReleaseStatus(status);

  context.tensors[source] = {
      data.data(), data.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {4}};
  context.ort_api.GetTensorSizeInBytes = [](const OrtValue*, size_t*) noexcept -> OrtStatus* {
    return DataTransferStubContext::current_->ort_api.CreateStatus(ORT_FAIL, "size failure");
  };
  status = transfer->CopyTensors(transfer, sources, destinations, nullptr, 1);
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(context.ort_api.GetErrorCode(status), ORT_FAIL);
  EXPECT_STREQ(context.ort_api.GetErrorMessage(status), "size failure");
  context.ort_api.ReleaseStatus(status);

  transfer->Release(transfer);
}

TEST(QnnUnit_DataTransferTest, CopiesMultipleTensorsInBothDirections) {
  DataTransferStubContext context;
  UseDataTransferStubs use(context);
  OrtDataTransferImpl* transfer = CreateDataTransfer(context);

  std::vector<uint8_t> cpu_data{1, 2, 3, 4};
  std::vector<uint8_t> shared_data(cpu_data.size(), 0);
  std::vector<uint8_t> shared_data_2{5, 6, 7};
  std::vector<uint8_t> cpu_data_2(shared_data_2.size(), 0);
  auto* cpu_value = MakeFakeOrtValue(20);
  auto* shared_value = MakeFakeOrtValue(21);
  auto* shared_value_2 = MakeFakeOrtValue(22);
  auto* cpu_value_2 = MakeFakeOrtValue(23);
  context.tensors[cpu_value] = {
      cpu_data.data(), cpu_data.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  context.tensors[shared_value] = {
      shared_data.data(), shared_data.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  context.tensors[shared_value_2] = {
      shared_data_2.data(), shared_data_2.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  context.tensors[cpu_value_2] = {
      cpu_data_2.data(), cpu_data_2.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};

  const OrtValue* sources[] = {cpu_value, shared_value_2};
  OrtValue* destinations[] = {shared_value, cpu_value_2};
  EXPECT_EQ(transfer->CopyTensors(transfer, sources, destinations, nullptr, 2), nullptr);
  EXPECT_EQ(shared_data, cpu_data);
  EXPECT_EQ(cpu_data_2, shared_data_2);
  transfer->Release(transfer);
}

TEST(QnnUnit_DataTransferTest, RejectsSizeMismatch) {
  DataTransferStubContext context;
  UseDataTransferStubs use(context);
  OrtDataTransferImpl* transfer = CreateDataTransfer(context);

  std::vector<uint8_t> source_data{1, 2, 3, 4};
  std::vector<uint8_t> destination_data(3, 0);
  auto* source = MakeFakeOrtValue(20);
  auto* destination = MakeFakeOrtValue(21);
  context.tensors[source] = {
      source_data.data(), source_data.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  context.tensors[destination] = {
      destination_data.data(), destination_data.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {}};
  const OrtValue* sources[] = {source};
  OrtValue* destinations[] = {destination};

  OrtStatus* status = transfer->CopyTensors(transfer, sources, destinations, nullptr, 1);
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(context.ort_api.GetErrorCode(status), ORT_INVALID_ARGUMENT);
  EXPECT_NE(std::string(context.ort_api.GetErrorMessage(status)).find("sizes differ"), std::string::npos);
  context.ort_api.ReleaseStatus(status);

  transfer->Release(transfer);
}

TEST(QnnUnit_DataTransferTest, RejectsStringAndTypeMismatchButAllowsReshape) {
  DataTransferStubContext context;
  UseDataTransferStubs use(context);
  OrtDataTransferImpl* transfer = CreateDataTransfer(context);

  std::vector<uint8_t> source_data{1, 2, 3, 4};
  std::vector<uint8_t> destination_data(4);
  auto* source = MakeFakeOrtValue(20);
  auto* destination = MakeFakeOrtValue(21);
  context.tensors[source] = {
      source_data.data(), source_data.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, {1}};
  context.tensors[destination] = {
      destination_data.data(), destination_data.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, {1}};
  const OrtValue* sources[] = {source};
  OrtValue* destinations[] = {destination};

  OrtStatus* status = transfer->CopyTensors(transfer, sources, destinations, nullptr, 1);
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(context.ort_api.GetErrorCode(status), ORT_INVALID_ARGUMENT);
  EXPECT_NE(std::string(context.ort_api.GetErrorMessage(status)).find("string tensors"), std::string::npos);
  context.ort_api.ReleaseStatus(status);

  context.tensors[source].element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  context.tensors[destination].element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  status = transfer->CopyTensors(transfer, sources, destinations, nullptr, 1);
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(context.ort_api.GetErrorCode(status), ORT_INVALID_ARGUMENT);
  EXPECT_NE(std::string(context.ort_api.GetErrorMessage(status)).find("types differ"), std::string::npos);
  context.ort_api.ReleaseStatus(status);

  context.tensors[destination].element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  context.tensors[destination].shape = {1, 1};
  status = transfer->CopyTensors(transfer, sources, destinations, nullptr, 1);
  EXPECT_EQ(status, nullptr);
  EXPECT_EQ(destination_data, source_data);

  transfer->Release(transfer);
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
