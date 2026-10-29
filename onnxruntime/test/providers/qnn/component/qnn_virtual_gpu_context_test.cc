// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS && \
    defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnxruntime_ep_device_ep_metadata_keys.h"
#include "test/providers/qnn/infra/qnn_test_windows_compat.h"
#include "core/providers/qnn/qnn_provider_factory.h"
#include "test/providers/qnn/qnn_test_utils.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

// Compile for GPU device 0x36334630 (SoC model 88), then execute its
// embedded context on the matching physical GPU. Constructing the virtual
// device through OrtEpApi avoids changing process-wide enumeration settings.
//
// Disable reason: will be enabled when the EP upgrades to a QAIRT release
// where the GPU backend supports Windows offline compilation.
TEST_F(QnnGPUBackendTests, DISABLED_VirtualGpuCompileContextLoadsOnPhysicalGpu) {
  constexpr uint32_t kGpuDeviceId = 0x36334630;
  const auto& api = Ort::GetApi();
  const auto& ep_api = *api.GetEpApi();
  const ApiPtrs api_ptrs{api, ep_api, *api.GetModelEditorApi()};
  RegisteredEpDeviceUniquePtr physical_ep;
  {
    Ort::SessionOptions registration_options;
    RegisterQnnEpLibrary(physical_ep, registration_options, kQnnExecutionProvider, {{"backend_type", "gpu"}});
  }
  ASSERT_NE(physical_ep, nullptr);
  const OrtHardwareDevice* physical_gpu = api.EpDevice_Device(physical_ep.get());
  ASSERT_EQ(api.HardwareDevice_Type(physical_gpu), OrtHardwareDeviceType_GPU);
  if (api.HardwareDevice_DeviceId(physical_gpu) != kGpuDeviceId) {
    GTEST_SKIP() << "Requires GPU device 0x36334630 (SoC model 88)";
  }
  const char* is_virtual = api.GetKeyValue(api.HardwareDevice_Metadata(physical_gpu), kOrtHardwareDevice_MetadataKey_IsVirtual);
  ASSERT_TRUE(is_virtual == nullptr || std::string(is_virtual) != "1");

  ModelTestBuilder builder;
  std::vector<float> input_data{-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, 7.0f};
  const std::vector<int64_t> shape{2, 3};
  MakeTestInput(builder, "input", TestInputDef<float>(shape, false, input_data));
  MakeTestInput(builder, "bias", TestInputDef<float>(shape, true, std::vector<float>(6, 1.0f)));
  builder.AddNode("add", "Add", {"input", "bias"}, {"output"});
  builder.MakeOutput<float>("output", shape);
  auto* opset = builder.model_.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);
  builder.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  std::string model;
  ASSERT_TRUE(builder.model_.SerializeToString(&model));

  Ort::AllocatorWithDefaultOptions allocator;
  void* compiled_buffer = nullptr;
  size_t compiled_size = 0;
  auto release_buffer = gsl::finally([&]() {
    if (compiled_buffer) allocator.Free(compiled_buffer);
  });
  {
    OrtKeyValuePairs* metadata_raw = nullptr;
    api.CreateKeyValuePairs(&metadata_raw);
    std::unique_ptr<OrtKeyValuePairs, decltype(api.ReleaseKeyValuePairs)> metadata(metadata_raw, api.ReleaseKeyValuePairs);
    api.AddKeyValuePair(metadata.get(), kOrtHardwareDevice_MetadataKey_IsVirtual, "1");
    OrtHardwareDevice* virtual_gpu_raw = nullptr;
    Ort::ThrowOnError(ep_api.CreateHardwareDevice(OrtHardwareDeviceType_GPU, api.HardwareDevice_VendorId(physical_gpu),
                                                  kGpuDeviceId, api.HardwareDevice_Vendor(physical_gpu), metadata.get(), &virtual_gpu_raw));
    std::unique_ptr<OrtHardwareDevice, decltype(ep_api.ReleaseHardwareDevice)> virtual_gpu(
        virtual_gpu_raw, ep_api.ReleaseHardwareDevice);
    QnnEpFactory factory("QNNExecutionProvider.virtual", api_ptrs);
    const OrtHardwareDevice* devices[] = {virtual_gpu.get()};
    OrtEpDevice* ep_device_raw = nullptr;
    size_t count = 0;
    Ort::ThrowOnError(factory.GetSupportedDevices(&factory, devices, 1, &ep_device_raw, 1, &count));
    std::unique_ptr<OrtEpDevice, decltype(ep_api.ReleaseEpDevice)> ep_device(ep_device_raw, ep_api.ReleaseEpDevice);
    ASSERT_EQ(count, 1u);
    ASSERT_NE(ep_device, nullptr);
    const char* target = api.GetKeyValue(api.EpDevice_EpMetadata(ep_device.get()), "target_architecture");
    ASSERT_NE(target, nullptr);
    ASSERT_STREQ(target, "88");

    Ort::SessionOptions options;
    options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
    options.AppendExecutionProvider_V2(*ort_env, {Ort::ConstEpDevice(ep_device.get())},
                                       std::unordered_map<std::string, std::string>{{"offload_graph_io_quantization", "0"}});
    Ort::ModelCompilationOptions compilation(*ort_env, options);
    compilation.SetInputModelFromBuffer(model.data(), model.size());
    compilation.SetEpContextEmbedMode(true);
    compilation.SetOutputModelBuffer(allocator, &compiled_buffer, &compiled_size);
    auto status = Ort::CompileModel(*ort_env, compilation);
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  }
  ASSERT_NE(compiled_buffer, nullptr);
  ASSERT_GT(compiled_size, 0u);
  ONNX_NAMESPACE::ModelProto compiled_model;
  ASSERT_TRUE(compiled_model.ParseFromString(std::string(static_cast<const char*>(compiled_buffer), compiled_size)));
  ASSERT_EQ(compiled_model.graph().node_size(), 1);
  const auto& context = compiled_model.graph().node(0);
  ASSERT_EQ(context.op_type(), "EPContext");
  bool virtual_source = false;
  for (const auto& attribute : context.attribute()) {
    if (attribute.name() == "source") virtual_source = attribute.s() == "QNNExecutionProvider.virtual";
  }
  ASSERT_TRUE(virtual_source);

  Ort::SessionOptions options;
  options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
  options.AddConfigEntry(kOrtSessionOptionsDisableModelCompile, "1");
  options.AppendExecutionProvider_V2(*ort_env, {Ort::ConstEpDevice(physical_ep.get())},
                                     std::unordered_map<std::string, std::string>{{"offload_graph_io_quantization", "0"}});
  Ort::Session gpu_session(*ort_env, compiled_buffer, compiled_size, options);
  Ort::Session cpu_session(*ort_env, model.data(), model.size(), Ort::SessionOptions{});
  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto input = Ort::Value::CreateTensor<float>(memory, input_data.data(), input_data.size(),
                                               shape.data(), shape.size());
  const char* input_names[] = {"input"};
  const char* output_names[] = {"output"};
  auto actual = gpu_session.Run(Ort::RunOptions{}, input_names, &input, 1, output_names, 1);
  auto expected = cpu_session.Run(Ort::RunOptions{}, input_names, &input, 1, output_names, 1);
  ASSERT_EQ(actual.size(), 1u);
  ASSERT_EQ(actual[0].GetTensorTypeAndShapeInfo().GetShape(), shape);
  for (size_t i = 0; i < input_data.size(); ++i) {
    EXPECT_FLOAT_EQ(actual[0].GetTensorData<float>()[i], expected[0].GetTensorData<float>()[i]);
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif
