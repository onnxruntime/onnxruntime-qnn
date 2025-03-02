

#include <iostream>
#include "model_info.hpp"
#include <onnxruntime_cxx_api.h>

OnnxModelInfo::OnnxModelInfo(const OrtApi* g_ort, const OrtSession* session, OrtAllocator* allocator) {
    g_ort->SessionGetInputCount(session, &num_in_tensors);
    g_ort->SessionGetOutputCount(session, &num_out_tensors);

    in_tensor_names.resize(num_in_tensors);
    in_tensor_dims.resize(num_in_tensors);
    in_tensor_element_types.resize(num_in_tensors);
    in_tensors.resize(num_in_tensors);
    for (size_t i = 0; i < num_in_tensors; i++) {
        // Get tensor name, tensor info
        g_ort->SessionGetInputName(session, i, allocator, &in_tensor_names[i]);

        OrtTypeInfo* type_info;
        const OrtTensorTypeAndShapeInfo* tensor_info;
        g_ort->SessionGetInputTypeInfo(session, i, &type_info);
        g_ort->CastTypeInfoToTensorInfo(type_info, &tensor_info);
        g_ort->GetTensorElementType(tensor_info, &in_tensor_element_types[i]);

        // Get tensor shapes/dims
        size_t num_dims;
        g_ort->GetDimensionsCount(tensor_info, &num_dims);
        in_tensor_dims[i].resize(num_dims);
        g_ort->GetDimensions(tensor_info, in_tensor_dims[i].data(), num_dims);
        for (size_t j=0; j<in_tensor_dims[i].size(); ++j) {
            // If onnx model has dynamic dimension on tensors, e.g. [N, 3, 224, 224]
            // g_ort->GetDimensions yields [-1, 3, 224, 224]. We need to handle it with abs().
            in_tensor_dims[i][j] = abs(in_tensor_dims[i][j]);
        }

        if (type_info) g_ort->ReleaseTypeInfo(type_info);
    }
    out_tensor_names.resize(num_out_tensors);
    out_tensor_dims.resize(num_out_tensors);
    out_tensor_element_types.resize(num_out_tensors);
    out_tensors.resize(num_out_tensors);
    for (size_t i = 0; i < num_out_tensors; i++) {
        g_ort->SessionGetOutputName(session, i, allocator, &out_tensor_names[i]);
        OrtTypeInfo* type_info;
        const OrtTensorTypeAndShapeInfo* tensor_info;
        g_ort->SessionGetOutputTypeInfo(session, i, &type_info);
        g_ort->CastTypeInfoToTensorInfo(type_info, &tensor_info);
        g_ort->GetTensorElementType(tensor_info, &out_tensor_element_types[i]);

        // Get tensor shapes/dims
        size_t num_dims;
        g_ort->GetDimensionsCount(tensor_info, &num_dims);
        out_tensor_dims[i].resize(num_dims);
        g_ort->GetDimensions(tensor_info, out_tensor_dims[i].data(), num_dims);
        for (size_t j=0; j<out_tensor_dims[i].size(); ++j) {
            // If onnx model has dynamic dimension on tensors, e.g. [N, 3, 224, 224]
            // g_ort->GetDimensions yields [-1, 3, 224, 224]. We need to handle it with abs().
            out_tensor_dims[i][j] = abs(out_tensor_dims[i][j]);
        }

        if (type_info) g_ort->ReleaseTypeInfo(type_info);
    }
}
size_t OnnxModelInfo::get_num_in_tensors() { return num_in_tensors; }
std::vector<char*> OnnxModelInfo::get_in_tensor_names() { return in_tensor_names; }
std::vector<std::vector<int64_t>> OnnxModelInfo::get_in_tensor_dims() { return in_tensor_dims; }
std::vector<ONNXTensorElementDataType> OnnxModelInfo::get_in_tensor_element_types() { return in_tensor_element_types; }
std::vector<OrtValue*>& OnnxModelInfo::get_in_tensors() { return in_tensors; }

size_t OnnxModelInfo::get_num_out_tensors() { return num_out_tensors; }
std::vector<char*> OnnxModelInfo::get_out_tensor_names() { return out_tensor_names; }
std::vector<std::vector<int64_t>> OnnxModelInfo::get_out_tensor_dims() { return out_tensor_dims; }
std::vector<ONNXTensorElementDataType> OnnxModelInfo::get_out_tensor_element_types() { return out_tensor_element_types; }
std::vector<OrtValue*>& OnnxModelInfo::get_out_tensors() { return out_tensors; }
void OnnxModelInfo::PrintOnnxModelInfo() {
    std::cout << "num_in_tensors: " << num_in_tensors << std::endl;
    std::cout << "num_out_tensors: " << num_out_tensors << std::endl;
    for (size_t i = 0; i < num_in_tensors; i++) {
        std::cout << "in_tensor_dims " << i << ": [";
        for (size_t j=0; j<in_tensor_dims[i].size(); ++j) {
            std::cout << ' ' << in_tensor_dims[i][j];
        }
        std::cout << " ]" << std::endl;
    }
    for (size_t i = 0; i < num_out_tensors; i++) {
        std::cout << "out_tensor_dims " << i << ": [";
        for (size_t j=0; j<out_tensor_dims[i].size(); ++j) {
            std::cout << ' ' << out_tensor_dims[i][j];
        }
        std::cout << " ]" << std::endl;
        std::cout << "TensorElementType " << out_tensor_element_types[i] << std::endl;
        std::cout << "ElementSize " << GetONNXTypeSize(out_tensor_element_types[i]) << std::endl;
    }
}

size_t GetONNXTypeSize(ONNXTensorElementDataType dtype) {
    switch (dtype) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return 8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return 1;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            return 8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            return 1;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return 8;
        default:
            throw std::runtime_error("Unsupported ONNX data type");
    }
}