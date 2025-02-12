#include <stdexcept>
#include <filesystem>
#include <onnxruntime_cxx_api.h>
#include "core/platform/path_lib.h"

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

std::basic_string<PATH_CHAR_TYPE> find_model_path(std::string model_dir) {
    std::basic_string<PATH_CHAR_TYPE> ret(ORT_TSTR(""));
    for (auto const& file_entry : std::filesystem::directory_iterator(model_dir)) {
        if (file_entry.is_regular_file() && file_entry.path().extension() == ORT_TSTR(".onnx")) {
            ret = file_entry.path().native();
        }
    }
    return ret;
}

std::vector<std::basic_string<PATH_CHAR_TYPE>> find_test_data_sets(std::string model_dir) {
    std::vector<std::basic_string<PATH_CHAR_TYPE>> ret = {};
    const std::basic_string<PATH_CHAR_TYPE> prefix = ORT_TSTR("test_data_set_");
    for (auto const& dir_entry : std::filesystem::directory_iterator(model_dir)) {
        if (dir_entry.is_directory() && dir_entry.path().filename().native().compare(0, prefix.size(), prefix) == 0) {
            ret.push_back(dir_entry.path().native());
        }
    }
    return ret;
}