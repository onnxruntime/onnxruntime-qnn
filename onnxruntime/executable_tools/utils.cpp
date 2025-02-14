#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include "core/platform/path_lib.h"
#include "onnx/onnx_pb.h"

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

void load_input_tensors_from_raws(
    std::filesystem::path inp_dir,
    const OrtApi* g_ort,
    size_t& num_input_tensors,
    std::vector<std::vector<int64_t>>& input_tensor_dims,
    std::vector<ONNXTensorElementDataType>& input_tensor_element_types,
    std::vector<OrtValue*>& input_tensors,
    std::vector<std::vector<float>>& input_data
) {
    // Multiple input .raw in each inp_dir (test_data_set_X)
    input_data.resize(num_input_tensors);
    for (size_t in_idx=0; in_idx < num_input_tensors; in_idx++) {
        #ifdef _WIN32
            std::wstring infile_name = std::wstring(L"input_") + std::to_wstring(in_idx) + std::wstring(L".raw");
        #else
            std::string infile_name = std::string("input_") + std::to_string(in_idx) + std::string(".raw")
        #endif
        auto infile_path = (inp_dir / infile_name);
        // input data
        size_t input_data_size = 1;
        std::cout << "input_tensor_dims " << in_idx << ": [";
        for (size_t j=0; j<input_tensor_dims[in_idx].size(); ++j) {
            std::cout << ' ' << input_tensor_dims[in_idx][j];
            input_data_size = input_data_size* input_tensor_dims[in_idx][j];
        }
        std::cout << " ]" << std::endl;

        input_data[in_idx].resize(input_data_size);
        size_t input_data_length = input_data_size * sizeof(float);
            
        // CreateTensor in Ort using input_data
        // The input_data should not be released until Inference completes
        OrtMemoryInfo* memory_info;
        g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
        g_ort->CreateTensorWithDataAsOrtValue(
            memory_info, reinterpret_cast<void*>(input_data[in_idx].data()),
            input_data_length,
            input_tensor_dims[in_idx].data(),
            input_tensor_dims[in_idx].size(),
            input_tensor_element_types[in_idx],
            &input_tensors[in_idx]
        );
        // Read Input .raw
        std::ifstream input_raw_file(infile_path, std::ios::binary);
        // calculate number of bytes
        input_raw_file.seekg(0, std::ios::end);
        const size_t num_elements = input_raw_file.tellg() / sizeof(float);
        // read the data
        input_raw_file.seekg(0, std::ios::beg);
        input_raw_file.read(
            reinterpret_cast<char*>(&input_data[in_idx][0]),
            num_elements * sizeof(float)
        );
    }
    return;
}

void load_input_tensors_from_pbs(
    std::filesystem::path inp_dir,
    const OrtApi* g_ort,
    size_t& num_input_tensors,
    std::vector<std::vector<int64_t>>& input_tensor_dims,
    std::vector<ONNXTensorElementDataType>& input_tensor_element_types,
    std::vector<OrtValue*>& input_tensors,
    std::vector<std::vector<float>>& input_data
) {
    // Multiple input .pb in each inp_dir (test_data_set_X)
    input_data.resize(num_input_tensors);
    for (size_t in_idx=0; in_idx < num_input_tensors; in_idx++) {
        #ifdef _WIN32
            std::wstring infile_name = std::wstring(L"input_") + std::to_wstring(in_idx) + std::wstring(L".pb");
        #else
            std::string infile_name = std::string("input_") + std::to_string(in_idx) + std::string(".pb")
        #endif
        const std::filesystem::path infile_path = (inp_dir / infile_name);
        std::ifstream inp(infile_path, std::ios::binary);
        // input_X.pb -> String
        std::string buffer;
        std::ifstream file(infile_path, std::ios::binary);
        file.seekg(0, std::ios::end);
        buffer.resize(file.tellg());
        file.seekg(0, std::ios::beg);
        file.read(&buffer[0], buffer.size());
        file.close();

        // String -> TensorProto
        ONNX_NAMESPACE::TensorProto tensor_proto;
        tensor_proto.ParseFromString(buffer);

        // TensorProto -> std::vector<float>
        size_t input_data_size = tensor_proto.raw_data().size();
        std::cout << "input_data_size " << input_data_size << std::endl;
        input_data[in_idx].resize(input_data_size);
        std::memcpy(
            input_data[in_idx].data(),
            tensor_proto.raw_data().data(), input_data_size
        );
        OrtMemoryInfo* memory_info;
        g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
        g_ort->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void*>(input_data[in_idx].data()),
            input_data_size,
            input_tensor_dims[in_idx].data(),
            input_tensor_dims[in_idx].size(),
            input_tensor_element_types[in_idx],
            &input_tensors[in_idx]
        );
    }
    return;
}

void dump_output_tensors_to_raws(
    std::filesystem::path out_dir,
    const OrtApi* g_ort,
    size_t& num_output_tensors,
    std::vector<std::vector<int64_t>>& output_tensor_dims,
    std::vector<ONNXTensorElementDataType>& output_tensor_element_types,
    std::vector<OrtValue*>& output_tensors
) {
    // Dump output of each out tensor
    for (size_t out_idx=0; out_idx < num_output_tensors; out_idx++) {
        // output data
        size_t output_data_size = 1;
        for (size_t j=0; j<output_tensor_dims[out_idx].size(); ++j) {
            output_data_size = output_data_size* output_tensor_dims[out_idx][j];
        }
        size_t element_size = GetONNXTypeSize(output_tensor_element_types[out_idx]);
        std::cout << "ElementCount " << output_data_size << std::endl;
        std::cout << "TensorElementType " << output_tensor_element_types[out_idx] << std::endl;
        std::cout << "ElementSize " << element_size << std::endl;
        void* output_buffer;
        g_ort->GetTensorMutableData(output_tensors[out_idx], &output_buffer);
        #ifdef _WIN32
            std::wstring outfile = std::wstring(L"out_") + std::to_wstring(out_idx) + std::wstring(L".raw");
        #else
            std::string outfile = std::string("out_") + std::to_string(out_idx) + std::string(".raw")
        #endif
        std::ofstream fout(out_dir / outfile, std::ios::binary);
        fout.write(reinterpret_cast<const char*>(output_buffer), output_data_size * element_size);
        fout.close();
    }
    return;
}

void dump_output_tensors_to_pbs(
    std::filesystem::path out_dir,
    const OrtApi* g_ort,
    size_t& num_output_tensors,
    std::vector<std::vector<int64_t>>& output_tensor_dims,
    std::vector<ONNXTensorElementDataType>& output_tensor_element_types,
    std::vector<OrtValue*>& output_tensors
) {
    // Dump output of each out tensor
    for (size_t out_idx=0; out_idx < num_output_tensors; out_idx++) {
        // output data
        ONNX_NAMESPACE::TensorProto tensor_proto;
        size_t output_data_size = 1;
        for (size_t j=0; j<output_tensor_dims[out_idx].size(); ++j) {
            tensor_proto.add_dims((int)output_tensor_dims[out_idx][j]);
            output_data_size = output_data_size* output_tensor_dims[out_idx][j];
        }
        size_t element_size = GetONNXTypeSize(output_tensor_element_types[out_idx]);
        std::cout << "ElementCount " << output_data_size << std::endl;
        std::cout << "TensorElementType " << output_tensor_element_types[out_idx] << std::endl;
        std::cout << "ElementSize " << element_size << std::endl;
        void* output_buffer;
        g_ort->GetTensorMutableData(output_tensors[out_idx], &output_buffer);
        #ifdef _WIN32
            std::wstring outfile = std::wstring(L"out_") + std::to_wstring(out_idx) + std::wstring(L".pb");
        #else
            std::string outfile = std::string("out_") + std::to_string(out_idx) + std::string(".pb")
        #endif
        tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT);
        tensor_proto.set_raw_data(output_buffer, output_data_size * element_size);
        std::cout << "[";
        for (size_t j=0; j<tensor_proto.dims_size(); ++j) {
            std::cout << " " << tensor_proto.dims((int) j);
        }
        std::cout << "]" << std::endl;
        std::ofstream fout(out_dir / outfile, std::ios::binary);
        tensor_proto.SerializeToOstream(&fout);
        fout.close();
    }
    return;
}
