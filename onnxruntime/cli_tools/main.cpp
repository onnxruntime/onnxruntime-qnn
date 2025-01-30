#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
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

int main(int, char* argv[]) {
    std::string model_dir(argv[1]);
    std::cout << model_dir << std::endl;
    std::basic_string<PATH_CHAR_TYPE> model_path = find_model_path(model_dir);
    if (model_path.size() <= 0) {
        std::cout << ".onnx model should be provided" << std::endl;
        exit(0);
    }

    // model
    std::wcout << "model_path " << model_path << std::endl;
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    std::cout << ORT_API_VERSION << std::endl;
    OrtEnv* env;
    g_ort->CreateEnv(ORT_LOGGING_LEVEL_VERBOSE, "test", &env);
    OrtSessionOptions* session_options;
    g_ort->CreateSessionOptions(&session_options);
    g_ort->SetIntraOpNumThreads(session_options, 1);
    g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC);

    // std::vector<const char*> options_keys = {"runtime"};
    // std::vector<const char*> options_values = {"QnnCpu.dll"};

    // g_ort->SessionOptionsAppendExecutionProvider(
    //     session_options, "QNN",
    //     options_keys.data(), options_values.data(), options_keys.size()
    // );

    OrtSession* session;
    g_ort->CreateSession(env, model_path.c_str(), session_options, &session);
    std::cout << "Successfully CreateSession" << std::endl;

    OrtAllocator* allocator;
    g_ort->GetAllocatorWithDefaultOptions(&allocator);
    size_t num_input_nodes;
    g_ort->SessionGetInputCount(session, &num_input_nodes);
    std::cout << "num_input_nodes: " << num_input_nodes << std::endl;

    std::vector<const char*> input_node_names;
    std::vector<std::vector<int64_t>> input_node_dims;
    std::vector<ONNXTensorElementDataType> input_types;
    std::vector<OrtValue*> input_tensors;

    input_node_names.resize(num_input_nodes);
    input_node_dims.resize(num_input_nodes);
    input_types.resize(num_input_nodes);
    input_tensors.resize(num_input_nodes);
    for (size_t i = 0; i < num_input_nodes; i++) {
        // Get input node names
        char* input_name;
        g_ort->SessionGetInputName(session, i, allocator, &input_name);
        input_node_names[i] = input_name;
        std::cout << "input_name " << i << ": " << input_node_names[i] << std::endl;

        // Get input node types
        OrtTypeInfo* type_info;
        g_ort->SessionGetInputTypeInfo(session, i, &type_info);
        const OrtTensorTypeAndShapeInfo* tensor_info;
        g_ort->CastTypeInfoToTensorInfo(type_info, &tensor_info);
        ONNXTensorElementDataType type;
        g_ort->GetTensorElementType(tensor_info, &type);
        input_types[i] = type;
        std::cout << "input_types " << i << ": " << input_types[i] << std::endl;

        // Get input shapes/dims
        size_t num_dims;
        g_ort->GetDimensionsCount(tensor_info, &num_dims);
        input_node_dims[i].resize(num_dims);
        g_ort->GetDimensions(tensor_info, input_node_dims[i].data(), num_dims);
        std::cout << "input_node_dims " << i << ": [";
        for (int j=0; j<input_node_dims[i].size(); ++j)
            std::cout << ' ' << input_node_dims[i][j];
        std::cout << " ]" << std::endl;

        if (type_info) g_ort->ReleaseTypeInfo(type_info);
    }

    // Get output
    size_t num_output_nodes;
    std::vector<const char*> output_node_names;
    std::vector<std::vector<int64_t>> output_node_dims;
    std::vector<OrtValue*> output_tensors;
    g_ort->SessionGetOutputCount(session, &num_output_nodes);
    output_node_names.resize(num_output_nodes);
    output_node_dims.resize(num_output_nodes);
    output_tensors.resize(num_output_nodes);
    for (size_t i = 0; i < num_output_nodes; i++) {
        // Get output node names
        char* output_name;
        g_ort->SessionGetOutputName(session, i, allocator, &output_name);
        output_node_names[i] = output_name;
        std::cout << "output_node_names " << i << ": " << output_node_names[i] << std::endl;

        OrtTypeInfo* type_info;
        g_ort->SessionGetOutputTypeInfo(session, i, &type_info);
        const OrtTensorTypeAndShapeInfo* tensor_info;
        g_ort->CastTypeInfoToTensorInfo(type_info, &tensor_info);

        // Get output shapes/dims
        size_t num_dims;
        g_ort->GetDimensionsCount(tensor_info, &num_dims);
        output_node_dims[i].resize(num_dims);
        g_ort->GetDimensions(tensor_info, (int64_t*)output_node_dims[i].data(), num_dims);

        if (type_info) g_ort->ReleaseTypeInfo(type_info);
    }

    // Multiple test_data_set_X
    std::vector<std::basic_string<PATH_CHAR_TYPE>> test_data_sets = find_test_data_sets(model_dir);
    for (auto const test_data_set : test_data_sets) {
        std::wcout << test_data_set << std::endl;
        auto test_data_set_iter = std::filesystem::directory_iterator(test_data_set);
        size_t in_idx = 0;
        // Multiple input .pb in each test_data_set_X
        for (auto it=std::filesystem::begin(test_data_set_iter); it != std::filesystem::end(test_data_set_iter); ++it, ++in_idx) {
            auto test_data_entry = *it;
            auto filename_str = test_data_entry.path().filename().native();
            const std::basic_string<PATH_CHAR_TYPE> prefix = ORT_TSTR("input_");
            if (filename_str.compare(0, prefix.size(), prefix) == 0) {
                std::wcout << "filename_str " << filename_str << std::endl;
                std::cout << input_node_names[in_idx] << std::endl;
                // input data
                size_t input_data_size = 1;
                std::cout << "input_node_dims " << in_idx << ": [";
                for (int j=0; j<input_node_dims[in_idx].size(); ++j) {
                    std::cout << ' ' << input_node_dims[in_idx][j];
                    input_data_size = input_data_size* input_node_dims[in_idx][j];
                }
                std::cout << " ]" << std::endl;
                size_t input_data_length = input_data_size * sizeof(float);
                std::vector<float> input_data(input_data_size, 1.0);
                // Inference
                OrtMemoryInfo* memory_info;
                g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
                g_ort->CreateTensorWithDataAsOrtValue(
                    memory_info, reinterpret_cast<void*>(input_data.data()),
                    input_data_length, input_node_dims[in_idx].data(), input_node_dims[in_idx].size(), input_types[in_idx], &input_tensors[in_idx]
                );
                // Read Input .pb
                std::ifstream input_raw_file(filename_str, std::ios::binary);
                // calculate number of bytes
                input_raw_file.seekg(0, std::ios::end);
                const size_t num_elements = input_raw_file.tellg() / sizeof(float);
                // read the data
                input_raw_file.seekg(0, std::ios::beg);
                input_raw_file.read(reinterpret_cast<char*>(&input_data[0]), num_elements * sizeof(float));
                g_ort->Run(
                    session, nullptr,
                    input_node_names.data(), (const OrtValue* const*)input_tensors.data(), input_tensors.size(),
                    output_node_names.data(), output_node_names.size(), output_tensors.data()
                );
                std::cout << "Successfully Inference " << in_idx << std::endl;
            }
        }
        // Dump output of each out node
        for (size_t out_idx=0; out_idx < num_output_nodes; out_idx++) {
            std::cout << output_node_names[out_idx] << std::endl;
            OrtValue* actual_output_value = output_tensors[out_idx];
            OrtTensorTypeAndShapeInfo* out_tensor_info;
            g_ort->GetTensorTypeAndShape(actual_output_value, &out_tensor_info);
            size_t n_element;
            g_ort->GetTensorShapeElementCount(out_tensor_info, &n_element);
            ONNXTensorElementDataType out_dtype;
            g_ort->GetTensorElementType(out_tensor_info, &out_dtype);
            size_t element_size = GetONNXTypeSize(out_dtype);
            std::cout << "GetElementCount " << n_element << std::endl;
            std::cout << "GetTensorElementType " << out_dtype << std::endl;
            std::cout << "GetONNXTypeSize " << element_size << std::endl;
            void* output_buffer;
            g_ort->GetTensorMutableData(output_tensors[out_idx], &output_buffer);
            #ifdef _WIN32
                std::wstring outfile = std::wstring(L"out_") + std::to_wstring(out_idx) + std::wstring(L".raw");
            #else
                std::string outfile = std::string("out_") + std::to_string(out_idx) + std::string(".raw")
            #endif
            std::ofstream fout(std::filesystem::path(test_data_set) / outfile, std::ios::binary);
            fout.write(reinterpret_cast<const char*>(output_buffer), n_element * element_size);
            fout.close();
        }
        std::cout << "Successfully Save Outputs" << std::endl;
    }
    return 0;
}
