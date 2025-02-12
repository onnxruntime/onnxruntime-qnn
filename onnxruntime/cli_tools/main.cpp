#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <filesystem>
#include <fstream>
#include <string>
#include "core/platform/path_lib.h"
#include "utils.hpp"

void collect_in_out_tensor_information(
    const OrtApi* g_ort, const OrtSession* session, OrtAllocator* allocator,
    size_t& num_tensors,
    std::vector<const char*>& tensor_names,
    std::vector<std::vector<int64_t>>& tensor_dims,
    std::vector<ONNXTensorElementDataType>& tensor_element_types,
    std::vector<OrtValue*>& tensors,
    bool is_input
) {
    if (is_input) {
        g_ort->SessionGetInputCount(session, &num_tensors);
    }
    else {
        g_ort->SessionGetOutputCount(session, &num_tensors);
    }
    tensor_names.resize(num_tensors);
    tensor_dims.resize(num_tensors);
    tensor_element_types.resize(num_tensors);
    tensors.resize(num_tensors);
    for (size_t i = 0; i < num_tensors; i++) {
        // Get tensor name, tensor info
        char* tensor_name;
        OrtTypeInfo* type_info;
        if (is_input) {
            g_ort->SessionGetInputName(session, i, allocator, &tensor_name);
            g_ort->SessionGetInputTypeInfo(session, i, &type_info);
        } else {
            g_ort->SessionGetOutputName(session, i, allocator, &tensor_name);
            g_ort->SessionGetOutputTypeInfo(session, i, &type_info);
        }
        const OrtTensorTypeAndShapeInfo* tensor_info;
        ONNXTensorElementDataType tensor_element_type;
        g_ort->CastTypeInfoToTensorInfo(type_info, &tensor_info);
        g_ort->GetTensorElementType(tensor_info, &tensor_element_type);
        tensor_names[i] = tensor_name;
        tensor_element_types[i] = tensor_element_type;

        // Get tensor shapes/dims
        size_t num_dims;
        g_ort->GetDimensionsCount(tensor_info, &num_dims);
        tensor_dims[i].resize(num_dims);
        g_ort->GetDimensions(tensor_info, tensor_dims[i].data(), num_dims);
        std::cout << "tensor_dims " << i << ": [";
        for (size_t j=0; j<tensor_dims[i].size(); ++j) {
            // If onnx model has dynamic dimension on tensors, e.g. [N, 3, 224, 224]
            // g_ort->GetDimensions yields [-1, 3, 224, 224]. We need to handle it with abs().
            tensor_dims[i][j] = abs(tensor_dims[i][j]);
            std::cout << ' ' << tensor_dims[i][j];
        }
        std::cout << " ]" << std::endl;

        if (type_info) g_ort->ReleaseTypeInfo(type_info);
    }
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

    std::vector<const char*> options_keys = {"backend_path"};
    std::vector<const char*> options_values = {R"(C:\Users\hungjuiw\workspace\onnxruntime\build\build_cli_tools\RelWithDebInfo\RelWithDebInfo\QnnCpu.dll)"};

    g_ort->SessionOptionsAppendExecutionProvider(
        session_options, "QNN",
        options_keys.data(), options_values.data(), options_keys.size()
    );

    OrtSession* session;
    g_ort->CreateSession(env, model_path.c_str(), session_options, &session);
    std::cout << "Successfully CreateSession" << std::endl;

    OrtAllocator* allocator;
    g_ort->GetAllocatorWithDefaultOptions(&allocator);

    // Get input tensors
    size_t num_input_tensors;
    std::vector<const char*> input_tensor_names;
    std::vector<std::vector<int64_t>> input_tensor_dims;
    std::vector<ONNXTensorElementDataType> input_tensor_element_types;
    std::vector<OrtValue*> input_tensors;
    collect_in_out_tensor_information(
        g_ort, session, allocator,
        num_input_tensors,
        input_tensor_names,
        input_tensor_dims,
        input_tensor_element_types,
        input_tensors,
        true
    );
    // Get output tensors
    size_t num_output_tensors;
    std::vector<const char*> output_tensor_names;
    std::vector<std::vector<int64_t>> output_tensor_dims;
    std::vector<ONNXTensorElementDataType> output_tensor_element_types;
    std::vector<OrtValue*> output_tensors;
    collect_in_out_tensor_information(
        g_ort, session, allocator,
        num_output_tensors,
        output_tensor_names,
        output_tensor_dims,
        output_tensor_element_types,
        output_tensors,
        false
    );
    std::cout << "input_name "  << input_tensor_names[0] << std::endl;
    std::cout << "output_name "  << output_tensor_names[0] << std::endl;
    std::cout << "num_input_tensors: " << num_input_tensors << std::endl;
    std::cout << "num_output_tensors: " << num_output_tensors << std::endl;

    // Multiple test_data_set_X
    std::vector<std::basic_string<PATH_CHAR_TYPE>> test_data_sets = find_test_data_sets(model_dir);
    for (auto const& test_data_set : test_data_sets) {
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
                std::cout << input_tensor_names[in_idx] << std::endl;
                // input data
                size_t input_data_size = 1;
                std::cout << "input_tensor_dims " << in_idx << ": [";
                for (size_t j=0; j<input_tensor_dims[in_idx].size(); ++j) {
                    std::cout << ' ' << input_tensor_dims[in_idx][j];
                    input_data_size = input_data_size* input_tensor_dims[in_idx][j];
                }
                std::cout << " ]" << std::endl;

                size_t input_data_length = input_data_size * sizeof(float);
                std::vector<float> input_data(input_data_size, 1.0);
                
                // Inference
                OrtMemoryInfo* memory_info;
                g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
                g_ort->CreateTensorWithDataAsOrtValue(
                    memory_info, reinterpret_cast<void*>(input_data.data()),
                    input_data_length, input_tensor_dims[in_idx].data(), input_tensor_dims[in_idx].size(), input_tensor_element_types[in_idx], &input_tensors[in_idx]
                );
                // Read Input .pb
                std::ifstream input_raw_file(filename_str, std::ios::binary);
                // calculate number of bytes
                input_raw_file.seekg(0, std::ios::end);
                const size_t num_elements = input_raw_file.tellg() / sizeof(float);
                // read the data
                input_raw_file.seekg(0, std::ios::beg);
                input_raw_file.read(reinterpret_cast<char*>(&input_data[0]), num_elements * sizeof(float));
            }
        }
        g_ort->Run(
            session, nullptr,
            input_tensor_names.data(), (const OrtValue* const*)input_tensors.data(), input_tensors.size(),
            output_tensor_names.data(), output_tensor_names.size(), output_tensors.data()
        );
        std::cout << "Successfully Inference " << in_idx << std::endl;
        dump_output_tensors_to_raws(
            std::filesystem::path(test_data_set), g_ort,
            num_output_tensors,
            output_tensor_dims,
            output_tensor_element_types,
            output_tensors
        );
        std::cout << "Successfully Save Outputs" << std::endl;
    }
    return 0;
}
