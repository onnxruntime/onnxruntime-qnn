#include <filesystem>
#include <onnxruntime_cxx_api.h>
#include "core/platform/path_lib.h"

size_t GetONNXTypeSize(ONNXTensorElementDataType dtype);

std::basic_string<PATH_CHAR_TYPE> find_model_path(std::string model_dir);

std::vector<std::basic_string<PATH_CHAR_TYPE>> find_test_data_sets(std::string model_dir);

void load_input_tensors_from_raws(
    std::filesystem::path inp_dir,
    const OrtApi* g_ort,
    size_t& num_input_tensors,
    std::vector<std::vector<int64_t>>& input_tensor_dims,
    std::vector<ONNXTensorElementDataType>& input_tensor_element_types,
    std::vector<OrtValue*>& input_tensors,
    std::vector<std::vector<float>>& input_data
);

void dump_output_tensors_to_raws(
    std::filesystem::path out_dir,
    const OrtApi* g_ort,
    size_t& num_output_tensors,
    std::vector<std::vector<int64_t>>& output_tensor_dims,
    std::vector<ONNXTensorElementDataType>& output_tensor_element_types,
    std::vector<OrtValue*>& output_tensors
);

void load_input_tensors_from_pbs(
    std::filesystem::path inp_dir,
    const OrtApi* g_ort,
    size_t& num_input_tensors,
    std::vector<std::vector<int64_t>>& input_tensor_dims,
    std::vector<ONNXTensorElementDataType>& input_tensor_element_types,
    std::vector<OrtValue*>& input_tensors,
    std::vector<std::vector<float>>& input_data
);

void dump_output_tensors_to_pbs(
    std::filesystem::path out_dir,
    const OrtApi* g_ort,
    size_t& num_output_tensors,
    std::vector<std::vector<int64_t>>& output_tensor_dims,
    std::vector<ONNXTensorElementDataType>& output_tensor_element_types,
    std::vector<OrtValue*>& output_tensors
);