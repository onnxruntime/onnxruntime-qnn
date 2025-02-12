#include <onnxruntime_cxx_api.h>
#include "core/platform/path_lib.h"

size_t GetONNXTypeSize(ONNXTensorElementDataType dtype);

std::basic_string<PATH_CHAR_TYPE> find_model_path(std::string model_dir);

std::vector<std::basic_string<PATH_CHAR_TYPE>> find_test_data_sets(std::string model_dir);