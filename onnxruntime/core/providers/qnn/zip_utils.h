#pragma once
#include <string>
namespace onnxruntime {
bool UnzipFile(const std::string& zip_path,
              const std::string& output_dir);
}  // namespace onnxruntime