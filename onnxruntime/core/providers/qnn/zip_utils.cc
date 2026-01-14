#include "core/providers/qnn/zip_utils.h"
#include <zip.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstring>

namespace fs = std::filesystem;
namespace onnxruntime {
bool UnzipFile(const std::string& zip_path,
              const std::string& output_dir) {
 int err = 0;
 zip* za = zip_open(zip_path.c_str(), ZIP_RDONLY, &err);
 if (!za) {
    std::cout << "Cannot open zip: " << err << std::endl;
    return false;
 } 
 fs::create_directories(output_dir);
 zip_int64_t num_entries = zip_get_num_entries(za, 0);
 for (zip_int64_t i = 0; i < num_entries; ++i) {
   struct zip_stat st;
   zip_stat_init(&st);
   zip_stat_index(za, i, 0, &st);
   std::string out_path = output_dir + "/" + st.name;
   // directory entry
   if (st.name[strlen(st.name) - 1] == '/') {
     fs::create_directories(out_path);
     continue;
   }
   fs::create_directories(fs::path(out_path).parent_path());
   zip_file* zf = zip_fopen_index(za, i, 0);
   if (!zf) continue;
   std::ofstream out(out_path, std::ios::binary);
   char buffer[8192];
   zip_int64_t bytes_read;
   while ((bytes_read = zip_fread(zf, buffer, sizeof(buffer))) > 0) {
     out.write(buffer, bytes_read);
   }
   zip_fclose(zf);
 }
 zip_close(za);
 return true;
}
}  // namespace onnxruntime