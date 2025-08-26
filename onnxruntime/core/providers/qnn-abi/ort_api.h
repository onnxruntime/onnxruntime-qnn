// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

// This compilation unit (ort_api.h/.cc) encapsulates the interface between the EP and ORT in a manner
// that allows QNN EP to built either as a static library or a dynamic shared library.
// The preprocessor macro `BUILD_QNN_EP_STATIC_LIB` is defined and set to 1 if QNN EP
// is built as a static library.
// Includes when building QNN EP as a shared library
#include "core/providers/shared_library/provider_api.h"
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"

#include "core/session/onnxruntime_c_api.h"

#include "core/common/inlined_containers.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/onnxruntime_run_options_config_keys.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace onnxruntime {

// Redefine this macro for convenience since it is widely used.
// Original definition would try to log some message through default env which causes segfault.
#undef ORT_RETURN_IF_ERROR
#define ORT_RETURN_IF_ERROR(fn) \
  do {                          \
    Status _status = (fn);      \
    if (!_status.IsOK()) {      \
      return _status;           \
    }                           \
  } while (0)

#define RETURN_IF_NOT_OK(fn, ort_api)                                                                         \
  do {                                                                                                        \
    Status status = (fn);                                                                                     \
    if (!status.IsOK()) {                                                                                     \
      return (ort_api).CreateStatus(static_cast<OrtErrorCode>(status.Code()), status.ErrorMessage().c_str()); \
    }                                                                                                         \
  } while (0)

#define RETURN_IF_ERROR(fn)    \
  do {                         \
    OrtStatus* _status = (fn); \
    if (_status != nullptr) {  \
      return _status;          \
    }                          \
  } while (0)

#define RETURN_STATUS_IF_ERROR(fn, ort_api)                 \
  do {                                                      \
    OrtStatus* _status = (fn);                              \
    if (_status != nullptr) {                               \
      const char* msg = (ort_api).GetErrorMessage(_status); \
      (ort_api).ReleaseStatus(_status);                     \
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, msg);       \
    }                                                       \
  } while (0)

#define RETURN_IF(cond, ort_api, msg)                    \
  do {                                                   \
    if ((cond)) {                                        \
      return (ort_api).CreateStatus(ORT_EP_FAIL, (msg)); \
    }                                                    \
  } while (0)

#define RETURN_IF_NOT(cond, ort_api, msg) \
  RETURN_IF(!(cond), ort_api, msg)

#define QNN_RETURN_IF_STATUS_NOT_OK(ort_api_fn_call, ort_api, ret_val) \
  do {                                                                 \
    if (OrtStatus* _status = (ort_api_fn_call)) {                      \
      (ort_api).ReleaseStatus(_status);                                \
      return (ret_val);                                                \
    }                                                                  \
  } while (0)

struct ApiPtrs {
  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
  const OrtModelEditorApi& model_editor_api;
};

// Helper to release Ort one or more objects obtained from the public C API at the end of their scope.
template <typename T>
struct DeferOrtRelease {
  DeferOrtRelease(T** object_ptr, std::function<void(T*)> release_func)
      : objects_(object_ptr), count_(1), release_func_(release_func) {}

  DeferOrtRelease(T** objects, size_t count, std::function<void(T*)> release_func)
      : objects_(objects), count_(count), release_func_(release_func) {}

  ~DeferOrtRelease() {
    if (objects_ != nullptr && count_ > 0) {
      for (size_t i = 0; i < count_; ++i) {
        if (objects_[i] != nullptr) {
          release_func_(objects_[i]);
          objects_[i] = nullptr;
        }
      }
    }
  }
  T** objects_ = nullptr;
  size_t count_ = 0;
  std::function<void(T*)> release_func_ = nullptr;
};

template <typename T>
struct Factory {
  template <typename... Params>
  static inline std::unique_ptr<T> Create(Params&&... params) {
#if BUILD_QNN_EP_STATIC_LIB
    return std::make_unique<T>(std::forward<Params>(params)...);
#else
    return T::Create(std::forward<Params>(params)...);
#endif
  }
};

namespace QDQ {

// Define NodeGroup structure similar to the one in shared/utils.h
struct OrtNodeGroup {
  std::vector<const OrtNode*> dq_nodes;
  std::vector<const OrtNode*> q_nodes;
  const OrtNode* target_node;
  const OrtNode* redundant_clip_node{nullptr};
};

}  // namespace QDQ

struct OrtNodeUnitIODef {
  // TODO: Update this.
  struct QuantParam {
    const OrtValueInfo* scale;
    const OrtValueInfo* zero_point{nullptr};
    std::optional<int64_t> axis{std::nullopt};
  };

  std::string name;
  ONNXTensorElementDataType type;
  std::vector<int64_t> shape;
  std::optional<QuantParam> quant_param;

  bool Exists() const noexcept { return !name.empty(); }
};

class OrtNodeUnit {
 public:
  // NodeUnit type
  enum class Type : uint8_t {
    SingleNode,  // The NodeUnit contains a single node
    QDQGroup,    // The NodeUnit contain a QDQ group of nodes, such as "DQ->Sigmoid->Q"
  };

 public:
  explicit OrtNodeUnit(const OrtNode* node, const OrtApi& ort_api);
  explicit OrtNodeUnit(const OrtGraph* graph, const QDQ::OrtNodeGroup& node_group, const OrtApi& ort_api);

  Type UnitType() const noexcept { return type_; }

  const std::vector<OrtNodeUnitIODef>& Inputs() const noexcept { return inputs_; }
  const std::vector<OrtNodeUnitIODef>& Outputs() const noexcept { return outputs_; }

  std::string Domain() const noexcept { return Ort::ConstNode(target_node_).GetDomain(); }
  std::string OpType() const noexcept { return Ort::ConstNode(target_node_).GetOperatorType(); }
  std::string Name() const noexcept { return Ort::ConstNode(target_node_).GetName(); }
  int SinceVersion() const noexcept { return Ort::ConstNode(target_node_).GetSinceVersion(); }
  // TODO: Id is in fact not equal to index.
  size_t Index() const noexcept { return Ort::ConstNode(target_node_).GetId(); }

  const OrtNode& GetNode() const noexcept { return *target_node_; }
  const OrtNode* GetRedundantClipNode() const noexcept { return redundant_clip_node_; }
  const std::vector<const OrtNode*>& GetDQNodes() const noexcept { return dq_nodes_; }
  const std::vector<const OrtNode*>& GetQNodes() const noexcept { return q_nodes_; }
  std::vector<const OrtNode*> GetAllNodesInGroup() const noexcept {
    std::vector<const OrtNode*> all_nodes = dq_nodes_;
    all_nodes.push_back(target_node_);
    if (redundant_clip_node_) {
      all_nodes.push_back(redundant_clip_node_);
    }
    all_nodes.reserve(all_nodes.size() + q_nodes_.size());
    for (auto& n : q_nodes_)
      all_nodes.push_back(n);
    return all_nodes;
  }

  size_t GetInputEdgesCount(const OrtApi& ort_api) const;
  std::vector<const OrtNode*> GetOutputNodes(const OrtApi& ort_api) const;

 private:
  // // Initialization for a NodeUnit that contains a single node
  Status InitForSingleNode(const OrtApi& ort_api);

  const std::vector<const OrtNode*> dq_nodes_;  // dq nodes for this NodeUnit, not necessarily all inputs
  const OrtNode* target_node_;
  const OrtNode* redundant_clip_node_ = nullptr;  // Optional redundant clip node for the QDQ group, nullptr if not present.
  const std::vector<const OrtNode*> q_nodes_;     // q-nodes for this NodeUnit. not necessarily all outputs
  const Type type_;

  std::vector<OrtNodeUnitIODef> inputs_;
  std::vector<OrtNodeUnitIODef> outputs_;
};

/**
 * Wrapping onnxruntime::Node for retrieving attribute values
 */

class OrtNodeAttrHelper {
 public:
  explicit OrtNodeAttrHelper(const OrtNode& node);

  // Get the attributes from the target node of the node_unit
  explicit OrtNodeAttrHelper(const OrtNodeUnit& node_unit);

  /*
   * Get with default
   */
  float Get(const std::string& key, float def_val) const;
  std::vector<float> Get(const std::string& key, const std::vector<float>& def_val) const;

  int64_t Get(const std::string& key, int64_t def_val) const;
  std::vector<int64_t> Get(const std::string& key, const std::vector<int64_t>& def_val) const;

  std::string Get(const std::string& key, std::string def_val) const;
  std::vector<std::string> Get(const std::string& key, const std::vector<std::string>& def_val) const;

  // Convert the i() or ints() of the attribute from int64_t to int32_t
  int32_t Get(const std::string& key, int32_t def_val) const;
  std::vector<int32_t> Get(const std::string& key, const std::vector<int32_t>& def_val) const;

  // Convert the i() or ints() of the attribute from int64_t to uint32_t
  uint32_t Get(const std::string& key, uint32_t def_val) const;
  std::vector<uint32_t> Get(const std::string& key, const std::vector<uint32_t>& def_val) const;

  /*
   * Get without default.
   */
  std::optional<float> GetFloat(const std::string& key) const;
  std::optional<std::vector<float>> GetFloats(const std::string& key) const;

  std::optional<int64_t> GetInt64(const std::string& key) const;
  std::optional<std::vector<int64_t>> GetInt64s(const std::string& key) const;

  std::optional<std::string> GetString(const std::string& key) const;

  bool HasAttr(const std::string& key) const;

 private:
  const OrtNode& node_;
};

OrtStatus* GetSessionConfigEntryOrDefault(const OrtApi& ort_api,
                                          const OrtSessionOptions& session_options,
                                          const std::string& config_key,
                                          const std::string& default_val,
                                          /*out*/ std::string& config_val);

PathString GetModelPathString(const OrtGraph* graph, const OrtApi& ort_api);

/**
 * Returns a lowercase version of the input string.
 * /param str The string to lowercase.
 * /return The lowercased string.
 */
inline std::string GetLowercaseString(std::string str) {
  // https://en.cppreference.com/w/cpp/string/byte/tolower
  // The behavior of tolower from <cctype> is undefined if the argument is neither representable as unsigned char
  // nor equal to EOF. To use tolower safely with a plain char (or signed char), the argument must be converted to
  // unsigned char.
  std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return str;
}

// Refer to OrtSessionOptions::GetProviderOptionPrefix.
std::string GetProviderOptionPrefix(const std::string& provider_name);

// TODO
// Not sure why Env::Default() fails inside EP, replicate below implementations from "core/platform/posix/env.cc" and
// "core/platform/windows/env.cc" to here.
PathString OrtGetRuntimePath();

Status OrtLoadDynamicLibrary(const PathString& wlibrary_filename, bool global_symbols, void** handle);

Status OrtUnloadDynamicLibrary(void* handle);

Status OrtGetSymbolFromLibrary(void* handle, const std::string& symbol_name, void** symbol);

// non-macro equivalent of TEMP_FAILURE_RETRY, described here:
// https://www.gnu.org/software/libc/manual/html_node/Interrupted-Primitives.html
template <typename TFunc, typename... TFuncArgs>
long int TempFailureRetry(TFunc retriable_operation, TFuncArgs&&... args) {
  long int result;
  do {
    result = retriable_operation(std::forward<TFuncArgs>(args)...);
  } while (result == -1 && errno == EINTR);
  return result;
}

Status ReadFileIntoBuffer(const ORTCHAR_T* file_path, int64_t offset, size_t length, gsl::span<char> buffer);

}  // namespace onnxruntime
