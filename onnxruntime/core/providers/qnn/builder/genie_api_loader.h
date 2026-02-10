
#pragma once

#include <mutex>
#include <stdexcept>
#include <vector>


// Genie symbol types declarations

typedef void* GenieNodeConfig;
typedef void* GenieNode;
typedef void* GenieLog;

typedef int Genie_Status_t;
typedef int GenieNode_IOName_t;



typedef enum {
  GENIE_LOG_LEVEL_ERROR   = 0,
  GENIE_LOG_LEVEL_WARN    = 1,
  GENIE_LOG_LEVEL_INFO    = 2,
  GENIE_LOG_LEVEL_VERBOSE = 3
} GenieLog_Level_t;

typedef void* GenieLogConfig_Handle_t;


typedef void (*GenieLog_Callback_t)(GenieLog logHandle,
                                    const char* fmt,
                                    GenieLog_Level_t level,
                                    uint64_t timestamp,
                                    va_list argp);


// Output CallBack
typedef void (*GenieNode_IOCallback_t)(
    const void* data,
    const size_t dataSize,
    const char* outputConfig,
    void* userData);

typedef Genie_Status_t (*TYPE_GenieNodeConfig_createFromJson)(
    const char* json,
    GenieNodeConfig** out);

typedef Genie_Status_t (*TYPE_GenieNode_create)(
    GenieNodeConfig* cfg,
    GenieNode** out);

typedef Genie_Status_t (*TYPE_GenieNode_setData)(
    GenieNode* dlg,
    GenieNode_IOName_t io_name,
    const void* data,
    const size_t dataSize,
    const char* dataConfig);

typedef Genie_Status_t (*TYPE_GenieNode_getData)(
    GenieNode* dlg,
    GenieNode_IOName_t io_name,
    const char* ioConfig,
    GenieNode_IOCallback_t nodeCallback);

typedef Genie_Status_t (*TYPE_GenieNode_execute)(
    GenieNode* dlg,
    const char* executionConfig,
    const void* userData);

typedef Genie_Status_t (*TYPE_GenieLog_create)(
  const GenieLogConfig_Handle_t configHandle,
  const GenieLog_Callback_t callback,
  GenieLog_Level_t logLevel,
  GenieLog** out_logger);

  
typedef Genie_Status_t (*TYPE_GenieNodeConfig_bindLogger)(
  GenieNodeConfig* cfg,
  GenieLog* logger);

  
typedef Genie_Status_t (*TYPE_GenieLog_free)(
  GenieLog* logger);


typedef Genie_Status_t (*TYPE_GenieNode_free)(
    GenieNode* dlg);

typedef Genie_Status_t (*TYPE_GenieNodeConfig_free)(
    GenieNodeConfig* cfg);

// GenieApi: holds all resolved function pointers
struct GenieApi {
    TYPE_GenieNodeConfig_createFromJson  NodeConfig_createFromJson;
    TYPE_GenieNode_create                Node_create;
    TYPE_GenieNode_setData               Node_setData;
    TYPE_GenieNode_getData               Node_getData;
    TYPE_GenieNode_execute               Node_execute;
    TYPE_GenieNode_free                  Node_free;
    TYPE_GenieNodeConfig_free            NodeConfig_free;
    TYPE_GenieLog_create                 Log_create;
    TYPE_GenieNodeConfig_bindLogger      NodeConfig_bindLogger;
    TYPE_GenieLog_free                   Log_free;

};

// GenieApiLoader: resolves and owns symbol table
class GenieApiLoader {
public:
    explicit GenieApiLoader(void* shared_library_handle);

    // Lazy initialize & return reference
    const GenieApi& Get();

private:
    void Init();                  // loads symbols
    GenieApi api_;                // resolved symbols
    void* handle_;                // dlopen/LoadLibrary handle
    std::once_flag init_flag_;    // ensures Init() runs once
};



// Per-node runtime state used by ONNX Runtime during execution
struct GenieNodeState {
    const GenieApi* api = nullptr;
    GenieNodeConfig* config = nullptr;
    GenieNode*       node   = nullptr;
    GenieLog*        genieLogger = nullptr;

    struct IODesc {
        GenieNode_IOName_t io_name;
        std::string        ort_name;
        std::vector<int64_t> shape;
        size_t             elem_size;
        size_t             byte_size;
    };
    std::vector<IODesc> inputs;
    std::vector<IODesc> outputs;
    std::mutex mu;
};

struct GenieNodeBuilder {
  const GenieApi* api;
  std::string json_config;
  std::vector<GenieNodeState::IODesc> inputs;
  std::vector<GenieNodeState::IODesc> outputs;
};
