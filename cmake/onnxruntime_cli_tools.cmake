set(CLI_TOOLS_SRC_DIR ${ONNXRUNTIME_ROOT}/cli_tools)
onnxruntime_add_executable(cli_tool ${CLI_TOOLS_SRC_DIR}/main.cpp ${CLI_TOOLS_SRC_DIR}/utils.cpp)
target_link_libraries(cli_tool onnxruntime)
