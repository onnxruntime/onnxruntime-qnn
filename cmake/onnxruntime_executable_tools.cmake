set(EXECUTABLE_TOOLS_SRC_DIR ${REPO_ROOT}/samples/executable_tools)
onnxruntime_add_executable(
    executable_tools
    ${EXECUTABLE_TOOLS_SRC_DIR}/main.cpp
    ${EXECUTABLE_TOOLS_SRC_DIR}/utils.cpp
    ${EXECUTABLE_TOOLS_SRC_DIR}/model_info.cpp
)
target_link_libraries(executable_tools onnxruntime onnx)
