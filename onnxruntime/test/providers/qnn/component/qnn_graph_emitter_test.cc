// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <string>
#include <type_traits>

#include "gtest/gtest.h"
#include "core/providers/qnn/builder/qnn_graph_emitter.h"

namespace onnxruntime::test {
namespace {

static_assert(!std::is_copy_constructible_v<qnn::QnnGraphEmitter>);
static_assert(!std::is_copy_assignable_v<qnn::QnnGraphEmitter>);
static_assert(!std::is_move_constructible_v<qnn::QnnGraphEmitter>);
static_assert(!std::is_move_assignable_v<qnn::QnnGraphEmitter>);

struct StubGraph {
  unsigned finalize_calls = 0;
  unsigned execute_calls = 0;
  unsigned config_calls = 0;
  Qnn_ErrorHandle_t finalize_result = QNN_SUCCESS;
  Qnn_ErrorHandle_t execute_result = QNN_SUCCESS;
  Qnn_ErrorHandle_t config_result = QNN_SUCCESS;
  const Qnn_Tensor_t* inputs = nullptr;
  Qnn_Tensor_t* outputs = nullptr;
  uint32_t num_inputs = 0;
  uint32_t num_outputs = 0;
  Qnn_ProfileHandle_t profile = nullptr;
  Qnn_SignalHandle_t signal = nullptr;
  const QnnGraph_Config_t** configs = nullptr;
};

struct StubContext {
  Qnn_GraphHandle_t graph;
  Qnn_ErrorHandle_t create_result = QNN_SUCCESS;
  Qnn_ErrorHandle_t retrieve_result = QNN_SUCCESS;
  unsigned create_calls = 0;
  unsigned retrieve_calls = 0;
  std::string name;
  const QnnGraph_Config_t** configs = nullptr;
};

class QnnGraphEmitterTest : public ::testing::Test {
 protected:
  QnnGraphEmitterTest() {
    qnn_interface_.errorGetMessage = [](Qnn_ErrorHandle_t, const char** message) -> Qnn_ErrorHandle_t {
      *message = "stub QNN error";
      return QNN_SUCCESS;
    };
    qnn_interface_.graphCreate = [](Qnn_ContextHandle_t context, const char* name,
                                    const QnnGraph_Config_t** configs, Qnn_GraphHandle_t* graph) -> Qnn_ErrorHandle_t {
      auto& stub = *static_cast<StubContext*>(context);
      ++stub.create_calls;
      stub.name = name;
      stub.configs = configs;
      *graph = stub.graph;  // Exercise SDK failures that still write an output handle.
      return stub.create_result;
    };
    qnn_interface_.graphRetrieve = [](Qnn_ContextHandle_t context, const char* name,
                                      Qnn_GraphHandle_t* graph) -> Qnn_ErrorHandle_t {
      auto& stub = *static_cast<StubContext*>(context);
      ++stub.retrieve_calls;
      stub.name = name;
      *graph = stub.graph;
      return stub.retrieve_result;
    };
    qnn_interface_.graphFinalize = [](Qnn_GraphHandle_t graph, Qnn_ProfileHandle_t profile,
                                      Qnn_SignalHandle_t signal) -> Qnn_ErrorHandle_t {
      auto& stub = *static_cast<StubGraph*>(graph);
      ++stub.finalize_calls;
      stub.profile = profile;
      stub.signal = signal;
      return stub.finalize_result;
    };
    qnn_interface_.graphExecute = [](Qnn_GraphHandle_t graph, const Qnn_Tensor_t* inputs, uint32_t num_inputs,
                                     Qnn_Tensor_t* outputs, uint32_t num_outputs, Qnn_ProfileHandle_t profile,
                                     Qnn_SignalHandle_t signal) -> Qnn_ErrorHandle_t {
      auto& stub = *static_cast<StubGraph*>(graph);
      ++stub.execute_calls;
      stub.inputs = inputs;
      stub.outputs = outputs;
      stub.num_inputs = num_inputs;
      stub.num_outputs = num_outputs;
      stub.profile = profile;
      stub.signal = signal;
      return stub.execute_result;
    };
    qnn_interface_.graphSetConfig = [](Qnn_GraphHandle_t graph,
                                       const QnnGraph_Config_t** configs) -> Qnn_ErrorHandle_t {
      auto& stub = *static_cast<StubGraph*>(graph);
      ++stub.config_calls;
      stub.configs = configs;
      return stub.config_result;
    };
  }

  StubGraph original_graph_;
  StubGraph replacement_graph_;
  StubContext original_context_{&original_graph_};
  StubContext replacement_context_{&replacement_graph_};
  QNN_INTERFACE_VER_TYPE qnn_interface_{};
  qnn::QnnGraphEmitter emitter_{qnn_interface_};
};

TEST_F(QnnGraphEmitterTest, CreateGraphForwardsArgumentsAndRejectsReinitialization) {
  QnnGraph_Config_t config{};
  const QnnGraph_Config_t* configs[]{&config, nullptr};
  ASSERT_TRUE(emitter_.CreateGraph(&original_context_, "original", configs).IsOK());
  EXPECT_EQ(original_context_.configs, configs);
  EXPECT_EQ(original_context_.name, "original");

  EXPECT_FALSE(emitter_.CreateGraph(&replacement_context_, "replacement").IsOK());
  EXPECT_EQ(replacement_context_.create_calls, 0u);
  EXPECT_EQ(emitter_.Graph(), &original_graph_);
  EXPECT_EQ(emitter_.Context(), &original_context_);
}

TEST_F(QnnGraphEmitterTest, FailedCreationDoesNotInstallPartialHandlesAndCanBeRetried) {
  original_context_.create_result = QNN_COMMON_ERROR_INVALID_ARGUMENT;
  original_context_.retrieve_result = QNN_COMMON_ERROR_INVALID_ARGUMENT;
  EXPECT_FALSE(emitter_.CreateGraph(&original_context_, "graph").IsOK());
  EXPECT_EQ(emitter_.Graph(), nullptr);
  EXPECT_EQ(emitter_.Context(), nullptr);
  EXPECT_EQ(original_context_.retrieve_calls, 1u);

  original_context_.create_result = QNN_SUCCESS;
  EXPECT_TRUE(emitter_.CreateGraph(&original_context_, "graph").IsOK());
  EXPECT_EQ(emitter_.Graph(), &original_graph_);
  EXPECT_EQ(emitter_.Context(), &original_context_);
}

TEST_F(QnnGraphEmitterTest, RetrievalReplacesHandlesForFinalizeExecuteAndSetConfig) {
  ASSERT_TRUE(emitter_.CreateGraph(&original_context_, "original").IsOK());
  ASSERT_TRUE(emitter_.RetrieveGraph(&replacement_context_, "replacement").IsOK());
  EXPECT_EQ(emitter_.Graph(), &replacement_graph_);
  EXPECT_EQ(emitter_.Context(), &replacement_context_);
  EXPECT_EQ(replacement_context_.name, "replacement");

  int profile_sentinel = 0;
  int signal_sentinel = 0;
  Qnn_ProfileHandle_t profile = &profile_sentinel;
  Qnn_SignalHandle_t signal = &signal_sentinel;
  ASSERT_TRUE(emitter_.Finalize(profile).IsOK());
  EXPECT_EQ(replacement_graph_.profile, profile);
  EXPECT_EQ(replacement_graph_.signal, nullptr);

  Qnn_Tensor_t inputs[2]{};
  Qnn_Tensor_t outputs[1]{};
  EXPECT_EQ(emitter_.Execute(inputs, 2, outputs, 1, profile, signal), QNN_SUCCESS);
  EXPECT_EQ(replacement_graph_.inputs, inputs);
  EXPECT_EQ(replacement_graph_.outputs, outputs);
  EXPECT_EQ(replacement_graph_.num_inputs, 2u);
  EXPECT_EQ(replacement_graph_.num_outputs, 1u);
  EXPECT_EQ(replacement_graph_.profile, profile);
  EXPECT_EQ(replacement_graph_.signal, signal);

  const QnnGraph_Config_t* configs[]{nullptr};
  EXPECT_TRUE(emitter_.SetConfig(configs).IsOK());
  EXPECT_EQ(replacement_graph_.configs, configs);
  EXPECT_EQ(replacement_graph_.finalize_calls, 1u);
  EXPECT_EQ(replacement_graph_.execute_calls, 1u);
  EXPECT_EQ(replacement_graph_.config_calls, 1u);
  EXPECT_EQ(original_graph_.finalize_calls, 0u);
  EXPECT_EQ(original_graph_.execute_calls, 0u);
  EXPECT_EQ(original_graph_.config_calls, 0u);
}

TEST_F(QnnGraphEmitterTest, FailedRetrievalDoesNotReplaceEitherHandle) {
  ASSERT_TRUE(emitter_.CreateGraph(&original_context_, "original").IsOK());
  replacement_context_.retrieve_result = QNN_COMMON_ERROR_INVALID_ARGUMENT;
  EXPECT_FALSE(emitter_.RetrieveGraph(&replacement_context_, "replacement").IsOK());
  EXPECT_EQ(emitter_.Graph(), &original_graph_);
  EXPECT_EQ(emitter_.Context(), &original_context_);
}

TEST_F(QnnGraphEmitterTest, SuccessfulRetrievalWithNullGraphDoesNotReplaceEitherHandle) {
  ASSERT_TRUE(emitter_.CreateGraph(&original_context_, "original").IsOK());
  replacement_context_.graph = nullptr;
  EXPECT_FALSE(emitter_.RetrieveGraph(&replacement_context_, "replacement").IsOK());
  EXPECT_EQ(emitter_.Graph(), &original_graph_);
  EXPECT_EQ(emitter_.Context(), &original_context_);
}

TEST_F(QnnGraphEmitterTest, ExecutePreservesSdkErrorsIncludingSsrCommunicationError) {
  ASSERT_TRUE(emitter_.RetrieveGraph(&original_context_, "graph").IsOK());
  for (Qnn_ErrorHandle_t error : {QNN_COMMON_ERROR_INVALID_ARGUMENT, QNN_COMMON_ERROR_SYSTEM_COMMUNICATION}) {
    original_graph_.execute_result = error;
    EXPECT_EQ(emitter_.Execute(nullptr, 0, nullptr, 0, nullptr, nullptr), error);
  }
}

TEST_F(QnnGraphEmitterTest, SetConfigRejectsUnavailableFunctionPointer) {
  ASSERT_TRUE(emitter_.RetrieveGraph(&original_context_, "graph").IsOK());
  qnn_interface_.graphSetConfig = nullptr;
  const auto status = emitter_.SetConfig(nullptr);
  EXPECT_FALSE(status.IsOK());
  EXPECT_NE(status.GetErrorMessage().find("Invalid function pointer"), std::string::npos);
  EXPECT_EQ(original_graph_.config_calls, 0u);
}

TEST_F(QnnGraphEmitterTest, SetConfigReportsSdkFailureWithRetrievedGraphName) {
  ASSERT_TRUE(emitter_.RetrieveGraph(&replacement_context_, "replacement").IsOK());
  replacement_graph_.config_result = QNN_COMMON_ERROR_INVALID_ARGUMENT;
  const auto status = emitter_.SetConfig(nullptr);
  EXPECT_FALSE(status.IsOK());
  EXPECT_NE(status.GetErrorMessage().find("replacement"), std::string::npos);
  EXPECT_NE(status.GetErrorMessage().find("stub QNN error"), std::string::npos);
  EXPECT_EQ(replacement_graph_.config_calls, 1u);
}

TEST_F(QnnGraphEmitterTest, FinalizeReportsSdkFailure) {
  ASSERT_TRUE(emitter_.RetrieveGraph(&original_context_, "graph").IsOK());
  original_graph_.finalize_result = QNN_COMMON_ERROR_INVALID_ARGUMENT;
  const auto status = emitter_.Finalize(nullptr);
  EXPECT_FALSE(status.IsOK());
  EXPECT_NE(status.GetErrorMessage().find("stub QNN error"), std::string::npos);
}

TEST_F(QnnGraphEmitterTest, UninitializedOperationsFailBeforeSdkDispatch) {
  EXPECT_FALSE(emitter_.Finalize(nullptr).IsOK());
  EXPECT_EQ(emitter_.Execute(nullptr, 0, nullptr, 0, nullptr, nullptr), QNN_COMMON_ERROR_INVALID_ARGUMENT);
  EXPECT_FALSE(emitter_.SetConfig(nullptr).IsOK());
  EXPECT_EQ(original_graph_.finalize_calls, 0u);
  EXPECT_EQ(original_graph_.execute_calls, 0u);
  EXPECT_EQ(original_graph_.config_calls, 0u);
}

}  // namespace
}  // namespace onnxruntime::test

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
