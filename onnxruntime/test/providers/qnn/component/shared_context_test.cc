// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstdint>
#include <string>
#include <vector>

#include "core/providers/qnn/shared_context.h"

namespace onnxruntime {
namespace test {

namespace {

qnn::QnnGraphContext MakeGraphContext(uint32_t soc_model, const std::string& graph_name) {
  qnn::QnnGraphContext context;
  context.htp_arch_per_soc = {QNN_HTP_DEVICE_ARCH_V73};
  context.soc_model_per_soc = {soc_model};
  context.qnn_graph_wrappers.emplace_back().graph_name = graph_name;
  return context;
}

class QnnUnit_SharedContextTest : public ::testing::Test {
 protected:
  void SetUp() override {
    SharedContext::GetInstance().ResetSharedQnnGraphContexts();
  }

  void TearDown() override {
    SharedContext::GetInstance().ResetSharedQnnGraphContexts();
  }
};

TEST_F(QnnUnit_SharedContextTest, TakeSharedQnnGraphContexts_MultipleAppends_ConsumesWholeBatch) {
  auto& shared_context = SharedContext::GetInstance();
  shared_context.AppendSharedQnnGraphContext(MakeGraphContext(60, "first"));
  shared_context.AppendSharedQnnGraphContext(MakeGraphContext(60, "second"));

  const auto contexts = shared_context.TakeSharedQnnGraphContexts();

  ASSERT_EQ(contexts.size(), 2u);
  EXPECT_EQ(contexts[0].htp_arch_per_soc, std::vector<QnnHtpDevice_Arch_t>{QNN_HTP_DEVICE_ARCH_V73});
  EXPECT_EQ(contexts[0].soc_model_per_soc, std::vector<uint32_t>{60});
  ASSERT_EQ(contexts[0].qnn_graph_wrappers.size(), 1u);
  ASSERT_EQ(contexts[1].qnn_graph_wrappers.size(), 1u);
  EXPECT_EQ(contexts[0].qnn_graph_wrappers[0].graph_name, "first");
  EXPECT_EQ(contexts[1].qnn_graph_wrappers[0].graph_name, "second");
  EXPECT_TRUE(shared_context.TakeSharedQnnGraphContexts().empty());
}

TEST_F(QnnUnit_SharedContextTest, TakeSharedQnnGraphContexts_SubsequentAppend_StartsIndependentBatch) {
  auto& shared_context = SharedContext::GetInstance();
  shared_context.AppendSharedQnnGraphContext(MakeGraphContext(60, "first_batch"));
  const auto first_batch = shared_context.TakeSharedQnnGraphContexts();

  shared_context.AppendSharedQnnGraphContext(MakeGraphContext(60, "second_batch"));
  const auto second_batch = shared_context.TakeSharedQnnGraphContexts();

  ASSERT_EQ(first_batch.size(), 1u);
  ASSERT_EQ(second_batch.size(), 1u);
  ASSERT_EQ(first_batch[0].qnn_graph_wrappers.size(), 1u);
  ASSERT_EQ(second_batch[0].qnn_graph_wrappers.size(), 1u);
  EXPECT_EQ(first_batch[0].qnn_graph_wrappers[0].graph_name, "first_batch");
  EXPECT_EQ(second_batch[0].qnn_graph_wrappers[0].graph_name, "second_batch");
}

TEST_F(QnnUnit_SharedContextTest, ResetSharedQnnGraphContexts_PendingBatch_PreservesTakenContexts) {
  auto& shared_context = SharedContext::GetInstance();
  shared_context.AppendSharedQnnGraphContext(MakeGraphContext(60, "taken"));
  const auto contexts = shared_context.TakeSharedQnnGraphContexts();

  shared_context.AppendSharedQnnGraphContext(MakeGraphContext(60, "discarded"));
  shared_context.ResetSharedQnnGraphContexts();

  EXPECT_TRUE(shared_context.TakeSharedQnnGraphContexts().empty());
  ASSERT_EQ(contexts.size(), 1u);
  ASSERT_EQ(contexts[0].qnn_graph_wrappers.size(), 1u);
  EXPECT_EQ(contexts[0].qnn_graph_wrappers[0].graph_name, "taken");

  shared_context.AppendSharedQnnGraphContext(MakeGraphContext(60, "after_reset"));
  const auto next_batch = shared_context.TakeSharedQnnGraphContexts();
  ASSERT_EQ(next_batch.size(), 1u);
  ASSERT_EQ(next_batch[0].qnn_graph_wrappers.size(), 1u);
  EXPECT_EQ(next_batch[0].qnn_graph_wrappers[0].graph_name, "after_reset");
}

}  // namespace

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
