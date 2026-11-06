// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "core/providers/qnn/shared_context.h"

namespace onnxruntime {
namespace test {

namespace {

qnn::QnnGraphContext MakeGraphContext(uint32_t soc_model, const std::string& graph_name, int32_t vtcm_size_in_mb = 0) {
  qnn::QnnGraphContext context;
  qnn::HtpGraphConfigs_t config;
  config.vtcm_size_in_mb = vtcm_size_in_mb;
  context.soc_targets = {{soc_model, QNN_HTP_DEVICE_ARCH_V73, config}};
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
  auto first_context = MakeGraphContext(60, "first", 4);
  qnn::HtpGraphConfigs_t first_target_config;
  first_target_config.vtcm_size_in_mb = 8;
  first_context.soc_targets.push_back({88, QNN_HTP_DEVICE_ARCH_V81, first_target_config});
  shared_context.AppendSharedQnnGraphContext(std::move(first_context));

  auto second_context = MakeGraphContext(60, "second", 12);
  qnn::HtpGraphConfigs_t second_target_config;
  second_target_config.vtcm_size_in_mb = 16;
  second_context.soc_targets.push_back({88, QNN_HTP_DEVICE_ARCH_V81, second_target_config});
  shared_context.AppendSharedQnnGraphContext(std::move(second_context));

  const auto contexts = shared_context.TakeSharedQnnGraphContexts();

  ASSERT_EQ(contexts.size(), 2u);
  for (const auto& context : contexts) {
    ASSERT_EQ(context.soc_targets.size(), 2u);
    EXPECT_EQ(context.soc_targets[0].htp_arch, QNN_HTP_DEVICE_ARCH_V73);
    EXPECT_EQ(context.soc_targets[0].soc_model, 60u);
    EXPECT_EQ(context.soc_targets[1].htp_arch, QNN_HTP_DEVICE_ARCH_V81);
    EXPECT_EQ(context.soc_targets[1].soc_model, 88u);
  }
  EXPECT_EQ(contexts[0].soc_targets[0].htp_graph_cfg.vtcm_size_in_mb, 4);
  EXPECT_EQ(contexts[0].soc_targets[1].htp_graph_cfg.vtcm_size_in_mb, 8);
  EXPECT_EQ(contexts[1].soc_targets[0].htp_graph_cfg.vtcm_size_in_mb, 12);
  EXPECT_EQ(contexts[1].soc_targets[1].htp_graph_cfg.vtcm_size_in_mb, 16);
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
