// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#if defined(_WIN32) && !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <atomic>
#include <cstddef>
#include <thread>

#include "core/providers/qnn/builder/qnn_thread_pool.h"

namespace onnxruntime {
namespace test {

TEST(QnnUnit_ThreadPoolTest, WaitForQueuedJobsToFinishWaitsForEveryJob) {
  constexpr size_t kNumJobs = 512;
  std::atomic<size_t> completed_jobs{0};

  qnn::thread::QnnJobThreadPool thread_pool(4);
  thread_pool.Start();

  for (size_t i = 0; i < kNumJobs; ++i) {
    thread_pool.SubmitJob([i, &completed_jobs]() {
      if ((i % 2) == 0) {
        std::this_thread::yield();
      }
      completed_jobs.fetch_add(1, std::memory_order_relaxed);
    });
  }

  thread_pool.WaitForQueuedJobsToFinish();
  EXPECT_EQ(completed_jobs.load(std::memory_order_relaxed), kNumJobs);
  thread_pool.Stop();
}

}  // namespace test
}  // namespace onnxruntime

#endif  // defined(_WIN32) && !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
