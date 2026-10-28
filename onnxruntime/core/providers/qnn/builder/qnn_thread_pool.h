// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#if defined(_WIN32)
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace onnxruntime {
namespace qnn {
namespace thread {

class QnnJobThreadPool {
 private:
  // Class that contains main job thread
  // - Checks the thread pool job queue for jobs
  // - Run any available jobs. Wait for new submission to queue otherwise
  class QnnJobThread {
   public:
    QnnJobThread(uint8_t thread_num, QnnJobThreadPool* thread_pool_ptr);
    QnnJobThread(QnnJobThread&&) = delete;
    QnnJobThread& operator=(QnnJobThread&&) = delete;

    ~QnnJobThread();

    // Starts the main thread/loop
    // 1. Checks if there is a job in the queue
    // 2a. If a job exists, run it and notify the pool when it finishes
    // 2b. If a job does not exist, then wait for a new submission
    // 3. Continue to do steps 1, 2a, and 2b until stopped
    void Start();

    // Stops main thread/loop
    void Stop();

   private:
    bool IsStopped() const {
      std::unique_lock<std::mutex> lock(thread_state_mutex_);
      return thread_stopped_;
    }

    const uint8_t thread_num_;
    QnnJobThreadPool* const tp_;

    std::function<bool()> exit_predicate_;

    mutable std::mutex thread_state_mutex_;
    bool thread_stopped_ = true;

    std::unique_ptr<std::thread> thread_;
  };

 public:
  QnnJobThreadPool(uint8_t max_num_threads);

  ~QnnJobThreadPool();

  // Creates and starts all job threads
  void Start() const;

  // Stops all job threads
  void Stop() const;

  // Waits for job queue to empty out and all active jobs to finish running
  void WaitForQueuedJobsToFinish();

  // Submits job to queue but job may sit in queue until threadpool starts
  // Notifies one waiting job thread
  void SubmitJob(std::function<void()> job);

 private:
  // Function to be called by job thread waiting for a new job to be available
  // Check interval is currently 200ms
  // thread_num - job thread identifier for debugging purposes
  // exit_predicate - a function that returns true when this function
  //                  must exit regardless of job availability
  // Returns when a job is available or if the exit predicate is met
  void WaitForJobQueueUpdate(const uint8_t thread_num, const std::function<bool()>& exit_predicate);

  // Returns a job if one is available, nullptr otherwise
  std::function<void()> GetJobFromQueueIfExists(const uint8_t thread_num);

  // Decrements the number of jobs owned by workers and wakes completion waiters.
  void NotifyJobFinished();

  bool IsRunning() const {
    std::unique_lock<std::mutex> lock(state_mutex_);
    return running_;
  }

  std::condition_variable job_submitted_cv_;
  std::condition_variable jobs_finished_cv_;

  std::mutex queue_mutex_;
  std::queue<std::function<void()>> job_queue_;
  size_t in_flight_jobs_ = 0;

  std::vector<std::unique_ptr<QnnJobThread>> thread_pool_;

  mutable std::mutex state_mutex_;
  mutable bool running_;
};

}  // namespace thread
}  // namespace qnn
}  // namespace onnxruntime
#endif
