// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if defined(_WIN32)
#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_thread_pool.h"

namespace onnxruntime {
namespace qnn {
namespace thread {

QnnJobThreadPool::QnnJobThread::QnnJobThread(uint8_t thread_num, QnnJobThreadPool* thread_pool_ptr)
    : thread_num_(thread_num), tp_(thread_pool_ptr) {
  ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(),
              ORT_LOGGING_LEVEL_VERBOSE,
              ("QnnJobThread: Thread " + std::to_string(thread_num_) + " created").c_str());

  // Used to exit out of QnnJobThreadPool::WaitForJobQueueUpdate() regardless of job queue status
  exit_predicate_ = [this]() {
    return IsStopped();
  };
}

QnnJobThreadPool::QnnJobThread::~QnnJobThread() {
  try {
    Stop();
  } catch (const std::exception& e) {
    ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(),
                ORT_LOGGING_LEVEL_ERROR,
                ("QnnJobThread: Thread " + std::to_string(thread_num_) + ": Error on destruction: " + std::string(e.what())).c_str());
  }
}

void QnnJobThreadPool::QnnJobThread::Start() {
  // Only created thread if no thread exists and the current state is stopped
  {
    std::unique_lock<std::mutex> lock(thread_state_mutex_);
    if (!thread_ && thread_stopped_) {
      thread_stopped_ = false;
    } else {
      return;
    }
  }

  thread_ = std::make_unique<std::thread>([this]() {
    do {
      auto job = tp_->GetJobFromQueueIfExists(thread_num_);
      if (job) {
        SetActive();
        tp_->NotifyJobStarted();

        job();

        SetInactive();
      } else {
        tp_->WaitForJobQueueUpdate(thread_num_, exit_predicate_);
      }
    } while (!IsStopped());
  });

  ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(),
              ORT_LOGGING_LEVEL_VERBOSE,
              ("QnnJobThread: Thread " + std::to_string(thread_num_) + " started").c_str());
}

void QnnJobThreadPool::QnnJobThread::Stop() {
  // Only stop if the thread exists and current state is running
  {
    std::unique_lock<std::mutex> lock(thread_state_mutex_);
    if (thread_ && !thread_stopped_) {
      ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(),
                  ORT_LOGGING_LEVEL_VERBOSE,
                  ("QnnJobThread: Thread " + std::to_string(thread_num_) + " stopping").c_str());
      thread_stopped_ = true;
    } else {
      return;
    }
  }

  thread_->join();
  thread_.reset();

  ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(),
              ORT_LOGGING_LEVEL_VERBOSE,
              ("QnnJobThread: Thread " + std::to_string(thread_num_) + " stopped").c_str());
}

void QnnJobThreadPool::QnnJobThread::WaitUntilInactive() {
  std::unique_lock<std::mutex> lock(thread_activity_mutex_);
  if (thread_active_) {
    thread_activity_change_cv_.wait(lock, [this]() {
      return !thread_active_;
    });
  }
}

QnnJobThreadPool::QnnJobThreadPool(uint8_t max_num_threads)
    : thread_pool_(max_num_threads), running_(false) {
  for (uint8_t thread_num = 0; thread_num < max_num_threads; thread_num++) {
    thread_pool_[thread_num] = std::make_unique<QnnJobThread>(thread_num, this);
  }
}

QnnJobThreadPool::~QnnJobThreadPool() {
  try {
    Stop();
  } catch (const std::exception& e) {
    ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(),
                ORT_LOGGING_LEVEL_ERROR,
                ("QnnJobThreadPool: Error on destruction: " + std::string(e.what())).c_str());
  }
}

void QnnJobThreadPool::Start() const {
  if (IsRunning()) {
    return;
  }

  ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(), ORT_LOGGING_LEVEL_VERBOSE, "QnnJobThreadPool: Start");
  std::unique_lock<std::mutex> s_lock(state_mutex_);
  running_ = true;

  for (auto& thread : thread_pool_) {
    thread->Start();
  }
}

void QnnJobThreadPool::Stop() const {
  if (!IsRunning()) {
    return;
  }

  ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(), ORT_LOGGING_LEVEL_VERBOSE, "QnnJobThreadPool: Stop");
  std::unique_lock<std::mutex> s_lock(state_mutex_);
  running_ = false;

  for (auto& thread : thread_pool_) {
    thread->Stop();
  }
}

void QnnJobThreadPool::WaitForQueuedJobsToFinish() {
  ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(),
              ORT_LOGGING_LEVEL_VERBOSE,
              "QnnJobThreadPool: Waiting for all jobs to finish");

  // Block all newly submitted jobs from entering the queue
  std::unique_lock<std::mutex> lock(queue_mutex_);
  // Only wait until queue is empty if thread pool has not been stopped
  if (IsRunning() && !job_queue_.empty()) {
    job_started_cv_.wait(lock, [this]() {
      return job_queue_.empty();
    });
  }

  for (auto& thread : thread_pool_) {
    thread->WaitUntilInactive();
  }

  ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(), ORT_LOGGING_LEVEL_VERBOSE, "QnnJobThreadPool: Done waiting on all jobs");
}

void QnnJobThreadPool::SubmitJob(std::function<void()> job) {
  ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(), ORT_LOGGING_LEVEL_VERBOSE, "QnnJobThreadPool: Job submitted");

  std::unique_lock<std::mutex> lock(queue_mutex_);
  job_queue_.push(std::move(job));
  ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(),
              ORT_LOGGING_LEVEL_VERBOSE,
              ("QnnJobThreadPool: Job pushed to queue, current size: " + std::to_string(job_queue_.size())).c_str());
  job_submitted_cv_.notify_one();
}

void QnnJobThreadPool::WaitForJobQueueUpdate(const uint8_t thread_num, const std::function<bool()>& exit_predicate) {
  std::unique_lock<std::mutex> lock(queue_mutex_);
  ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(),
              ORT_LOGGING_LEVEL_VERBOSE,
              ("QnnJobThreadPool: Thread " + std::to_string(thread_num) + " waiting for a job").c_str());
  job_submitted_cv_.wait_for(lock, std::chrono::milliseconds(200), [this, &exit_predicate] {
    return !job_queue_.empty() || exit_predicate();
  });
}

std::function<void()> QnnJobThreadPool::GetJobFromQueueIfExists(const uint8_t thread_num) {
  std::unique_lock<std::mutex> lock(queue_mutex_);
  ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(),
              ORT_LOGGING_LEVEL_VERBOSE,
              ("QnnJobThreadPool: Thread " + std::to_string(thread_num) + " checking for job, queue size: " + std::to_string(job_queue_.size())).c_str());
  if (!job_queue_.empty()) {
    ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(),
                ORT_LOGGING_LEVEL_VERBOSE,
                ("QnnJobThreadPool: Thread " + std::to_string(thread_num) + " received a job").c_str());
    auto job = job_queue_.front();
    job_queue_.pop();
    return job;
  }

  return nullptr;
}

}  // namespace thread
}  // namespace qnn
}  // namespace onnxruntime
#endif