#include <iostream>
#include <thread>
#include <mutex>
#include <functional>
#include <condition_variable>
#include <atomic>

class Timer {
 public:
  enum class threadState {
    IDLE,     // Timer is created
    LAUNCH,   // Timer starts counting down
    CALLING,  // Callback function is called
    DEINIT    // Timer is deinit
  };
  // constructor
  Timer() = default;
  // destructor
  ~Timer();

  template <class T_Rep, class T_Period>
  bool remainingDuration(std::chrono::duration<T_Rep, T_Period>& duration) {
    std::unique_lock<std::mutex> lk(mtx);
    if (threadStatus == threadState::LAUNCH) {
      duration = std::chrono::duration_cast<std::chrono::duration<T_Rep, T_Period>>(endTime - std::chrono::steady_clock::now());
      return true;
    } else if (threadStatus == threadState::CALLING || threadStatus == threadState::IDLE) {
      duration = std::chrono::duration<T_Rep, T_Period>::zero();
      return true;
    } else {
      duration = std::chrono::duration<T_Rep, T_Period>::zero();
      return false;
    }
  }

  template <class T_Rep, class T_Period>
  bool launch(const std::chrono::duration<T_Rep, T_Period>& timeoutVal) {
    std::unique_lock<std::mutex> lk(mtx);
    if (threadStatus != threadState::IDLE) {
      return false;
    }
    endTime = std::chrono::steady_clock::now() + timeoutVal;
    threadStatus = threadState::LAUNCH;
    isTimerLaunched = true;
    kcv.notify_all();
    return true;
  }

  bool initialize(std::function<void(void*)> callbackFn, void* callbackArg);
  void deinitialize();
  void abortTimer();

 private:
  std::thread bkgThread;
  void bkgTimer();
  std::mutex mtx;
  std::condition_variable kcv;
  std::function<void(void*)> ktimeoutFn;
  void* ktimeoutArg{nullptr};
  std::atomic<threadState> threadStatus{threadState::DEINIT};
  std::chrono::time_point<std::chrono::steady_clock> endTime;
  std::atomic<bool> isTimerStopped = false;
  std::atomic<bool> isTimerDeinit = false;
  std::atomic<bool> isTimerLaunched = false;
};
