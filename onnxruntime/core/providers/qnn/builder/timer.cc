#include "timer.h"


void Timer::deinitialize () {
    std::unique_lock<std::mutex> lk(mtx);
    isTimerDeinit = true;
    kcv.notify_all();
    lk.unlock();
    if (bkgThread.joinable()) {
        bkgThread.join();
    }
}

Timer::~Timer() { this->deinitialize();}

void Timer::bkgTimer() {
    {
        std::unique_lock<std::mutex> lk(mtx);
        threadStatus = threadState::IDLE;
        kcv.notify_all();
    }
    while (true) {
        std::unique_lock<std::mutex> lk(mtx);

        if (threadStatus == threadState::IDLE) {
            kcv.wait(lk, [&]() {
                return isTimerLaunched || isTimerStopped || isTimerDeinit ;
            });
        }

        if (isTimerDeinit) {
            threadStatus = threadState::DEINIT;
            isTimerDeinit = false;
            return;
        }

        if (isTimerStopped) {
            threadStatus = threadState::IDLE;
            isTimerStopped = false;
        }

        if (threadStatus == threadState::LAUNCH) {
            bool isElapsed = !kcv.wait_until(lk, endTime, [&]() {
                return isTimerStopped || isTimerDeinit;
            });
            if (isElapsed) {
                threadStatus = threadState::CALLING;
                lk.unlock();
                ktimeoutFn(ktimeoutArg);
                lk.lock();
                threadStatus = threadState::IDLE;
            }
            isTimerLaunched = false;
        }
    }
}

bool Timer::initialize(std::function<void(void*)> callbackFn, void* callbackArg) {
    std::unique_lock<std::mutex> lk(mtx);
    ktimeoutArg = callbackArg;
    ktimeoutFn = callbackFn;
    bkgThread = std::thread(&Timer::bkgTimer, this);
    kcv.wait(lk, [&]{ return threadStatus == threadState::IDLE ; });
    return true;
}

void Timer::abortTimer() {
    std::unique_lock<std::mutex> lk(mtx);
    isTimerStopped = true;
    kcv.notify_all();
}
