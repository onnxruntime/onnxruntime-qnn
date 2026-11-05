// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// Licensed under the MIT License.

#pragma once

#include <chrono>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace onnxruntime::qnn::pipeline_metrics {

using Clock = std::chrono::steady_clock;
using Field = std::pair<std::string_view, std::string_view>;

inline const char* OutputPath() {
  return std::getenv("QNN_PIPELINE_METRICS_PATH");
}

inline bool Enabled() {
  const char* path = OutputPath();
  return path != nullptr && path[0] != '\0';
}

inline std::string EscapeJson(std::string_view value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (const char ch : value) {
    switch (ch) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped += ch;
        break;
    }
  }
  return escaped;
}

inline int ProcessId() {
#ifdef _WIN32
  return _getpid();
#else
  return static_cast<int>(getpid());
#endif
}

inline void AppendLine(const std::string& line) {
  const char* path = OutputPath();
  if (path == nullptr || path[0] == '\0') {
    return;
  }

#ifdef _WIN32
  const int fd = _open(path, _O_CREAT | _O_APPEND | _O_WRONLY | _O_BINARY, _S_IREAD | _S_IWRITE);
  if (fd >= 0) {
    _write(fd, line.data(), static_cast<unsigned int>(line.size()));
    _close(fd);
  }
#else
  const int fd = open(path, O_CREAT | O_APPEND | O_WRONLY, 0644);
  if (fd >= 0) {
    const ssize_t ignored = write(fd, line.data(), line.size());
    (void)ignored;
    close(fd);
  }
#endif
}

inline void Record(std::string_view event,
                   Clock::duration duration = Clock::duration::zero(),
                   std::initializer_list<Field> fields = {}) noexcept {
  try {
    if (!Enabled()) {
      return;
    }

    const auto timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  Clock::now().time_since_epoch())
                                  .count();
    const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();

    std::ostringstream output;
    output << "{\"timestamp_ns\":" << timestamp_ns
           << ",\"pid\":" << ProcessId()
           << ",\"thread\":\"" << std::hash<std::thread::id>{}(std::this_thread::get_id())
           << "\",\"event\":\"" << EscapeJson(event)
           << "\",\"duration_us\":" << duration_us;
    for (const auto& [key, value] : fields) {
      output << ",\"" << EscapeJson(key) << "\":\"" << EscapeJson(value) << '"';
    }
    output << "}\n";
    AppendLine(output.str());
  } catch (...) {
    // Metrics are diagnostic-only and must never affect test behavior.
  }
}

class Scope {
 public:
  explicit Scope(std::string_view event)
      : event_(event), enabled_(Enabled()), start_(enabled_ ? Clock::now() : Clock::time_point{}) {
  }

  ~Scope() noexcept {
    try {
      if (enabled_) {
        Record(event_, Clock::now() - start_);
      }
    } catch (...) {
      // Metrics are diagnostic-only and must never affect test behavior.
    }
  }

  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;

 private:
  std::string event_;
  bool enabled_;
  Clock::time_point start_;
};

}  // namespace onnxruntime::qnn::pipeline_metrics
