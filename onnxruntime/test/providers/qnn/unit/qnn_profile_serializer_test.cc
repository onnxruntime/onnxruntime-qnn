// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Function-level unit tests for qnn_profile_serializer.cc — Serializer helpers
// that convert QNN profile events into CSV rows and (optionally) QNN system
// profile log records.
//
// These tests do not require a QNN backend or hardware. All QNN system profile
// C entry points are supplied through captureless lambdas installed on a
// caller-provided QNN_SYSTEM_INTERFACE_VER_TYPE, and file output is written to
// unique paths under std::filesystem::temp_directory_path().

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <unistd.h>

#include "QnnInterface.h"
#include "QnnProfile.h"
#include "QnnTypes.h"
#include "System/QnnSystemInterface.h"
#include "System/QnnSystemProfile.h"

#include "core/providers/qnn/builder/op_tracing/qnn_op_tracing_types.h"
#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_profile_serializer.h"

namespace onnxruntime {
namespace test {

using onnxruntime::qnn::profile::ProfilingInfo;
using onnxruntime::qnn::profile::Serializer;

// ---------------------------------------------------------------------------
// Test fixture and helpers
// ---------------------------------------------------------------------------

namespace {

// State touched by the stubbed QNN system profile function pointers. Every
// stub reaches this state via the thread_local pointer below (captureless
// lambdas cannot see test locals directly). See qnn-ep-coverage-guide.md
// §OrtApi stub infrastructure for the pattern rationale.
struct ProfileSerializerStubState {
  // Return values injected into each stubbed call.
  Qnn_ErrorHandle_t create_ret = QNN_SYSTEM_PROFILE_NO_ERROR;
  Qnn_ErrorHandle_t serialize_ret = QNN_SYSTEM_PROFILE_NO_ERROR;
  Qnn_ErrorHandle_t free_ret = QNN_SYSTEM_PROFILE_NO_ERROR;

  // Observed data from the last call.
  int create_calls = 0;
  int serialize_calls = 0;
  int free_calls = 0;
  QnnSystemProfile_MethodType_t last_method_type = QNN_SYSTEM_PROFILE_METHOD_TYPE_NONE;
  uint64_t last_start_time = 0;
  uint64_t last_stop_time = 0;
  uint32_t last_num_events = 0;
  std::string last_app_name;
  std::string last_backend_version;
  std::string last_file_name;
  std::string last_file_dir;
};

thread_local ProfileSerializerStubState* g_profile_stub_state = nullptr;

struct UseProfileStubs {
  ProfileSerializerStubState* prev = nullptr;
  explicit UseProfileStubs(ProfileSerializerStubState& s) noexcept {
    prev = g_profile_stub_state;
    g_profile_stub_state = &s;
  }
  ~UseProfileStubs() { g_profile_stub_state = prev; }

  UseProfileStubs(const UseProfileStubs&) = delete;
  UseProfileStubs& operator=(const UseProfileStubs&) = delete;
};

// Returns a QNN_SYSTEM_INTERFACE_VER_TYPE with the three profile-serialization
// entry points wired to captureless lambdas that read/write g_profile_stub_state.
// Pass this to Serializer's constructor to unit-test SerializeEventsToQnnLog
// without needing a real libQnnSystem.so.
QNN_SYSTEM_INTERFACE_VER_TYPE MakeStubSystemInterface() {
  QNN_SYSTEM_INTERFACE_VER_TYPE sys = QNN_SYSTEM_INTERFACE_VER_TYPE_INIT;

  sys.systemProfileCreateSerializationTarget =
      [](QnnSystemProfile_SerializationTarget_t target,
         QnnSystemProfile_SerializationTargetConfig_t* configs,
         uint32_t num_configs,
         QnnSystemProfile_SerializationTargetHandle_t* out) -> Qnn_ErrorHandle_t {
    if (!g_profile_stub_state) return QNN_SYSTEM_PROFILE_ERROR_INVALID_ARGUMENT;
    g_profile_stub_state->create_calls++;
    if (target.type == QNN_SYSTEM_PROFILE_SERIALIZATION_TARGET_FILE) {
      g_profile_stub_state->last_file_name =
          target.file.fileName ? target.file.fileName : "";
      g_profile_stub_state->last_file_dir =
          target.file.fileDirectory ? target.file.fileDirectory : "";
    }
    if (configs && num_configs > 0 &&
        configs[0].type == QNN_SYSTEM_PROFILE_SERIALIZATION_TARGET_CONFIG_SERIALIZATION_HEADER) {
      if (configs[0].serializationHeader.appName) {
        g_profile_stub_state->last_app_name = configs[0].serializationHeader.appName;
      }
      if (configs[0].serializationHeader.backendVersion) {
        g_profile_stub_state->last_backend_version = configs[0].serializationHeader.backendVersion;
      }
    }
    // Give downstream code a non-null handle to reason about.
    static int kCookie = 0xCAFE;
    if (out) *out = &kCookie;
    return g_profile_stub_state->create_ret;
  };

  sys.systemProfileSerializeEventData =
      [](QnnSystemProfile_SerializationTargetHandle_t /*handle*/,
         const QnnSystemProfile_ProfileData_t** event_data,
         uint32_t num_events) -> Qnn_ErrorHandle_t {
    if (!g_profile_stub_state) return QNN_SYSTEM_PROFILE_ERROR_INVALID_ARGUMENT;
    g_profile_stub_state->serialize_calls++;
    if (event_data && num_events > 0 && event_data[0]) {
      g_profile_stub_state->last_method_type = event_data[0]->v1.header.methodType;
      g_profile_stub_state->last_start_time = event_data[0]->v1.header.startTime;
      g_profile_stub_state->last_stop_time = event_data[0]->v1.header.stopTime;
      g_profile_stub_state->last_num_events = event_data[0]->v1.numProfilingEvents;
    }
    return g_profile_stub_state->serialize_ret;
  };

  sys.systemProfileFreeSerializationTarget =
      [](QnnSystemProfile_SerializationTargetHandle_t /*handle*/) -> Qnn_ErrorHandle_t {
    if (!g_profile_stub_state) return QNN_SYSTEM_PROFILE_ERROR_INVALID_ARGUMENT;
    g_profile_stub_state->free_calls++;
    return g_profile_stub_state->free_ret;
  };

  return sys;
}

// Generates a unique CSV filepath under the system temp directory so parallel
// or sequential test runs never share a file.
std::string UniqueCsvPath(const std::string& tag) {
  static int counter = 0;
  auto base = std::filesystem::temp_directory_path();
  std::ostringstream ss;
  ss << "qnn_profile_serializer_ut_" << tag << "_" << ::getpid() << "_" << ++counter << ".csv";
  return (base / ss.str()).string();
}

// Reads a file into a std::string. Returns empty string if the file is missing.
std::string ReadFile(const std::string& path) {
  std::ifstream in(path);
  if (!in.is_open()) return {};
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

QnnProfile_EventData_t MakeEvent(QnnProfile_EventType_t type,
                                 QnnProfile_EventUnit_t unit,
                                 QnnProfile_EventValue_t value,
                                 const char* identifier) {
  QnnProfile_EventData_t e = QNN_PROFILE_EVENT_DATA_INIT;
  e.type = type;
  e.unit = unit;
  e.value = value;
  e.identifier = identifier;
  return e;
}

}  // namespace

class QnnUnit_ProfileSerializerTest : public ::testing::Test {
 protected:
  void SetUp() override { g_profile_stub_state = nullptr; }
  void TearDown() override {
    for (const auto& p : temp_files_to_remove_) {
      std::error_code ec;
      std::filesystem::remove(p, ec);
    }
    g_profile_stub_state = nullptr;
  }

  std::string NewCsvPath(const std::string& tag) {
    auto p = UniqueCsvPath(tag);
    temp_files_to_remove_.push_back(p);
    return p;
  }

 private:
  std::vector<std::string> temp_files_to_remove_;
};

// ===========================================================================
// Constructor
// ===========================================================================

TEST_F(QnnUnit_ProfileSerializerTest, Constructor_AbsolutePath_UsesParentDirectory) {
  // With an absolute path, the constructor takes the parent directory verbatim
  // (does not prepend current_path()). No public getter exposes output_directory_
  // directly, but SerializeEventsToQnnLog forwards it to the stub, so we can
  // inspect it there.
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("abs");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;
  info.graph_name = "g";

  ProfileSerializerStubState state;
  UseProfileStubs use(state);

  auto sys = MakeStubSystemInterface();
  Serializer s(info, sys, /*tracelogging_provider_ep_enabled=*/false);
  ASSERT_TRUE(s.SerializeEventsToQnnLog().IsOK());

  // Filename was derived by replacing ".csv" with "_qnn.log".
  std::filesystem::path csv_path(info.csv_output_filepath);
  auto stem = csv_path.stem().string();
  EXPECT_EQ(state.last_file_name, stem + "_qnn.log");
  EXPECT_EQ(state.last_file_dir, csv_path.parent_path().string());
}

TEST_F(QnnUnit_ProfileSerializerTest, Constructor_RelativePath_PrependsCurrentPath) {
  // A relative path with no root causes the constructor to use current_path()
  // as the output directory prefix.
  ProfilingInfo info;
  info.csv_output_filepath = "profile.csv";  // no root, no parent
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;
  info.graph_name = "g";

  ProfileSerializerStubState state;
  UseProfileStubs use(state);

  auto sys = MakeStubSystemInterface();
  Serializer s(info, sys, false);
  ASSERT_TRUE(s.SerializeEventsToQnnLog().IsOK());

  EXPECT_EQ(state.last_file_name, "profile_qnn.log");
  // path("foo") / path("") may append a preferred separator on some libc++
  // implementations, so compare with equivalence rather than exact string.
  const std::string cur = std::filesystem::current_path().string();
  ASSERT_FALSE(state.last_file_dir.empty());
  EXPECT_EQ(state.last_file_dir.compare(0, cur.size(), cur), 0)
      << "expected prefix=" << cur << " got=" << state.last_file_dir;
}

// ===========================================================================
// InitCsvFile
// ===========================================================================

TEST_F(QnnUnit_ProfileSerializerTest, InitCsvFile_NewFileNoTraceLookup_WritesBaseHeader) {
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("init_new");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;

  {
    auto sys = MakeStubSystemInterface();
    Serializer s(info, sys, false);
    ASSERT_TRUE(s.InitCsvFile().IsOK());
  }  // Serializer destruction flushes and closes the ofstream.
  auto contents = ReadFile(info.csv_output_filepath);
  EXPECT_NE(contents.find("Msg Timestamp,Message,Time"), std::string::npos);
  EXPECT_EQ(contents.find("ONNX Source Ops"), std::string::npos);
}

TEST_F(QnnUnit_ProfileSerializerTest, InitCsvFile_NewFileWithTraceLookup_WritesExtendedHeader) {
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("init_new_trace");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;
  onnxruntime::qnn::OpTraceLookup trace_lookup;
  info.op_trace_lookup = &trace_lookup;

  {
    auto sys = MakeStubSystemInterface();
    Serializer s(info, sys, false);
    ASSERT_TRUE(s.InitCsvFile().IsOK());
  }
  auto contents = ReadFile(info.csv_output_filepath);
  EXPECT_NE(contents.find("ONNX Source Ops"), std::string::npos);
}

TEST_F(QnnUnit_ProfileSerializerTest, InitCsvFile_ExistingFile_DoesNotRewriteHeader) {
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("init_existing");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;

  // Pre-create the file with a marker line so we can verify append semantics.
  {
    std::ofstream f(info.csv_output_filepath);
    f << "existing-marker\n";
  }
  {
    auto sys = MakeStubSystemInterface();
    Serializer s(info, sys, false);
    ASSERT_TRUE(s.InitCsvFile().IsOK());
  }
  auto contents = ReadFile(info.csv_output_filepath);
  // Marker still present, and header line was NOT prepended.
  EXPECT_NE(contents.find("existing-marker"), std::string::npos);
  EXPECT_EQ(contents.find("Msg Timestamp"), std::string::npos);
}

TEST_F(QnnUnit_ProfileSerializerTest, InitCsvFile_UnopenableDirectory_ReturnsError) {
  // Path with a non-existent parent directory. ofstream cannot open it,
  // so InitCsvFile should return an error.
  ProfilingInfo info;
  info.csv_output_filepath = "/nonexistent_dir_xy_1234567/profile.csv";
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;

  auto sys = MakeStubSystemInterface();
  Serializer s(info, sys, false);
  EXPECT_FALSE(s.InitCsvFile().IsOK());
}

// ===========================================================================
// ProcessEvent
// ===========================================================================

TEST_F(QnnUnit_ProfileSerializerTest, ProcessEvent_WithOutFile_WritesRow) {
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("proc_row");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;

  {
    auto sys = MakeStubSystemInterface();
    Serializer s(info, sys, false);
    ASSERT_TRUE(s.InitCsvFile().IsOK());
    auto e = MakeEvent(QNN_PROFILE_EVENTTYPE_EXECUTE, QNN_PROFILE_EVENTUNIT_MICROSEC, 42, "some_id");
    ASSERT_TRUE(s.ProcessEvent(1, "EVENT", e).IsOK());
  }
  auto contents = ReadFile(info.csv_output_filepath);
  // EXECUTE + US + BACKEND + EVENT + some_id + value 42 all appear in the row.
  EXPECT_NE(contents.find("EXECUTE"), std::string::npos);
  EXPECT_NE(contents.find("US"), std::string::npos);
  EXPECT_NE(contents.find("BACKEND"), std::string::npos);
  EXPECT_NE(contents.find("EVENT"), std::string::npos);
  EXPECT_NE(contents.find("some_id"), std::string::npos);
  EXPECT_NE(contents.find("42"), std::string::npos);
}

TEST_F(QnnUnit_ProfileSerializerTest, ProcessEvent_NullIdentifier_WritesNullPlaceholder) {
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("proc_null_id");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;

  {
    auto sys = MakeStubSystemInterface();
    Serializer s(info, sys, false);
    ASSERT_TRUE(s.InitCsvFile().IsOK());
    auto e = MakeEvent(QNN_PROFILE_EVENTTYPE_NODE, QNN_PROFILE_EVENTUNIT_CYCLES, 0, nullptr);
    ASSERT_TRUE(s.ProcessEvent(1, "EVENT", e).IsOK());
  }
  auto contents = ReadFile(info.csv_output_filepath);
  EXPECT_NE(contents.find(",NULL"), std::string::npos);
  EXPECT_NE(contents.find("NODE"), std::string::npos);
  EXPECT_NE(contents.find("CYCLES"), std::string::npos);
}

TEST_F(QnnUnit_ProfileSerializerTest, ProcessEvent_NoOutFile_ReturnsOkWithoutWriting) {
  // Skip InitCsvFile so outfile_ remains closed; only the QNN system profile
  // side of ProcessEvent runs (still records the event in AddEvent).
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("proc_no_outfile");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;

  auto sys = MakeStubSystemInterface();
  Serializer s(info, sys, false);

  auto e = MakeEvent(QNN_PROFILE_EVENTTYPE_INIT, QNN_PROFILE_EVENTUNIT_MICROSEC, 1, "id");
  EXPECT_TRUE(s.ProcessEvent(7, "EVENT", e).IsOK());
  EXPECT_NE(s.GetSystemEventPointer(7), nullptr);

  auto contents = ReadFile(info.csv_output_filepath);
  EXPECT_TRUE(contents.empty());
}

TEST_F(QnnUnit_ProfileSerializerTest, ProcessEvent_AllEventTypes_MapsToExpectedString) {
  // Exercises every arm of GetEventTypeString, including the two default arms:
  // event_type > QNN_PROFILE_EVENTTYPE_BACKEND -> "BACKEND"
  // event_type <= BACKEND but unmatched      -> "UNKNOWN".
  struct Case {
    QnnProfile_EventType_t type;
    const char* expected;
  };
  const Case cases[] = {
      {QNN_PROFILE_EVENTTYPE_INIT, "INIT"},
      {QNN_PROFILE_EVENTTYPE_FINALIZE, "FINALIZE"},
      {QNN_PROFILE_EVENTTYPE_EXECUTE, "EXECUTE"},
      {QNN_PROFILE_EVENTTYPE_NODE, "NODE"},
      {QNN_PROFILE_EVENTTYPE_EXECUTE_QUEUE_WAIT, "EXECUTE QUEUE WAIT"},
      {QNN_PROFILE_EVENTTYPE_EXECUTE_PREPROCESS, "EXECUTE PREPROCESS"},
      {QNN_PROFILE_EVENTTYPE_EXECUTE_DEVICE, "EXECUTE DEVICE"},
      {QNN_PROFILE_EVENTTYPE_EXECUTE_POSTPROCESS, "EXECUTE POSTPROCESS"},
      {QNN_PROFILE_EVENTTYPE_DEINIT, "DE-INIT"},
      {QNN_PROFILE_EVENTTYPE_BACKEND, "BACKEND"},
      {QNN_PROFILE_EVENTTYPE_BACKEND + 1, "BACKEND"},  // > BACKEND default arm
      {700u, "UNKNOWN"},                               // < BACKEND and unmatched
  };

  for (const auto& c : cases) {
    ProfilingInfo info;
    info.csv_output_filepath = NewCsvPath("proc_type");
    info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;
    {
      auto sys = MakeStubSystemInterface();
      Serializer s(info, sys, false);
      ASSERT_TRUE(s.InitCsvFile().IsOK());
      auto e = MakeEvent(c.type, QNN_PROFILE_EVENTUNIT_MICROSEC, 0, nullptr);
      ASSERT_TRUE(s.ProcessEvent(1, "EVENT", e).IsOK());
    }
    auto contents = ReadFile(info.csv_output_filepath);
    EXPECT_NE(contents.find(c.expected), std::string::npos)
        << "type=" << c.type << " expected=" << c.expected;
  }
}

TEST_F(QnnUnit_ProfileSerializerTest, ProcessEvent_AllUnits_MapsToExpectedString) {
  // Exercises every arm of GetUnitString, including the unknown-unit fallback.
  struct Case {
    QnnProfile_EventUnit_t unit;
    const char* expected;
  };
  const Case cases[] = {
      {QNN_PROFILE_EVENTUNIT_MICROSEC, "US"},
      {QNN_PROFILE_EVENTUNIT_BYTES, "BYTES"},
      {QNN_PROFILE_EVENTUNIT_CYCLES, "CYCLES"},
      {QNN_PROFILE_EVENTUNIT_COUNT, "COUNT"},
      {QNN_PROFILE_EVENTUNIT_OBJECT, "OBJECT"},
      {QNN_PROFILE_EVENTUNIT_BACKEND, "BACKEND"},
      {99999u, "UNKNOWN"},
  };
  for (const auto& c : cases) {
    ProfilingInfo info;
    info.csv_output_filepath = NewCsvPath("proc_unit");
    info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;
    {
      auto sys = MakeStubSystemInterface();
      Serializer s(info, sys, false);
      ASSERT_TRUE(s.InitCsvFile().IsOK());
      auto e = MakeEvent(QNN_PROFILE_EVENTTYPE_EXECUTE, c.unit, 0, nullptr);
      ASSERT_TRUE(s.ProcessEvent(1, "EVENT", e).IsOK());
    }
    auto contents = ReadFile(info.csv_output_filepath);
    EXPECT_NE(contents.find(c.expected), std::string::npos)
        << "unit=" << c.unit << " expected=" << c.expected;
  }
}

TEST_F(QnnUnit_ProfileSerializerTest, ProcessEvent_SubEventWithParent_Succeeds) {
  // Establish a parent event, reserve its sub-event list, then register a
  // sub-event referencing the same event_id as a mapped parent.
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("proc_sub_ok");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;

  auto sys = MakeStubSystemInterface();
  Serializer s(info, sys, false);

  auto parent_evt = MakeEvent(QNN_PROFILE_EVENTTYPE_EXECUTE, QNN_PROFILE_EVENTUNIT_MICROSEC, 1, "parent");
  ASSERT_TRUE(s.ProcessEvent(1, "EVENT", parent_evt).IsOK());
  auto* parent_sys_evt = s.GetSystemEventPointer(1);
  ASSERT_NE(parent_sys_evt, nullptr);

  s.AddSubEventList(4, parent_sys_evt);
  ASSERT_TRUE(s.SetParentSystemEvent(2, parent_sys_evt).IsOK());
  EXPECT_EQ(s.GetParentSystemEvent(2), parent_sys_evt);

  auto sub_evt = MakeEvent(QNN_PROFILE_EVENTTYPE_NODE, QNN_PROFILE_EVENTUNIT_CYCLES, 100, "child");
  EXPECT_TRUE(s.ProcessEvent(2, "SUB-EVENT", sub_evt).IsOK());
}

TEST_F(QnnUnit_ProfileSerializerTest, ProcessEvent_SubEventWithoutParent_ReturnsError) {
  // SUB-EVENT with no registered parent -> RETURN_IF fires (nullptr parent).
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("proc_sub_no_parent");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;

  auto sys = MakeStubSystemInterface();
  Serializer s(info, sys, false);

  auto sub_evt = MakeEvent(QNN_PROFILE_EVENTTYPE_NODE, QNN_PROFILE_EVENTUNIT_CYCLES, 0, "orphan");
  EXPECT_FALSE(s.ProcessEvent(9999, "SUB-EVENT", sub_evt).IsOK());
}

TEST_F(QnnUnit_ProfileSerializerTest, ProcessEvent_SubEventListMissingInMap_ReturnsError) {
  // Register a parent event but never call AddSubEventList -> AddSubEvent hits
  // the "map entry missing" branch and returns nullptr, which surfaces as an
  // error at the RETURN_IF in ProcessEvent.
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("proc_sub_no_list");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;

  auto sys = MakeStubSystemInterface();
  Serializer s(info, sys, false);

  auto parent_evt = MakeEvent(QNN_PROFILE_EVENTTYPE_EXECUTE, QNN_PROFILE_EVENTUNIT_MICROSEC, 1, "parent");
  ASSERT_TRUE(s.ProcessEvent(1, "EVENT", parent_evt).IsOK());
  auto* parent_sys_evt = s.GetSystemEventPointer(1);
  ASSERT_NE(parent_sys_evt, nullptr);

  ASSERT_TRUE(s.SetParentSystemEvent(2, parent_sys_evt).IsOK());
  // Deliberately skip AddSubEventList.
  auto sub_evt = MakeEvent(QNN_PROFILE_EVENTTYPE_NODE, QNN_PROFILE_EVENTUNIT_CYCLES, 0, "child");
  EXPECT_FALSE(s.ProcessEvent(2, "SUB-EVENT", sub_evt).IsOK());
}

// ===========================================================================
// LookupOnnxSources (exercised through ProcessEvent + op_trace_lookup)
// ===========================================================================

TEST_F(QnnUnit_ProfileSerializerTest, ProcessEvent_TraceLookup_EmitsOnnxSourceOpsColumn) {
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("lookup_ok");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;

  onnxruntime::qnn::OpTraceLookup trace_lookup;
  trace_lookup["add_node"] = {
      {"onnx_add_a", onnxruntime::qnn::TraceTargetType::kOp},
      {"onnx_add_b", onnxruntime::qnn::TraceTargetType::kOp},
      {"tensor_x", onnxruntime::qnn::TraceTargetType::kTensor},
  };
  info.op_trace_lookup = &trace_lookup;

  {
    auto sys = MakeStubSystemInterface();
    Serializer s(info, sys, false);
    ASSERT_TRUE(s.InitCsvFile().IsOK());
    // The ":OpId_17 (cycles)" suffix must be stripped before lookup.
    auto e = MakeEvent(QNN_PROFILE_EVENTTYPE_NODE, QNN_PROFILE_EVENTUNIT_CYCLES, 5,
                       "add_node:OpId_17 (cycles)");
    ASSERT_TRUE(s.ProcessEvent(1, "EVENT", e).IsOK());
  }
  auto contents = ReadFile(info.csv_output_filepath);
  EXPECT_NE(contents.find("ONNX Source Ops"), std::string::npos);
  // Both OP-typed sources appear, joined by ';'. Tensor-typed source omitted.
  EXPECT_NE(contents.find("onnx_add_a;onnx_add_b"), std::string::npos);
  EXPECT_EQ(contents.find("tensor_x"), std::string::npos);
}

TEST_F(QnnUnit_ProfileSerializerTest, ProcessEvent_TraceLookup_MissingKey_EmitsEmptyCell) {
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("lookup_miss");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;
  onnxruntime::qnn::OpTraceLookup trace_lookup;
  info.op_trace_lookup = &trace_lookup;
  {
    auto sys = MakeStubSystemInterface();
    Serializer s(info, sys, false);
    ASSERT_TRUE(s.InitCsvFile().IsOK());
    auto e = MakeEvent(QNN_PROFILE_EVENTTYPE_NODE, QNN_PROFILE_EVENTUNIT_CYCLES, 5, "unknown_id");
    ASSERT_TRUE(s.ProcessEvent(1, "EVENT", e).IsOK());
  }
  auto contents = ReadFile(info.csv_output_filepath);
  // The last column is empty (line ends with just "\n" after the trailing comma).
  EXPECT_NE(contents.find("unknown_id,\n"), std::string::npos);
}

TEST_F(QnnUnit_ProfileSerializerTest, ProcessEvent_TraceLookup_NullIdentifier_EmitsEmptyCell) {
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("lookup_null_id");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;
  onnxruntime::qnn::OpTraceLookup trace_lookup;
  trace_lookup["add_node"] = {{"src", onnxruntime::qnn::TraceTargetType::kOp}};
  info.op_trace_lookup = &trace_lookup;
  {
    auto sys = MakeStubSystemInterface();
    Serializer s(info, sys, false);
    ASSERT_TRUE(s.InitCsvFile().IsOK());
    auto e = MakeEvent(QNN_PROFILE_EVENTTYPE_NODE, QNN_PROFILE_EVENTUNIT_CYCLES, 5, nullptr);
    ASSERT_TRUE(s.ProcessEvent(1, "EVENT", e).IsOK());
  }
  auto contents = ReadFile(info.csv_output_filepath);
  // Identifier printed as NULL and the trace column is empty.
  EXPECT_NE(contents.find("NULL,\n"), std::string::npos);
  EXPECT_EQ(contents.find("src"), std::string::npos);
}

TEST_F(QnnUnit_ProfileSerializerTest, ProcessEvent_TraceLookup_TensorOnlySources_EmitsEmptyCell) {
  // Entry exists but has no OP-typed sources -> result must be empty.
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("lookup_tensor_only");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;
  onnxruntime::qnn::OpTraceLookup trace_lookup;
  trace_lookup["only_tensor"] = {{"t1", onnxruntime::qnn::TraceTargetType::kTensor},
                                 {"t2", onnxruntime::qnn::TraceTargetType::kTensor}};
  info.op_trace_lookup = &trace_lookup;
  {
    auto sys = MakeStubSystemInterface();
    Serializer s(info, sys, false);
    ASSERT_TRUE(s.InitCsvFile().IsOK());
    auto e = MakeEvent(QNN_PROFILE_EVENTTYPE_NODE, QNN_PROFILE_EVENTUNIT_CYCLES, 5, "only_tensor");
    ASSERT_TRUE(s.ProcessEvent(1, "EVENT", e).IsOK());
  }
  auto contents = ReadFile(info.csv_output_filepath);
  EXPECT_NE(contents.find("only_tensor,\n"), std::string::npos);
  EXPECT_EQ(contents.find("t1"), std::string::npos);
}

// ===========================================================================
// ProcessExtendedEvent
// ===========================================================================

namespace {

QnnProfile_ExtendedEventData_t MakeExtEvent(QnnProfile_EventType_t type,
                                            QnnProfile_EventUnit_t unit,
                                            Qnn_DataType_t scalar_type,
                                            const char* identifier,
                                            uint64_t timestamp = 12345) {
  QnnProfile_ExtendedEventData_t e = QNN_PROFILE_EXTENDED_EVENT_DATA_INIT;
  e.version = QNN_PROFILE_DATA_VERSION_1;
  e.v1.type = type;
  e.v1.unit = unit;
  e.v1.timestamp = timestamp;
  e.v1.identifier = identifier;
  e.v1.value.dataType = scalar_type;
  return e;
}

}  // namespace

TEST_F(QnnUnit_ProfileSerializerTest, ProcessExtendedEvent_Version1_WritesRow) {
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("ext_v1");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;
  {
    auto sys = MakeStubSystemInterface();
    Serializer s(info, sys, false);
    ASSERT_TRUE(s.InitCsvFile().IsOK());
    auto e = MakeExtEvent(QNN_PROFILE_EVENTTYPE_EXECUTE, QNN_PROFILE_EVENTUNIT_MICROSEC,
                          QNN_DATATYPE_UINT_32, "abc", 42);
    e.v1.value.uint32Value = 777;
    ASSERT_TRUE(s.ProcessExtendedEvent(1, "EVENT", e).IsOK());
  }
  auto contents = ReadFile(info.csv_output_filepath);
  EXPECT_NE(contents.find("42,EXECUTE,777,US"), std::string::npos);
  EXPECT_NE(contents.find("abc"), std::string::npos);
}

TEST_F(QnnUnit_ProfileSerializerTest, ProcessExtendedEvent_NullIdentifier_WritesNull) {
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("ext_null_id");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;
  {
    auto sys = MakeStubSystemInterface();
    Serializer s(info, sys, false);
    ASSERT_TRUE(s.InitCsvFile().IsOK());
    auto e = MakeExtEvent(QNN_PROFILE_EVENTTYPE_EXECUTE, QNN_PROFILE_EVENTUNIT_MICROSEC,
                          QNN_DATATYPE_UINT_32, nullptr, 5);
    e.v1.value.uint32Value = 1;
    ASSERT_TRUE(s.ProcessExtendedEvent(1, "EVENT", e).IsOK());
  }
  auto contents = ReadFile(info.csv_output_filepath);
  EXPECT_NE(contents.find(",NULL"), std::string::npos);
}

TEST_F(QnnUnit_ProfileSerializerTest, ProcessExtendedEvent_UnsupportedVersion_SkipsCsvRow) {
  // version != QNN_PROFILE_DATA_VERSION_1 -> CSV skipped but return is OK
  // (system event still added).
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("ext_ver_other");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;
  {
    auto sys = MakeStubSystemInterface();
    Serializer s(info, sys, false);
    ASSERT_TRUE(s.InitCsvFile().IsOK());
    auto e = MakeExtEvent(QNN_PROFILE_EVENTTYPE_EXECUTE, QNN_PROFILE_EVENTUNIT_MICROSEC,
                          QNN_DATATYPE_UINT_32, "id", 1);
    e.version = QNN_PROFILE_DATA_VERSION_UNDEFINED;
    ASSERT_TRUE(s.ProcessExtendedEvent(1, "EVENT", e).IsOK());
  }
  auto contents = ReadFile(info.csv_output_filepath);
  // Only the header line, no data row.
  EXPECT_EQ(contents.find("EXECUTE,"), std::string::npos);
}

TEST_F(QnnUnit_ProfileSerializerTest, ProcessExtendedEvent_AllScalarTypes_SerializesValue) {
  // Covers every arm of ExtractQnnScalarValue, including the default case.
  struct Case {
    Qnn_DataType_t dtype;
    // Setter operates on the value union; must match the field selected by dtype.
    void (*set)(Qnn_Scalar_t&);
    const char* expected_substr;
  };
  const Case cases[] = {
      {QNN_DATATYPE_INT_8, [](Qnn_Scalar_t& s) { s.int8Value = -12; }, "-12"},
      {QNN_DATATYPE_INT_16, [](Qnn_Scalar_t& s) { s.int16Value = -1234; }, "-1234"},
      {QNN_DATATYPE_INT_32, [](Qnn_Scalar_t& s) { s.int32Value = -100000; }, "-100000"},
      {QNN_DATATYPE_INT_64, [](Qnn_Scalar_t& s) { s.int64Value = -12345678901LL; }, "-12345678901"},
      {QNN_DATATYPE_UINT_8, [](Qnn_Scalar_t& s) { s.uint8Value = 250; }, "250"},
      {QNN_DATATYPE_UINT_16, [](Qnn_Scalar_t& s) { s.uint16Value = 65530; }, "65530"},
      {QNN_DATATYPE_UINT_32, [](Qnn_Scalar_t& s) { s.uint32Value = 4000000000u; }, "4000000000"},
      {QNN_DATATYPE_UINT_64, [](Qnn_Scalar_t& s) { s.uint64Value = 12345678901ULL; }, "12345678901"},
      {QNN_DATATYPE_FLOAT_16, [](Qnn_Scalar_t& s) { s.floatValue = 1.5f; }, "1.5"},
      {QNN_DATATYPE_FLOAT_32, [](Qnn_Scalar_t& s) { s.floatValue = 2.25f; }, "2.25"},
      {QNN_DATATYPE_SFIXED_POINT_8, [](Qnn_Scalar_t& s) { s.int32Value = -5; }, "-5"},
      {QNN_DATATYPE_SFIXED_POINT_16, [](Qnn_Scalar_t& s) { s.int32Value = -55; }, "-55"},
      {QNN_DATATYPE_SFIXED_POINT_32, [](Qnn_Scalar_t& s) { s.int32Value = -555; }, "-555"},
      {QNN_DATATYPE_UFIXED_POINT_8, [](Qnn_Scalar_t& s) { s.uint32Value = 6; }, "6"},
      {QNN_DATATYPE_UFIXED_POINT_16, [](Qnn_Scalar_t& s) { s.uint32Value = 66; }, "66"},
      {QNN_DATATYPE_UFIXED_POINT_32, [](Qnn_Scalar_t& s) { s.uint32Value = 666; }, "666"},
      {QNN_DATATYPE_BOOL_8, [](Qnn_Scalar_t& s) { s.bool8Value = 1; }, "true"},
      {QNN_DATATYPE_UNDEFINED, [](Qnn_Scalar_t&) {}, "UNKNOWN"},
  };

  for (const auto& c : cases) {
    ProfilingInfo info;
    info.csv_output_filepath = NewCsvPath("ext_scalar");
    info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;
    {
      auto sys = MakeStubSystemInterface();
      Serializer s(info, sys, false);
      ASSERT_TRUE(s.InitCsvFile().IsOK());
      auto e = MakeExtEvent(QNN_PROFILE_EVENTTYPE_EXECUTE, QNN_PROFILE_EVENTUNIT_MICROSEC,
                            c.dtype, "id", 1);
      c.set(e.v1.value);
      ASSERT_TRUE(s.ProcessExtendedEvent(1, "EVENT", e).IsOK());
    }
    auto contents = ReadFile(info.csv_output_filepath);
    EXPECT_NE(contents.find(c.expected_substr), std::string::npos)
        << "dtype=" << c.dtype << " expected=" << c.expected_substr;
  }
}

TEST_F(QnnUnit_ProfileSerializerTest, ProcessExtendedEvent_ScalarBool8False_SerializesFalse) {
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("ext_bool_false");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;
  {
    auto sys = MakeStubSystemInterface();
    Serializer s(info, sys, false);
    ASSERT_TRUE(s.InitCsvFile().IsOK());
    auto e = MakeExtEvent(QNN_PROFILE_EVENTTYPE_EXECUTE, QNN_PROFILE_EVENTUNIT_MICROSEC,
                          QNN_DATATYPE_BOOL_8, "id", 1);
    e.v1.value.bool8Value = 0;
    ASSERT_TRUE(s.ProcessExtendedEvent(1, "EVENT", e).IsOK());
  }
  auto contents = ReadFile(info.csv_output_filepath);
  EXPECT_NE(contents.find("false"), std::string::npos);
}

TEST_F(QnnUnit_ProfileSerializerTest, ProcessExtendedEvent_ScalarString_SerializesEscapedForms) {
  // String scalar: NULL pointer path, and non-NULL path.
  for (bool null_string : {false, true}) {
    ProfilingInfo info;
    info.csv_output_filepath = NewCsvPath("ext_string");
    info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;
    {
      auto sys = MakeStubSystemInterface();
      Serializer s(info, sys, false);
      ASSERT_TRUE(s.InitCsvFile().IsOK());
      auto e = MakeExtEvent(QNN_PROFILE_EVENTTYPE_EXECUTE, QNN_PROFILE_EVENTUNIT_MICROSEC,
                            QNN_DATATYPE_STRING, "id", 1);
      e.v1.value.stringValue = null_string ? nullptr : "hello";
      ASSERT_TRUE(s.ProcessExtendedEvent(1, "EVENT", e).IsOK());
    }
    auto contents = ReadFile(info.csv_output_filepath);
    if (null_string) {
      // dtype=STRING with null pointer -> "NULL" in the value column.
      EXPECT_NE(contents.find(",NULL,"), std::string::npos);
    } else {
      EXPECT_NE(contents.find("hello"), std::string::npos);
    }
  }
}

TEST_F(QnnUnit_ProfileSerializerTest, ProcessExtendedEvent_SubEventWithParent_Succeeds) {
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("ext_sub_ok");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;

  auto sys = MakeStubSystemInterface();
  Serializer s(info, sys, false);

  auto parent = MakeExtEvent(QNN_PROFILE_EVENTTYPE_EXECUTE, QNN_PROFILE_EVENTUNIT_MICROSEC,
                             QNN_DATATYPE_UINT_32, "p", 1);
  parent.v1.value.uint32Value = 10;
  ASSERT_TRUE(s.ProcessExtendedEvent(1, "EVENT", parent).IsOK());
  auto* parent_ptr = s.GetSystemEventPointer(1);
  ASSERT_NE(parent_ptr, nullptr);

  s.AddSubEventList(2, parent_ptr);
  ASSERT_TRUE(s.SetParentSystemEvent(2, parent_ptr).IsOK());

  auto sub = MakeExtEvent(QNN_PROFILE_EVENTTYPE_NODE, QNN_PROFILE_EVENTUNIT_CYCLES,
                          QNN_DATATYPE_UINT_32, "c", 1);
  sub.v1.value.uint32Value = 20;
  EXPECT_TRUE(s.ProcessExtendedEvent(2, "SUB-EVENT", sub).IsOK());
}

TEST_F(QnnUnit_ProfileSerializerTest, ProcessExtendedEvent_SubEventWithoutParent_ReturnsError) {
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("ext_sub_orphan");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;

  auto sys = MakeStubSystemInterface();
  Serializer s(info, sys, false);

  auto sub = MakeExtEvent(QNN_PROFILE_EVENTTYPE_NODE, QNN_PROFILE_EVENTUNIT_CYCLES,
                          QNN_DATATYPE_UINT_32, "orphan", 1);
  sub.v1.value.uint32Value = 20;
  EXPECT_FALSE(s.ProcessExtendedEvent(1, "SUB-EVENT", sub).IsOK());
}

TEST_F(QnnUnit_ProfileSerializerTest, ProcessExtendedEvent_SubEventListMissingInMap_ReturnsError) {
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("ext_sub_no_list");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;

  auto sys = MakeStubSystemInterface();
  Serializer s(info, sys, false);

  auto parent = MakeExtEvent(QNN_PROFILE_EVENTTYPE_EXECUTE, QNN_PROFILE_EVENTUNIT_MICROSEC,
                             QNN_DATATYPE_UINT_32, "p", 1);
  parent.v1.value.uint32Value = 10;
  ASSERT_TRUE(s.ProcessExtendedEvent(1, "EVENT", parent).IsOK());
  auto* parent_ptr = s.GetSystemEventPointer(1);
  ASSERT_NE(parent_ptr, nullptr);
  ASSERT_TRUE(s.SetParentSystemEvent(2, parent_ptr).IsOK());
  // No AddSubEventList: AddExtendedSubEvent returns nullptr -> error.

  auto sub = MakeExtEvent(QNN_PROFILE_EVENTTYPE_NODE, QNN_PROFILE_EVENTUNIT_CYCLES,
                          QNN_DATATYPE_UINT_32, "c", 1);
  sub.v1.value.uint32Value = 20;
  EXPECT_FALSE(s.ProcessExtendedEvent(2, "SUB-EVENT", sub).IsOK());
}

TEST_F(QnnUnit_ProfileSerializerTest, ProcessExtendedEvent_TraceLookup_EmitsOnnxSourceOpsColumn) {
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("ext_lookup");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;

  onnxruntime::qnn::OpTraceLookup trace_lookup;
  trace_lookup["node_x"] = {{"onnx_src", onnxruntime::qnn::TraceTargetType::kOp}};
  info.op_trace_lookup = &trace_lookup;

  {
    auto sys = MakeStubSystemInterface();
    Serializer s(info, sys, false);
    ASSERT_TRUE(s.InitCsvFile().IsOK());
    auto e = MakeExtEvent(QNN_PROFILE_EVENTTYPE_NODE, QNN_PROFILE_EVENTUNIT_CYCLES,
                          QNN_DATATYPE_UINT_32, "node_x:OpId_3 (cycles)", 1);
    e.v1.value.uint32Value = 1;
    ASSERT_TRUE(s.ProcessExtendedEvent(1, "EVENT", e).IsOK());
  }
  auto contents = ReadFile(info.csv_output_filepath);
  EXPECT_NE(contents.find("onnx_src"), std::string::npos);
}

// ===========================================================================
// Sub-event helpers: AddSubEventList / SetParentSystemEvent / Get*
// ===========================================================================

TEST_F(QnnUnit_ProfileSerializerTest, GetSystemEventPointer_UnknownId_ReturnsNull) {
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("get_evt_null");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;
  auto sys = MakeStubSystemInterface();
  Serializer s(info, sys, false);
  EXPECT_EQ(s.GetSystemEventPointer(42), nullptr);
}

TEST_F(QnnUnit_ProfileSerializerTest, GetParentSystemEvent_UnknownId_ReturnsNull) {
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("get_parent_null");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;
  auto sys = MakeStubSystemInterface();
  Serializer s(info, sys, false);
  EXPECT_EQ(s.GetParentSystemEvent(42), nullptr);
}

TEST_F(QnnUnit_ProfileSerializerTest, SetParentSystemEvent_Duplicate_ReturnsError) {
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("dup_parent");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;
  auto sys = MakeStubSystemInterface();
  Serializer s(info, sys, false);

  QnnSystemProfile_ProfileEventV1_t sentinel{};
  ASSERT_TRUE(s.SetParentSystemEvent(7, &sentinel).IsOK());
  EXPECT_FALSE(s.SetParentSystemEvent(7, &sentinel).IsOK());
}

TEST_F(QnnUnit_ProfileSerializerTest, AddSubEventList_ZeroCount_DoesNotRegisterMapEntry) {
  // Cannot inspect the private map directly, but if no entry exists,
  // ProcessEvent(SUB-EVENT) with a matching parent must fail (AddSubEvent
  // returns nullptr).
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("subevt_zero");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;
  auto sys = MakeStubSystemInterface();
  Serializer s(info, sys, false);

  auto parent = MakeEvent(QNN_PROFILE_EVENTTYPE_EXECUTE, QNN_PROFILE_EVENTUNIT_MICROSEC, 1, "p");
  ASSERT_TRUE(s.ProcessEvent(1, "EVENT", parent).IsOK());
  auto* parent_ptr = s.GetSystemEventPointer(1);
  ASSERT_NE(parent_ptr, nullptr);

  s.AddSubEventList(0, parent_ptr);  // zero -> map entry NOT created
  ASSERT_TRUE(s.SetParentSystemEvent(2, parent_ptr).IsOK());
  auto sub = MakeEvent(QNN_PROFILE_EVENTTYPE_NODE, QNN_PROFILE_EVENTUNIT_CYCLES, 0, "c");
  EXPECT_FALSE(s.ProcessEvent(2, "SUB-EVENT", sub).IsOK());
}

// ===========================================================================
// SerializeEventsToQnnLog
// ===========================================================================

TEST_F(QnnUnit_ProfileSerializerTest, SerializeEventsToQnnLog_NullCreateFn_ReturnsError) {
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("qnn_log_null_create");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;

  QNN_SYSTEM_INTERFACE_VER_TYPE sys = QNN_SYSTEM_INTERFACE_VER_TYPE_INIT;
  // Leave all three function pointers null.
  Serializer s(info, sys, false);
  EXPECT_FALSE(s.SerializeEventsToQnnLog().IsOK());
}

TEST_F(QnnUnit_ProfileSerializerTest, SerializeEventsToQnnLog_UnknownMethodType_ReturnsError) {
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("qnn_log_unknown");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::UNKNOWN;

  auto sys = MakeStubSystemInterface();
  ProfileSerializerStubState state;
  UseProfileStubs use(state);

  Serializer s(info, sys, false);
  EXPECT_FALSE(s.SerializeEventsToQnnLog().IsOK());
  EXPECT_EQ(state.create_calls, 0);
  EXPECT_EQ(state.serialize_calls, 0);
}

TEST_F(QnnUnit_ProfileSerializerTest, SerializeEventsToQnnLog_CreateTargetFails_ReturnsError) {
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("qnn_log_create_fail");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;

  auto sys = MakeStubSystemInterface();
  ProfileSerializerStubState state;
  state.create_ret = QNN_SYSTEM_PROFILE_ERROR_INVALID_ARGUMENT;
  UseProfileStubs use(state);

  Serializer s(info, sys, false);
  auto status = s.SerializeEventsToQnnLog();
  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ(state.create_calls, 1);
  EXPECT_EQ(state.serialize_calls, 0);
}

TEST_F(QnnUnit_ProfileSerializerTest, SerializeEventsToQnnLog_SerializeFails_ReturnsError) {
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("qnn_log_ser_fail");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;

  auto sys = MakeStubSystemInterface();
  ProfileSerializerStubState state;
  state.serialize_ret = QNN_SYSTEM_PROFILE_ERROR_MEM_ALLOC;
  UseProfileStubs use(state);

  Serializer s(info, sys, false);
  auto status = s.SerializeEventsToQnnLog();
  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ(state.create_calls, 1);
  EXPECT_EQ(state.serialize_calls, 1);
}

TEST_F(QnnUnit_ProfileSerializerTest, SerializeEventsToQnnLog_FreeFails_ReturnsError) {
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("qnn_log_free_fail");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;

  auto sys = MakeStubSystemInterface();
  ProfileSerializerStubState state;
  state.free_ret = QNN_SYSTEM_PROFILE_ERROR_INVALID_HANDLE;
  UseProfileStubs use(state);

  Serializer s(info, sys, false);
  auto status = s.SerializeEventsToQnnLog();
  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ(state.create_calls, 1);
  EXPECT_EQ(state.serialize_calls, 1);
  // FreeHandle is called twice: once explicitly at the end of the method (which
  // fails and triggers RETURN_IF), and again from the RAII wrapper's destructor.
  EXPECT_EQ(state.free_calls, 2);
}

TEST_F(QnnUnit_ProfileSerializerTest, SerializeEventsToQnnLog_Success_PropagatesHeader) {
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("qnn_log_ok");
  info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;
  info.graph_name = "graph42";
  info.start_time = 111;
  info.stop_time = 222;

  auto sys = MakeStubSystemInterface();
  ProfileSerializerStubState state;
  UseProfileStubs use(state);

  Serializer s(info, sys, false);
  // Register a couple of events so numProfilingEvents > 0 and the sub-event
  // pointer plumbing at the top of SerializeEventsToQnnLog executes.
  auto e1 = MakeEvent(QNN_PROFILE_EVENTTYPE_EXECUTE, QNN_PROFILE_EVENTUNIT_MICROSEC, 1, "p");
  ASSERT_TRUE(s.ProcessEvent(1, "EVENT", e1).IsOK());
  auto* parent_ptr = s.GetSystemEventPointer(1);
  ASSERT_NE(parent_ptr, nullptr);
  s.AddSubEventList(2, parent_ptr);
  ASSERT_TRUE(s.SetParentSystemEvent(2, parent_ptr).IsOK());
  auto e2 = MakeEvent(QNN_PROFILE_EVENTTYPE_NODE, QNN_PROFILE_EVENTUNIT_CYCLES, 2, "c");
  ASSERT_TRUE(s.ProcessEvent(2, "SUB-EVENT", e2).IsOK());

  EXPECT_TRUE(s.SerializeEventsToQnnLog().IsOK());
  EXPECT_EQ(state.create_calls, 1);
  EXPECT_EQ(state.serialize_calls, 1);
  // Two frees: the explicit call inside SerializeEventsToQnnLog and the RAII
  // wrapper's destructor.
  EXPECT_EQ(state.free_calls, 2);
  EXPECT_EQ(state.last_method_type, QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_EXECUTE);
  EXPECT_EQ(state.last_start_time, 111u);
  EXPECT_EQ(state.last_stop_time, 222u);
  EXPECT_EQ(state.last_app_name, "OnnxRuntime");
  EXPECT_FALSE(state.last_backend_version.empty());
  EXPECT_EQ(state.last_num_events, 1u);  // one top-level event ("p"); child is a sub-event
}

TEST_F(QnnUnit_ProfileSerializerTest, SerializeEventsToQnnLog_ParseMethodType_MapsAllMethods) {
  // Exercises every named arm of ParseMethodType. UNKNOWN is covered separately
  // by SerializeEventsToQnnLog_UnknownMethodType_ReturnsError above.
  struct Case {
    onnxruntime::qnn::ProfilingMethodType in;
    QnnSystemProfile_MethodType_t expected;
  };
  const Case cases[] = {
      {onnxruntime::qnn::ProfilingMethodType::EXECUTE, QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_EXECUTE},
      {onnxruntime::qnn::ProfilingMethodType::FINALIZE, QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_FINALIZE},
      {onnxruntime::qnn::ProfilingMethodType::EXECUTE_ASYNC,
       QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_EXECUTE_ASYNC},
      {onnxruntime::qnn::ProfilingMethodType::CREATE_FROM_BINARY,
       QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_CREATE_FROM_BINARY},
      {onnxruntime::qnn::ProfilingMethodType::DEINIT, QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_DEINIT},
      {onnxruntime::qnn::ProfilingMethodType::CONTEXT_CREATE,
       QNN_SYSTEM_PROFILE_METHOD_TYPE_APP_CONTEXT_CREATE},
      {onnxruntime::qnn::ProfilingMethodType::COMPOSE_GRAPHS,
       QNN_SYSTEM_PROFILE_METHOD_TYPE_APP_COMPOSE_GRAPHS},
      {onnxruntime::qnn::ProfilingMethodType::EXECUTE_IPS,
       QNN_SYSTEM_PROFILE_METHOD_TYPE_APP_EXECUTE_IPS},
      {onnxruntime::qnn::ProfilingMethodType::GRAPH_COMPONENT,
       QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_GRAPH_COMPONENT},
      {onnxruntime::qnn::ProfilingMethodType::LIB_LOAD,
       QNN_SYSTEM_PROFILE_METHOD_TYPE_APP_BACKEND_LIB_LOAD},
      {onnxruntime::qnn::ProfilingMethodType::APPLY_BINARY_SECTION,
       QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_APPLY_BINARY_SECTION},
      {onnxruntime::qnn::ProfilingMethodType::CONTEXT_FINALIZE,
       QNN_SYSTEM_PROFILE_METHOD_TYPE_CONTEXT_FINALIZE},
  };
  for (const auto& c : cases) {
    ProfilingInfo info;
    info.csv_output_filepath = NewCsvPath("qnn_log_method");
    info.method_type = c.in;

    ProfileSerializerStubState state;
    UseProfileStubs use(state);
    auto sys = MakeStubSystemInterface();
    Serializer s(info, sys, false);
    EXPECT_TRUE(s.SerializeEventsToQnnLog().IsOK()) << "input=" << static_cast<int>(c.in);
    EXPECT_EQ(state.last_method_type, c.expected) << "input=" << static_cast<int>(c.in);
  }
}

TEST_F(QnnUnit_ProfileSerializerTest, SerializeEventsToQnnLog_ParseMethodType_OutOfRange_MapsToNone) {
  // An out-of-range ProfilingMethodType passes the UNKNOWN gate but falls into
  // the default arm of ParseMethodType -> QNN_SYSTEM_PROFILE_METHOD_TYPE_NONE.
  ProfilingInfo info;
  info.csv_output_filepath = NewCsvPath("qnn_log_bad_method");
  info.method_type = static_cast<onnxruntime::qnn::ProfilingMethodType>(200);

  ProfileSerializerStubState state;
  UseProfileStubs use(state);
  auto sys = MakeStubSystemInterface();
  Serializer s(info, sys, false);
  EXPECT_TRUE(s.SerializeEventsToQnnLog().IsOK());
  EXPECT_EQ(state.last_method_type, QNN_SYSTEM_PROFILE_METHOD_TYPE_NONE);
}

TEST_F(QnnUnit_ProfileSerializerTest, SerializeEventsToQnnLog_ErrorStringsPropagate) {
  // The error message returned by RETURN_IF concatenates the value from
  // GetSystemProfileErrorString(). We assert on the presence of each mapped
  // error string in the returned status text to exercise every arm.
  struct Case {
    Qnn_ErrorHandle_t err;
    const char* expected_substr;
  };
  const Case cases[] = {
      {QNN_SYSTEM_PROFILE_ERROR_UNSUPPORTED_FEATURE, "Unsupported Feature"},
      {QNN_SYSTEM_PROFILE_ERROR_INVALID_HANDLE, "Invalid Handle"},
      {QNN_SYSTEM_PROFILE_ERROR_INVALID_ARGUMENT, "Invalid Argument"},
      {QNN_SYSTEM_PROFILE_ERROR_MEM_ALLOC, "Memory Allocation Error"},
      {0x7EADBEEFULL, "Unknown"},  // outside mapped errors -> "Unknown"
  };
  for (const auto& c : cases) {
    ProfilingInfo info;
    info.csv_output_filepath = NewCsvPath("qnn_log_err_str");
    info.method_type = onnxruntime::qnn::ProfilingMethodType::EXECUTE;

    ProfileSerializerStubState state;
    state.create_ret = c.err;  // force the create branch to fail with this error
    UseProfileStubs use(state);
    auto sys = MakeStubSystemInterface();
    Serializer s(info, sys, false);
    auto status = s.SerializeEventsToQnnLog();
    ASSERT_FALSE(status.IsOK());
    // Ort::Status message accessor: Message() returns the C-string.
    const std::string msg = status.GetErrorMessage();
    EXPECT_NE(msg.find(c.expected_substr), std::string::npos)
        << "err=" << c.err << " msg=" << msg;
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
