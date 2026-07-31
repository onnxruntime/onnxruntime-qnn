// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Function-level unit tests for OnnxOpAffinity (qnn_onnx_op_affinity.cc) and the
// utils::TrimWhitespace helper it relies on (builder/qnn_utils.h).
//
// OnnxOpAffinity is pure policy logic: it parses the op_affinity provider
// option (inline spec or "@<path>" JSON config file) and decides whether a node
// group's target op is kept off QNN. It touches no QNN API, so it is exercised
// entirely on the host with fake OrtNode pointers -- no device, no session.
//
// ShouldFilterOff() takes an OrtNodeUnit and a QnnBackendType (the session's
// backend). It reads only OpType(), which routes through the global Ort::GetApi().
// Tests therefore build a single-node OrtNodeUnit via stub OrtApi function
// pointers and install OrtGlobalApiOverride so OpType() resolves through the same
// stub (see qnn_unit_test_utils.h); the backend is passed in directly.

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/qnn_onnx_op_affinity.h"
#include "test/providers/qnn/unit/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {

using qnn::OnnxOpAffinity;
using qnn::QnnBackendType;

// Default session backend used by ShouldFilterOff tests that are not about backend scoping.
constexpr QnnBackendType kHtp = QnnBackendType::HTP;

// ============================================================
// OnnxOpAffinity::IsActive  (no OrtNodeUnit needed)
// ============================================================

TEST(QnnUnit_OnnxOpAffinityTest, IsActive_Default_False) {
  OnnxOpAffinity filter;  // exclude + empty -> no filtering
  EXPECT_FALSE(filter.IsActive());
}

TEST(QnnUnit_OnnxOpAffinityTest, IsActive_ExcludeEmptyList_False) {
  OnnxOpAffinity filter("exclude:");  // exclude with no op types
  EXPECT_FALSE(filter.IsActive());
}

TEST(QnnUnit_OnnxOpAffinityTest, IsActive_IncludeEmptyList_True) {
  // include with an empty list is still active: it forces ALL ops off QNN.
  OnnxOpAffinity filter("include:");
  EXPECT_TRUE(filter.IsActive());
}

// ============================================================
// OnnxOpAffinity config-file constructor
// ============================================================

namespace {

// Writes `contents` to a uniquely-named temp file and returns the path. The file
// is removed by RemoveTempFile() at the end of the test.
std::filesystem::path WriteTempConfig(const std::string& stem, const std::string& contents) {
  std::filesystem::path path = std::filesystem::temp_directory_path() / stem;
  std::ofstream ofs(path);
  ofs << contents;
  ofs.close();
  return path;
}

void RemoveTempFile(const std::filesystem::path& path) {
  std::error_code ec;
  std::filesystem::remove(path, ec);
}

}  // namespace

TEST(QnnUnit_OnnxOpAffinityTest, ConfigFileCtor_MissingFile_Throws) {
  const std::filesystem::path missing =
      std::filesystem::temp_directory_path() / "qnn_onnx_op_affinity_does_not_exist.json";
  RemoveTempFile(missing);  // ensure absent
  EXPECT_THROW(OnnxOpAffinity{missing}, std::runtime_error);
}

TEST(QnnUnit_OnnxOpAffinityTest, ConfigFileCtor_MalformedJson_Throws) {
  const std::filesystem::path path =
      WriteTempConfig("qnn_onnx_op_affinity_malformed.json", "{ this is not json ");
  EXPECT_ANY_THROW(OnnxOpAffinity{path});
  RemoveTempFile(path);
}

// The config-file ctor validates every field's type/value loudly (D-1), matching the inline path's
// error strategy instead of silently coercing a typo to a surprising default. FromOptionValue()
// catches these throws and degrades the whole filter to inactive + WARNING.

TEST(QnnUnit_OnnxOpAffinityTest, ConfigFileCtor_UnrecognizedMode_Throws) {
  // "Include" (wrong case) / "excludes" (extra s) would silently become exclude without validation.
  for (const char* bad_mode : {"Include", "excludes", "filter"}) {
    const std::filesystem::path path = WriteTempConfig(
        "qnn_onnx_op_affinity_bad_mode.json",
        std::string(R"({ "mode": ")") + bad_mode + R"(", "op_types": ["Conv"] })");
    EXPECT_THROW(OnnxOpAffinity{path}, std::runtime_error) << "mode=" << bad_mode;
    RemoveTempFile(path);
  }
}

TEST(QnnUnit_OnnxOpAffinityTest, ConfigFileCtor_NonStringBackend_Throws) {
  const std::filesystem::path path = WriteTempConfig(
      "qnn_onnx_op_affinity_bad_backend.json",
      R"({ "backend": 123, "op_types": ["Conv"] })");
  EXPECT_THROW(OnnxOpAffinity{path}, std::runtime_error);
  RemoveTempFile(path);
}

TEST(QnnUnit_OnnxOpAffinityTest, ConfigFileCtor_NonArrayOpTypes_Throws) {
  const std::filesystem::path path = WriteTempConfig(
      "qnn_onnx_op_affinity_bad_optypes.json",
      R"({ "mode": "exclude", "op_types": "Conv" })");
  EXPECT_THROW(OnnxOpAffinity{path}, std::runtime_error);
  RemoveTempFile(path);
}

TEST(QnnUnit_OnnxOpAffinityTest, ConfigFileCtor_NonStringOpTypeElement_Throws) {
  const std::filesystem::path path = WriteTempConfig(
      "qnn_onnx_op_affinity_bad_optype_elem.json",
      R"({ "op_types": ["Conv", 123] })");
  EXPECT_THROW(OnnxOpAffinity{path}, std::runtime_error);
  RemoveTempFile(path);
}

TEST(QnnUnit_OnnxOpAffinityTest, FromOptionValue_UnrecognizedMode_DegradesToInactive) {
  Ort::Logger logger = MakeNullLogger();
  const std::filesystem::path path = WriteTempConfig(
      "qnn_onnx_op_affinity_from_option_bad_mode.json",
      R"({ "mode": "Include", "op_types": ["Conv"] })");
  // The ctor throws on the bad mode; FromOptionValue catches and degrades to no filtering.
  OnnxOpAffinity filter = OnnxOpAffinity::FromOptionValue("@" + path.string(), logger);
  EXPECT_FALSE(filter.IsActive());
  RemoveTempFile(path);
}

// ============================================================
// OnnxOpAffinity::ShouldFilterOff -- fixture builds a single-node OrtNodeUnit
// whose OpType() is configurable via g_fake_op_type.
// ============================================================

namespace {

// Read by both InitForSingleNode (via the parameter OrtApi) and OrtNodeUnit::
// OpType() (via the global OrtApi). gtest runs tests sequentially, so a single
// process-wide string is safe.
std::string g_fake_op_type = "Softmax";

const OrtValueInfo* const kFakeIO =
    reinterpret_cast<const OrtValueInfo*>(uintptr_t{0x1000});
const OrtTypeInfo* const kFakeTypeInfo =
    reinterpret_cast<const OrtTypeInfo*>(uintptr_t{0x2000});
const OrtTensorTypeAndShapeInfo* const kFakeTensorShape =
    reinterpret_cast<const OrtTensorTypeAndShapeInfo*>(uintptr_t{0x3000});
const OrtNode* const kFakeNode =
    reinterpret_cast<const OrtNode*>(uintptr_t{0x100});

OrtStatus* StubOpType(const OrtNode*, const char** op) noexcept {
  *op = g_fake_op_type.c_str();
  return nullptr;
}
OrtStatus* StubNumInputs(const OrtNode*, size_t* n) noexcept {
  *n = 1;
  return nullptr;
}
OrtStatus* StubNumOutputs(const OrtNode*, size_t* n) noexcept {
  *n = 1;
  return nullptr;
}
OrtStatus* StubGetInputs(const OrtNode*, const OrtValueInfo** io, size_t) noexcept {
  io[0] = kFakeIO;
  return nullptr;
}
OrtStatus* StubGetOutputs(const OrtNode*, const OrtValueInfo** io, size_t) noexcept {
  io[0] = kFakeIO;
  return nullptr;
}
OrtStatus* StubValueInfoName(const OrtValueInfo*, const char** name) noexcept {
  static const char n[] = "x";
  *name = n;
  return nullptr;
}
OrtStatus* StubValueInfoTypeInfo(const OrtValueInfo*, const OrtTypeInfo** ti) noexcept {
  *ti = kFakeTypeInfo;
  return nullptr;
}
OrtStatus* StubCastTensorInfo(const OrtTypeInfo*, const OrtTensorTypeAndShapeInfo** ts) noexcept {
  *ts = kFakeTensorShape;
  return nullptr;
}
OrtStatus* StubElemType(const OrtTensorTypeAndShapeInfo*, ONNXTensorElementDataType* t) noexcept {
  *t = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  return nullptr;
}
bool StubHasShape(const OrtTensorTypeAndShapeInfo*) noexcept { return true; }
OrtStatus* StubDimsCount(const OrtTensorTypeAndShapeInfo*, size_t* n) noexcept {
  *n = 2;
  return nullptr;
}
OrtStatus* StubDims(const OrtTensorTypeAndShapeInfo*, int64_t* dims, size_t) noexcept {
  dims[0] = 3;
  dims[1] = 4;
  return nullptr;
}

// Builds an OrtApi sufficient to construct a single-node OrtNodeUnit for a plain
// (non-QDQ) op and to answer OpType().
OrtApi MakeSingleNodeApi() {
  OrtApi api{};
  api.Node_GetOperatorType = StubOpType;
  api.Node_GetNumInputs = StubNumInputs;
  api.Node_GetNumOutputs = StubNumOutputs;
  api.Node_GetInputs = StubGetInputs;
  api.Node_GetOutputs = StubGetOutputs;
  api.GetValueInfoName = StubValueInfoName;
  api.GetValueInfoTypeInfo = StubValueInfoTypeInfo;
  api.CastTypeInfoToTensorInfo = StubCastTensorInfo;
  api.GetTensorElementType = StubElemType;
  api.TensorTypeAndShape_HasShape = StubHasShape;
  api.GetDimensionsCount = StubDimsCount;
  api.GetDimensions = StubDims;
  return api;
}

}  // namespace

class QnnUnit_OnnxOpAffinityShouldFilterOffTest : public ::testing::Test {
 protected:
  // Returns a single-node OrtNodeUnit whose OpType() == op_type. The global API
  // override stays active for the whole test, so OpType() resolves through the
  // stub during ShouldFilterOff().
  OrtNodeUnit MakeNodeUnit(const std::string& op_type) {
    g_fake_op_type = op_type;
    return OrtNodeUnit(kFakeNode, api_);
  }

  OrtApi api_ = MakeSingleNodeApi();
  OrtGlobalApiOverride override_{&api_};
};

TEST_F(QnnUnit_OnnxOpAffinityShouldFilterOffTest, Inactive_NeverFiltersOff) {
  OnnxOpAffinity filter;  // default inactive
  OrtNodeUnit unit = MakeNodeUnit("Softmax");
  EXPECT_FALSE(filter.ShouldFilterOff(unit, kHtp));
}

TEST_F(QnnUnit_OnnxOpAffinityShouldFilterOffTest, Exclude_ListedOp_FilteredOff) {
  OnnxOpAffinity filter("Softmax");  // exclude:Softmax
  OrtNodeUnit unit = MakeNodeUnit("Softmax");
  EXPECT_TRUE(filter.ShouldFilterOff(unit, kHtp));
}

TEST_F(QnnUnit_OnnxOpAffinityShouldFilterOffTest, Exclude_UnlistedOp_Kept) {
  OnnxOpAffinity filter("Softmax");
  OrtNodeUnit unit = MakeNodeUnit("Conv");  // not in the exclude list
  EXPECT_FALSE(filter.ShouldFilterOff(unit, kHtp));
}

TEST_F(QnnUnit_OnnxOpAffinityShouldFilterOffTest, Include_ListedOp_Kept) {
  OnnxOpAffinity filter("include:Conv");
  OrtNodeUnit unit = MakeNodeUnit("Conv");  // in the include list -> stays on QNN
  EXPECT_FALSE(filter.ShouldFilterOff(unit, kHtp));
}

TEST_F(QnnUnit_OnnxOpAffinityShouldFilterOffTest, Include_UnlistedOp_FilteredOff) {
  OnnxOpAffinity filter("include:Conv");
  OrtNodeUnit unit = MakeNodeUnit("Softmax");  // not in include list -> forced off
  EXPECT_TRUE(filter.ShouldFilterOff(unit, kHtp));
}

TEST_F(QnnUnit_OnnxOpAffinityShouldFilterOffTest, Exclude_MatchIsCaseSensitive) {
  OnnxOpAffinity filter("Softmax");
  OrtNodeUnit unit = MakeNodeUnit("softmax");  // lowercase -> no match -> kept
  EXPECT_FALSE(filter.ShouldFilterOff(unit, kHtp));
}

TEST_F(QnnUnit_OnnxOpAffinityShouldFilterOffTest, IncludeEmpty_ForcesAllOff) {
  OnnxOpAffinity filter("include:");  // empty include list -> every op filtered off
  EXPECT_TRUE(filter.ShouldFilterOff(MakeNodeUnit("Conv"), kHtp));
  EXPECT_TRUE(filter.ShouldFilterOff(MakeNodeUnit("Softmax"), kHtp));
}

// A colon inside an op type (e.g. a UDO "custom:MyOp") must be preserved -- only
// the exact "exclude"/"include" prefixes (and known backend names) are treated as
// prefixes.
TEST_F(QnnUnit_OnnxOpAffinityShouldFilterOffTest, Exclude_UdoOpTypeWithColon_Preserved) {
  OnnxOpAffinity filter("custom:MyOp");  // no mode/backend prefix -> literal op type
  EXPECT_TRUE(filter.ShouldFilterOff(MakeNodeUnit("custom:MyOp"), kHtp));
  EXPECT_FALSE(filter.ShouldFilterOff(MakeNodeUnit("MyOp"), kHtp));
}

// ============================================================
// Backend scope (phase 1): a filter scoped to a backend applies only to that
// backend's session; scoped to another backend it is inert.
// ============================================================

TEST_F(QnnUnit_OnnxOpAffinityShouldFilterOffTest, BackendScope_Inline_MatchingBackend_FiltersOff) {
  OnnxOpAffinity filter("htp:exclude:Softmax");
  EXPECT_TRUE(filter.ShouldFilterOff(MakeNodeUnit("Softmax"), QnnBackendType::HTP));
}

TEST_F(QnnUnit_OnnxOpAffinityShouldFilterOffTest, BackendScope_Inline_OtherBackend_Inert) {
  OnnxOpAffinity filter("htp:exclude:Softmax");
  // Same op, but the session runs GPU: the htp-scoped filter must not apply.
  EXPECT_FALSE(filter.ShouldFilterOff(MakeNodeUnit("Softmax"), QnnBackendType::GPU));
}

TEST_F(QnnUnit_OnnxOpAffinityShouldFilterOffTest, BackendScope_Inline_DefaultsToExcludeMode) {
  OnnxOpAffinity filter("htp:Softmax");  // backend prefix, no explicit mode -> exclude
  EXPECT_TRUE(filter.ShouldFilterOff(MakeNodeUnit("Softmax"), QnnBackendType::HTP));
  EXPECT_FALSE(filter.ShouldFilterOff(MakeNodeUnit("Conv"), QnnBackendType::HTP));
}

TEST_F(QnnUnit_OnnxOpAffinityShouldFilterOffTest, BackendScope_Unscoped_AppliesToAnyBackend) {
  OnnxOpAffinity filter("exclude:Softmax");  // no backend scope
  EXPECT_TRUE(filter.ShouldFilterOff(MakeNodeUnit("Softmax"), QnnBackendType::HTP));
  EXPECT_TRUE(filter.ShouldFilterOff(MakeNodeUnit("Softmax"), QnnBackendType::GPU));
}

// AppliesToBackend is the single source of truth for "does this filter apply".
TEST(QnnUnit_OnnxOpAffinityTest, AppliesToBackend_ScopedMatchesOnlyItsBackend) {
  OnnxOpAffinity filter("htp:exclude:Softmax");
  EXPECT_TRUE(filter.AppliesToBackend(QnnBackendType::HTP));
  EXPECT_FALSE(filter.AppliesToBackend(QnnBackendType::GPU));
  EXPECT_FALSE(filter.AppliesToBackend(QnnBackendType::CPU));
}

TEST(QnnUnit_OnnxOpAffinityTest, AppliesToBackend_UnscopedMatchesAll) {
  OnnxOpAffinity filter("exclude:Softmax");
  EXPECT_TRUE(filter.AppliesToBackend(QnnBackendType::HTP));
  EXPECT_TRUE(filter.AppliesToBackend(QnnBackendType::GPU));
  EXPECT_TRUE(filter.AppliesToBackend(QnnBackendType::CPU));
}

// Device-independent drift protection (D-2): iterate EVERY QnnBackendType enumerator and assert that a
// filter scoped to that backend's own name (QnnBackendTypeToString) applies to it. This is the
// counterpart to the backend-scope parsing SSOT -- IsKnownBackendName derives the accepted names from
// QnnBackendTypeToString, and this test guarantees each of those names round-trips back to its own
// enumerator through AppliesToBackend. Unlike the per-backend E2E tests it needs no device/driver, so
// name drift in cpu/dsp/ir/htp_fp16 (the enumerators the E2E tests don't cover, or skip when hardware
// is absent) can never slip through unnoticed. "ir" (== SERIALIZER, where the enum name and the string
// differ most) is the case this most directly guards.
TEST(QnnUnit_OnnxOpAffinityTest, AppliesToBackend_EveryBackendMatchesItsOwnName) {
  for (uint8_t i = 0; i <= static_cast<uint8_t>(QnnBackendType::SERIALIZER); ++i) {
    const auto backend = static_cast<QnnBackendType>(i);
    const std::string name = qnn::QnnBackendTypeToString(backend);
    OnnxOpAffinity filter(name + ":exclude:Softmax");
    EXPECT_TRUE(filter.AppliesToBackend(backend))
        << "filter scoped to '" << name << "' did not apply to its own backend (enum " << int{i}
        << ") -- backend-name drift between QnnBackendTypeToString and the op_affinity scope parser.";
  }
}

// Backend scope combined with include mode. On the matching backend, include semantics apply as
// usual (listed ops stay on QNN, everything else is forced off).
TEST_F(QnnUnit_OnnxOpAffinityShouldFilterOffTest, BackendScope_IncludeMode_MatchingBackend_Parsed) {
  OnnxOpAffinity filter("htp:include:Conv");                                          // two prefixes: backend then mode
  EXPECT_FALSE(filter.ShouldFilterOff(MakeNodeUnit("Conv"), QnnBackendType::HTP));    // in list -> kept
  EXPECT_TRUE(filter.ShouldFilterOff(MakeNodeUnit("Softmax"), QnnBackendType::HTP));  // not in list -> off
}

// Backend gate must win over include's aggressive "force everything not-listed off" behavior: on a
// non-matching backend the whole filter is inert, so it must NOT push any op off QNN. This guards
// the early-return-before-include-logic ordering in ShouldFilterOff().
TEST_F(QnnUnit_OnnxOpAffinityShouldFilterOffTest, BackendScope_IncludeMode_OtherBackend_Inert) {
  OnnxOpAffinity filter("htp:include:Conv");
  EXPECT_FALSE(filter.ShouldFilterOff(MakeNodeUnit("Softmax"), QnnBackendType::GPU));  // NOT forced off
  EXPECT_FALSE(filter.ShouldFilterOff(MakeNodeUnit("Conv"), QnnBackendType::GPU));
}

// Backend prefixes are matched case-sensitively (like mode prefixes). "HTP:" is not a recognized
// backend name, so it is not treated as a scope; the whole value becomes a single literal op type
// "HTP:exclude:Softmax" (colons inside a token are legal, per the UDO rule). The filter therefore
// has no backend scope (applies to any backend) but only matches that exact odd op-type string.
TEST_F(QnnUnit_OnnxOpAffinityShouldFilterOffTest, BackendPrefix_IsCaseSensitive) {
  OnnxOpAffinity filter("HTP:exclude:Softmax");
  EXPECT_TRUE(filter.AppliesToBackend(QnnBackendType::HTP));                                      // no scope parsed
  EXPECT_FALSE(filter.ShouldFilterOff(MakeNodeUnit("Softmax"), QnnBackendType::HTP));             // plain Softmax not matched
  EXPECT_TRUE(filter.ShouldFilterOff(MakeNodeUnit("HTP:exclude:Softmax"), QnnBackendType::HTP));  // literal token
}

// ============================================================
// OnnxOpAffinity::FromOptionValue -- degrade-not-throw on bad input (needs logger)
// ============================================================

TEST(QnnUnit_OnnxOpAffinityTest, FromOptionValue_Empty_ReturnsInactive) {
  Ort::Logger logger = MakeNullLogger();
  OnnxOpAffinity filter = OnnxOpAffinity::FromOptionValue("", logger);
  EXPECT_FALSE(filter.IsActive());
}

TEST(QnnUnit_OnnxOpAffinityTest, FromOptionValue_Inline_ParsesActive) {
  Ort::Logger logger = MakeNullLogger();
  OnnxOpAffinity filter = OnnxOpAffinity::FromOptionValue("Softmax", logger);
  EXPECT_TRUE(filter.IsActive());
}

TEST(QnnUnit_OnnxOpAffinityTest, FromOptionValue_MissingConfigFile_DegradesToInactive) {
  Ort::Logger logger = MakeNullLogger();
  const std::filesystem::path missing =
      std::filesystem::temp_directory_path() / "qnn_onnx_op_affinity_from_option_missing.json";
  RemoveTempFile(missing);
  // "@<missing>" makes the config-file ctor throw; FromOptionValue catches and
  // returns an inactive filter rather than propagating.
  OnnxOpAffinity filter = OnnxOpAffinity::FromOptionValue("@" + missing.string(), logger);
  EXPECT_FALSE(filter.IsActive());
}

TEST(QnnUnit_OnnxOpAffinityTest, FromOptionValue_ValidConfigFile_ParsesActive) {
  Ort::Logger logger = MakeNullLogger();
  const std::filesystem::path path = WriteTempConfig(
      "qnn_onnx_op_affinity_from_option_valid.json",
      R"({ "mode": "include", "op_types": ["Conv"] })");
  OnnxOpAffinity filter = OnnxOpAffinity::FromOptionValue("@" + path.string(), logger);
  EXPECT_TRUE(filter.IsActive());
  RemoveTempFile(path);
}

TEST(QnnUnit_OnnxOpAffinityTest, FromOptionValue_ConfigFileBackendScope_AppliesToScopedBackend) {
  Ort::Logger logger = MakeNullLogger();
  const std::filesystem::path path = WriteTempConfig(
      "qnn_onnx_op_affinity_backend_scope.json",
      R"({ "backend": "htp", "mode": "exclude", "op_types": ["Softmax"] })");
  OnnxOpAffinity filter = OnnxOpAffinity::FromOptionValue("@" + path.string(), logger);
  EXPECT_TRUE(filter.IsActive());
  EXPECT_TRUE(filter.AppliesToBackend(QnnBackendType::HTP));
  EXPECT_FALSE(filter.AppliesToBackend(QnnBackendType::GPU));
  RemoveTempFile(path);
}

// Phase-2 syntax ("OpType[name=...]") is reserved but not implemented in phase 1: it must degrade
// the whole filter to inactive (a loud rejection via WARNING), not silently strip the qualifier.
TEST(QnnUnit_OnnxOpAffinityTest, FromOptionValue_Inline_PhaseTwoNameSyntax_DegradesToInactive) {
  Ort::Logger logger = MakeNullLogger();
  OnnxOpAffinity filter = OnnxOpAffinity::FromOptionValue("htp:exclude:GroupQueryAttention[name=gqa_12]", logger);
  EXPECT_FALSE(filter.IsActive());
}

TEST(QnnUnit_OnnxOpAffinityTest, FromOptionValue_ConfigFile_PhaseTwoNameSyntax_DegradesToInactive) {
  Ort::Logger logger = MakeNullLogger();
  const std::filesystem::path path = WriteTempConfig(
      "qnn_onnx_op_affinity_phase2_name.json",
      R"({ "mode": "exclude", "op_types": ["GroupQueryAttention[name=gqa_12]"] })");
  OnnxOpAffinity filter = OnnxOpAffinity::FromOptionValue("@" + path.string(), logger);
  EXPECT_FALSE(filter.IsActive());
  RemoveTempFile(path);
}

// An unknown backend name in a config "backend" key is kept as a scope, so the filter is
// configured but applies to no real backend -- FromOptionValue warns (suppressed here) and does not
// crash. (Inline, an unrecognized "word:" prefix is instead absorbed into the op-type list, same as
// a UDO colon, so this typo path is only reachable via the explicit config "backend" key.)
TEST(QnnUnit_OnnxOpAffinityTest, FromOptionValue_UnknownBackendScope_AppliesToNothing) {
  Ort::Logger logger = MakeNullLogger();
  const std::filesystem::path path = WriteTempConfig(
      "qnn_onnx_op_affinity_unknown_backend.json",
      R"({ "backend": "hpt", "mode": "exclude", "op_types": ["Softmax"] })");  // "hpt" typo
  OnnxOpAffinity filter = OnnxOpAffinity::FromOptionValue("@" + path.string(), logger);
  EXPECT_FALSE(filter.AppliesToBackend(QnnBackendType::HTP));
  EXPECT_FALSE(filter.AppliesToBackend(QnnBackendType::GPU));
  RemoveTempFile(path);
}

// ============================================================
// OnnxOpAffinity::WarnUnmatchedEntries -- must not crash with the null logger;
// exercises both the matched (no warning) and unmatched (warning) branches.
// ============================================================

TEST_F(QnnUnit_OnnxOpAffinityShouldFilterOffTest, WarnUnmatchedEntries_RunsAfterMatching) {
  Ort::Logger logger = MakeNullLogger();
  OnnxOpAffinity filter("exclude:Softmax,Typoo");
  // "Softmax" matches a node; "Typoo" never does.
  filter.ShouldFilterOff(MakeNodeUnit("Softmax"), kHtp);
  filter.WarnUnmatchedEntries(kHtp, logger);  // "Typoo" -> WARNING (suppressed by null logger)
  SUCCEED();
}

// A backend-scoped filter used on a different-backend session logs a single INFO and does not
// emit per-entry typo warnings (nothing matched because of the backend, not a typo). Just exercise
// the path with the null logger to confirm it does not crash and takes the early-return branch.
TEST_F(QnnUnit_OnnxOpAffinityShouldFilterOffTest, WarnUnmatchedEntries_BackendMismatch_NoCrash) {
  Ort::Logger logger = MakeNullLogger();
  OnnxOpAffinity filter("htp:exclude:Softmax");
  // Session runs GPU; nothing was ever matched. Must not warn about "Softmax" as a typo.
  filter.WarnUnmatchedEntries(QnnBackendType::GPU, logger);
  SUCCEED();
}

// GetSupportedNodes (and thus WarnUnmatchedEntries) runs multiple times per session in practice --
// ORT's partitioner calls GetCapabilityImpl twice per graph, and once per SoC in a multi-SoC session.
// WarnUnmatchedEntries() must clear matched_op_types_ so a match recorded in one call does not leak
// into the next call's typo check, and conversely an entry that only matches in a later call must not
// be penalized for having gone unmatched in an earlier one. Exercised here as a no-crash smoke test
// (this test file has no log-capturing infra to assert on which specific entries warn); the class
// invariant itself is documented on WarnUnmatchedEntries() in qnn_onnx_op_affinity.h.
TEST_F(QnnUnit_OnnxOpAffinityShouldFilterOffTest, WarnUnmatchedEntries_AcrossMultipleGetSupportedNodesCalls) {
  Ort::Logger logger = MakeNullLogger();
  OnnxOpAffinity filter("exclude:Softmax,Conv");

  // First "GetSupportedNodes" pass: only Softmax appears in this subgraph/SoC.
  filter.ShouldFilterOff(MakeNodeUnit("Softmax"), kHtp);
  filter.WarnUnmatchedEntries(kHtp, logger);  // "Conv" unmatched here -> WARNING (suppressed)

  // Second pass: only Conv appears. Must not still think Softmax is unmatched (it isn't checked
  // again since it already logged), and must correctly find Conv now matched -> no WARNING for it.
  filter.ShouldFilterOff(MakeNodeUnit("Conv"), kHtp);
  filter.WarnUnmatchedEntries(kHtp, logger);
  SUCCEED();
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
