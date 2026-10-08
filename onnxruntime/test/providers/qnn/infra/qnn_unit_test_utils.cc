// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "stub_backend_manager.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include "gtest/gtest.h"

// Public entry point from qnn_provider_factory.cc (extern "C" linkage). Declared
// locally rather than via a shared header -- qnn_provider_factory_test.cc
// declares the identical signature the same way.
extern "C" {
OrtStatus* CreateEpFactories(const char* registration_name,
                             const OrtApiBase* ort_api_base,
                             const OrtLogger* default_logger,
                             OrtEpFactory** factories,
                             size_t max_factories,
                             size_t* num_factories);
}

namespace onnxruntime {
namespace test {
namespace {

// QnnBackendManager::ReleaseResources() (and other EP-internal code) uses
// Ort::Status, whose move-assignment calls OrtRelease() -> Ort::GetApi() ->
// detail::Global::Api(). That function's cached OrtApi* is a function-local
// static defined inline in the vendored onnxruntime_cxx_api.h header.
//
// In the class-level-UT build the EP is compiled as its own SHARED library
// (onnxruntime_providers_qnn.dll/.so) that onnxruntime_provider_test merely
// links against (cmake/onnxruntime_unittests.cmake:
// target_link_libraries(onnxruntime_provider_test PRIVATE
// onnxruntime_providers_qnn)) -- it is NOT compiled directly into the test
// binary. Header-defined inline statics are not shared across a DLL/.so
// boundary: the EP library gets its own copy of detail::Global::Api()'s cache,
// independent of whatever copy exists inside onnxruntime_provider_test.exe
// itself. So seeding Ort::InitApi() from test-side code (anything compiled
// into the test executable, including this file or StubApiEnv) has no effect
// on the EP library's copy.
//
// QnnEpFactory's constructor -- reached via the real, always-exported
// CreateEpFactories() entry point -- is the EP's own intended way to seed that
// cache: qnn_provider_factory.cc calls Ort::InitApi(ort_api) once it has
// resolved a real OrtApi* from the host. That call executes as EP-library
// code, so it seeds the EP library's copy. Any test that builds a
// QnnBackendManager directly via Create()/StubBackendManager -- bypassing
// QnnEpFactory entirely (qnn_backend_manager_test.cc,
// qnn_backend_system_dlc_plugin_test.cc, qnn_ep_profiler_test.cc,
// qnn_backend_profiling_manager_test.cc, qnn_model_test.cc,
// onnx_ctx_model_helper_test.cc, backend_contexts.h) -- never runs that ctor,
// so the EP library's cache is left null. The first Ort::Status
// destroyed/reassigned during that manager's teardown
// (QnnBackendManager::ReleaseResources(), e.g.
// `result = ...ReleaseProfileHandle();`) then dereferences a null OrtApi* and
// crashes.
//
// Fix: force the EP library's own copy of the cache to be seeded exactly once,
// before any test runs, by calling the real CreateEpFactories() entry point
// with the REAL OrtApiBase (OrtGetApiBase()). max_factories = 0 deliberately
// stops CreateEpFactories right after its Ort::InitApi(ort_api) call
// (qnn_provider_factory.cc) and before it would touch
// OrtLoggingManager::SetDefaultLogger or construct a QnnEpFactory -- this
// mirrors qnn_provider_factory_test.cc's own
// CreateEpFactories_MaxFactoriesZero_ReturnsInvalidArgument test, whose
// comment already documents that seeding with the REAL api table this way is
// safe ("no pollution") for every other test sharing the process.
//
// Registered as a gtest Environment (rather than called from
// StubBackendManager/StubApiEnv) so it runs exactly once, before any test in
// the binary, regardless of which test file/test case happens to run first --
// a per-call-site seed would have to be duplicated at every site listed above
// and would be easy to miss when a new one is added.
class QnnEpApiSeedEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    OrtEpFactory* factories[1] = {nullptr};
    size_t num_factories = 0;
    OrtStatus* status = CreateEpFactories("qnn_unit_test_ep_api_seed", OrtGetApiBase(),
                                          /*default_logger*/ nullptr, factories,
                                          /*max_factories*/ 0, &num_factories);
    if (status != nullptr) {
      OrtGetApiBase()->GetApi(ORT_API_VERSION)->ReleaseStatus(status);
    }
  }
};

::testing::Environment* const g_qnn_ep_api_seed_env =
    ::testing::AddGlobalTestEnvironment(new QnnEpApiSeedEnvironment());

}  // namespace

namespace {

// Friend-injection helper: instantiating PrivateMember<Tag, Member> injects a
// GetPrivateMemberPtr(Tag) overload into the surrounding namespace that returns
// Member. The overload is findable only by ADL on the tag type.
//
// Instantiating this template does NOT by itself emit GetPrivateMemberPtr's
// body — an injected friend is not a member of the class template for
// instantiation purposes, so the explicit instantiations below only *declare*
// it. The body is emitted where the friend is odr-used; see the accessor
// definitions further down, which are what put it in this object file.
template <typename Tag, typename Tag::type Member>
struct PrivateMember {
  friend typename Tag::type GetPrivateMemberPtr(Tag) { return Member; }
};

// Defines a tag type whose GetPrivateMemberPtr() overload yields a
// pointer-to-member for the named private QnnBackendManager member, and
// explicitly instantiates PrivateMember to inject that overload.
// MemberType must not contain a comma.
//
// Naming a private member here is legal, not a trick played on the compiler:
// the usual access checking does not apply to names in the template-argument
// list of an *explicit instantiation* (C++17 [temp.spec]/6). So this is a
// standard-sanctioned member-pointer grab rather than a `#define private
// public` ODR violation, and core/providers/qnn/ needs no test-only hooks.
//
// These instantiations must stay in a .cc and out of qnn_unit_test_utils.h. An
// explicit instantiation definition of a given specialization may appear at
// most once in a program ([temp.explicit]/13); repeating it across translation
// units is IFNDR. The header is included by every unit/*_test.cc, so keeping
// them there relied on GCC/Clang leniency (the injected friend lands in a
// weak/comdat section, so duplicates merge). Here each is instantiated once.
//
// Maintenance: the tags name QnnBackendManager's private members directly, so a
// rename or retype in core/providers/qnn/ breaks the build on these lines.
#define QNN_UT_DEFINE_BACKEND_MANAGER_MEMBER_TAG(TagName, MemberType, member_name) \
  struct TagName {                                                                 \
    using type = MemberType qnn::QnnBackendManager::*;                             \
    friend type GetPrivateMemberPtr(TagName);                                      \
  };                                                                               \
  template struct PrivateMember<TagName, &qnn::QnnBackendManager::member_name>

QNN_UT_DEFINE_BACKEND_MANAGER_MEMBER_TAG(QnnInterfaceTag, QNN_INTERFACE_VER_TYPE, qnn_interface_);
QNN_UT_DEFINE_BACKEND_MANAGER_MEMBER_TAG(QnnSystemInterfaceTag, QNN_SYSTEM_INTERFACE_VER_TYPE, qnn_sys_interface_);
QNN_UT_DEFINE_BACKEND_MANAGER_MEMBER_TAG(QnnBackendHandleTag, Qnn_BackendHandle_t, backend_handle_);
QNN_UT_DEFINE_BACKEND_MANAGER_MEMBER_TAG(QnnValidatorInterfaceTag, QNN_INTERFACE_VER_TYPE,
                                         qnn_validator_interface_);
QNN_UT_DEFINE_BACKEND_MANAGER_MEMBER_TAG(QnnValidatorBackendHandleTag, Qnn_BackendHandle_t,
                                         validator_backend_handle_);
QNN_UT_DEFINE_BACKEND_MANAGER_MEMBER_TAG(QnnBackendTypeTag, qnn::QnnBackendType, qnn_backend_type_);
QNN_UT_DEFINE_BACKEND_MANAGER_MEMBER_TAG(QnnHtpArchTag, QnnHtpDevice_Arch_t, htp_arch_internal_);

}  // namespace

// StubBackendManager's private-member accessors, declared in the header.
//
// They are defined here, not inline in the class, for two reasons. Each call
// odr-uses the injected friend, which is what causes its body to be emitted in
// this object file — the explicit instantiations above only declare it. And
// defining them out-of-line leaves every unit/*_test.cc with an ordinary
// external call that the linker resolves against these definitions, so no
// other translation unit needs the machinery above.
//
// GetPrivateMemberPtr is reachable only by ADL on its tag type, so do not call
// it from a scope that declares a member of the same name: ordinary unqualified
// lookup finding a class member suppresses ADL ([basic.lookup.argdep]/1) and
// the call fails to compile.
//
// To expose another private member: add a tag + instantiation above, add an
// accessor definition here, and declare it on StubBackendManager in the header.
QNN_INTERFACE_VER_TYPE& StubBackendManager::QnnInterface() {
  return (*manager_).*GetPrivateMemberPtr(QnnInterfaceTag{});
}

QNN_SYSTEM_INTERFACE_VER_TYPE& StubBackendManager::SystemInterface() {
  return (*manager_).*GetPrivateMemberPtr(QnnSystemInterfaceTag{});
}

Qnn_BackendHandle_t& StubBackendManager::BackendHandle() {
  return (*manager_).*GetPrivateMemberPtr(QnnBackendHandleTag{});
}

QNN_INTERFACE_VER_TYPE& StubBackendManager::ValidatorInterface() {
  return (*manager_).*GetPrivateMemberPtr(QnnValidatorInterfaceTag{});
}

Qnn_BackendHandle_t& StubBackendManager::ValidatorBackendHandle() {
  return (*manager_).*GetPrivateMemberPtr(QnnValidatorBackendHandleTag{});
}

qnn::QnnBackendType& StubBackendManager::BackendType() {
  return (*manager_).*GetPrivateMemberPtr(QnnBackendTypeTag{});
}

QnnHtpDevice_Arch_t& StubBackendManager::HtpArch() {
  return (*manager_).*GetPrivateMemberPtr(QnnHtpArchTag{});
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
