// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "qnn_unit_test_utils.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

namespace onnxruntime {
namespace test {
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
