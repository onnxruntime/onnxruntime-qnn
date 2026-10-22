// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#ifdef USE_QAIRT_API
#include "core/providers/qnn/builder/qairt_profile_adapter.h"

#include "QairtCpp/System/QairtSystemProfile.hpp"
#include <filesystem>
#include <functional>
namespace onnxruntime::qnn {
Ort::Status QairtProfileAdapter::WriteLog(const std::string& path, const std::string& graph,
                                          uint64_t start, uint64_t stop, OrtProfilingOperation operation) {
  try {
    qairt::SystemApi system(manager_->GetSystemLibraryPath().c_str());
    auto header = system.make<qairt::SystemProfileHeaderV1>();
    header.setGraphName(graph);
    header.setStartTime(start);
    header.setStopTime(stop);
    header.setVisibility(qairt::SystemProfileVisibility::Public);
    auto method = qairt::SystemProfileMethodType::AppComposeGraphs;
    if (operation == OrtProfilingOperation::EXECUTE) method = qairt::SystemProfileMethodType::BackendExecute;
    if (operation == OrtProfilingOperation::FINALIZE) method = qairt::SystemProfileMethodType::BackendFinalize;
    if (operation == OrtProfilingOperation::CONTEXT_LOAD) method = qairt::SystemProfileMethodType::BackendCreateFromBinary;
    header.setMethodType(method);
    std::function<qairt::SystemProfileEventV1(QnnProfile_EventId_t)> convert = [&](QnnProfile_EventId_t id) {
      const auto& event = events_.at(id);
      auto converted = system.make<qairt::SystemProfileEventV1>();
      converted.setEventData(event.data);
      for (auto child : event.children) converted.addSubEvent(convert(child));
      return converted;
    };
    auto record = system.make<qairt::SystemProfileDataV1>();
    record.setHeader(header);
    for (auto root : roots_) record.addEvent(convert(root));
    auto data = system.make<qairt::SystemProfileData>();
    data.setProfileDataV1(record);
    std::vector<qairt::SystemProfileData> records;
    records.push_back(std::move(data));
    const auto output = std::filesystem::absolute(path);
    auto file = system.make<qairt::SystemProfileSerializationTargetFile>();
    file.setFileName(output.stem().string() + "_qnn.log");
    file.setFileDirectory(output.parent_path().string());
    auto info = system.make<qairt::SystemProfileSerializationTargetInfo>();
    info.setTargetFile(file);
    auto config = system.make<qairt::SystemProfileSerializationTargetConfiguration>();
    auto target = system.make<qairt::SystemProfileSerializationTarget>(info, config);
    target.serializeEventData(records);
    return Ort::Status();
  } catch (const std::exception& e) {
    return MAKE_EP_FAIL(e.what());
  }
}

Ort::Status QairtProfileAdapter::Create(ProfilingLevel level) {
  try {
    auto profile = manager_->GetBackend().createProfile(
        level == ProfilingLevel::BASIC ? QAIRT_PROFILE_LEVEL_BASIC : QAIRT_PROFILE_LEVEL_DETAILED);
    profile_ = std::make_shared<qairt::Profile>(std::move(profile));
    if (level == ProfilingLevel::OPTRACE) {
      auto config = manager_->GetApi().make<qairt::ProfileConfiguration>();
      config.setEnableOpTrace(true);
      profile_->setConfig(config);
    }
    return Ort::Status();
  } catch (const std::exception& e) {
    Release();
    return MAKE_EP_FAIL(e.what());
  }
}

void QairtProfileAdapter::Release() {
  events_.clear();
  roots_.clear();
  profile_.reset();
}

QnnProfile_EventId_t QairtProfileAdapter::AddEvent(qairt::ProfileEventData&& data) {
  if (data.getEventUnit() == qairt::ProfileEventUnit::Object) {
    try {
      // Some SDKs produce opaque events but do not expose their payload through
      // QAIRT. Keep collecting supported timing events and report the lost payload.
      auto object = data.getBackendOpaqueObject();
      (void)object;
    } catch (const qairt::Exception& e) {
      if (e != qairt::CommonError::NotSupported) throw;
      ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(), ORT_LOGGING_LEVEL_WARNING,
                  "QAIRT does not expose this opaque profiling payload; omitting it from profiling output.");
      return 0;
    }
  }
  const auto id = static_cast<QnnProfile_EventId_t>(events_.size() + 1);
  auto& event = events_.emplace(id, Event{std::move(data), {}, {}, {}, {}, {}}).first->second;
  event.identifier = event.data.getIdentifier();
  for (auto& child : profile_->getSubEvents(event.data)) {
    const auto child_id = AddEvent(std::move(child));
    if (child_id != 0) event.children.push_back(child_id);
  }
  return id;
}

Qnn_ErrorHandle_t QairtProfileAdapter::GetEvents(const QnnProfile_EventId_t** ids, uint32_t* count) {
  try {
    events_.clear();
    roots_.clear();
    for (auto& event : profile_->getEvents()) {
      const auto id = AddEvent(std::move(event));
      if (id != 0) roots_.push_back(id);
    }
    *ids = roots_.data();
    *count = static_cast<uint32_t>(roots_.size());
    return QNN_PROFILE_NO_ERROR;
  } catch (const std::exception& e) {
    ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(), ORT_LOGGING_LEVEL_ERROR, e.what());
    return QNN_COMMON_ERROR_GENERAL;
  }
}

Qnn_ErrorHandle_t QairtProfileAdapter::GetSubEvents(QnnProfile_EventId_t id,
                                                    const QnnProfile_EventId_t** ids, uint32_t* count) {
  const auto it = events_.find(id);
  if (it == events_.end()) return QNN_COMMON_ERROR_INVALID_ARGUMENT;
  *ids = it->second.children.data();
  *count = static_cast<uint32_t>(it->second.children.size());
  return QNN_PROFILE_NO_ERROR;
}

Qnn_ErrorHandle_t QairtProfileAdapter::GetExtendedEventData(QnnProfile_EventId_t id,
                                                            QnnProfile_ExtendedEventData_t* out) {
  auto it = events_.find(id);
  if (it == events_.end()) return QNN_COMMON_ERROR_INVALID_ARGUMENT;
  try {
    auto& event = it->second;
    *out = QNN_PROFILE_EXTENDED_EVENT_DATA_INIT;
    auto& value = out->v1;
    value.type = static_cast<QnnProfile_EventType_t>(event.data.getEventType());
    value.unit = static_cast<QnnProfile_EventUnit_t>(event.data.getEventUnit());
    value.identifier = event.identifier.c_str();
    value.timestamp = event.data.supportsExtendedFormat() ? event.data.getTimestamp() : 0;
    if (value.unit == QNN_PROFILE_EVENTUNIT_OBJECT) {
      auto object = event.data.getBackendOpaqueObject();
      auto data = object.getData();
      const auto* bytes = static_cast<const unsigned char*>(data.getData());
      event.opaque_data.clear();
      if (data.getDataSize()) {
        if (bytes == nullptr) return QNN_COMMON_ERROR_INVALID_ARGUMENT;
        event.opaque_data.assign(bytes, bytes + data.getDataSize());
      }
      event.file_name = object.getFileName();
      value.backendOpaqueObject.opaqueObject.data = event.opaque_data.data();
      value.backendOpaqueObject.opaqueObject.len = event.opaque_data.size();
      value.backendOpaqueObject.fileName = event.file_name.c_str();
      return QNN_PROFILE_NO_ERROR;
    }
    const auto& scalar = event.data.getScalar();
    value.value.dataType = static_cast<Qnn_DataType_t>(scalar.getDataType());
    switch (value.value.dataType) {
      case QNN_DATATYPE_INT_8:
        value.value.int8Value = scalar.getInt8Value();
        break;
      case QNN_DATATYPE_INT_16:
        value.value.int16Value = scalar.getInt16Value();
        break;
      case QNN_DATATYPE_INT_32:
        value.value.int32Value = scalar.getInt32Value();
        break;
      case QNN_DATATYPE_INT_64:
        value.value.int64Value = scalar.getInt64Value();
        break;
      case QNN_DATATYPE_UINT_8:
        value.value.uint8Value = scalar.getUint8Value();
        break;
      case QNN_DATATYPE_UINT_16:
        value.value.uint16Value = scalar.getUint16Value();
        break;
      case QNN_DATATYPE_UINT_32:
        value.value.uint32Value = scalar.getUint32Value();
        break;
      case QNN_DATATYPE_UINT_64:
        value.value.uint64Value = scalar.getUint64Value();
        break;
      case QNN_DATATYPE_FLOAT_32:
        value.value.floatValue = scalar.getFloatValue();
        break;
      case QNN_DATATYPE_FLOAT_64:
        value.value.doubleValue = scalar.getDoubleValue();
        break;
      case QNN_DATATYPE_BOOL_8:
        value.value.bool8Value = scalar.getBoolValue();
        break;
      case QNN_DATATYPE_STRING: {
        const auto* text = scalar.getStringValue();
        if (text == nullptr) return QNN_COMMON_ERROR_INVALID_ARGUMENT;
        event.string_value = text;
        value.value.stringValue = event.string_value.c_str();
        break;
      }
      default:
        return QNN_COMMON_ERROR_NOT_SUPPORTED;
    }
    return QNN_PROFILE_NO_ERROR;
  } catch (const std::exception& e) {
    ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(), ORT_LOGGING_LEVEL_ERROR, e.what());
    return QNN_COMMON_ERROR_GENERAL;
  }
}

Qnn_ErrorHandle_t QairtProfileAdapter::GetEventData(QnnProfile_EventId_t id, QnnProfile_EventData_t* out) {
  QnnProfile_ExtendedEventData_t extended = QNN_PROFILE_EXTENDED_EVENT_DATA_INIT;
  const auto result = GetExtendedEventData(id, &extended);
  if (result != QNN_PROFILE_NO_ERROR) return result;
  *out = QNN_PROFILE_EVENT_DATA_INIT;
  out->type = extended.v1.type;
  out->unit = extended.v1.unit;
  out->identifier = extended.v1.identifier;
  if (extended.v1.unit == QNN_PROFILE_EVENTUNIT_OBJECT || extended.v1.value.dataType != QNN_DATATYPE_UINT_64) {
    return QNN_COMMON_ERROR_NOT_SUPPORTED;
  }
  out->value = extended.v1.value.uint64Value;
  return QNN_PROFILE_NO_ERROR;
}

}  // namespace onnxruntime::qnn
#endif
