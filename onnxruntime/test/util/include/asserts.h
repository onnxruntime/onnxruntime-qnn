// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_cxx_api.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

// helpers to run a function and check the status, outputting any error if it fails.
// These macros work with Ort::Status (public API).
#define ASSERT_STATUS_OK(function)                                    \
  do {                                                                \
    Ort::Status _tmp_status = (function);                             \
    ASSERT_TRUE(_tmp_status.IsOK()) << _tmp_status.GetErrorMessage(); \
  } while (false)

#define EXPECT_STATUS_OK(function)                                    \
  do {                                                                \
    Ort::Status _tmp_status = (function);                             \
    EXPECT_TRUE(_tmp_status.IsOK()) << _tmp_status.GetErrorMessage(); \
  } while (false)

#define ASSERT_STATUS_NOT_OK(function)    \
  do {                                    \
    Ort::Status _tmp_status = (function); \
    ASSERT_FALSE(_tmp_status.IsOK());     \
  } while (false)

#define EXPECT_STATUS_NOT_OK(function)    \
  do {                                    \
    Ort::Status _tmp_status = (function); \
    EXPECT_FALSE(_tmp_status.IsOK());     \
  } while (false)

#define ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(function, msg)                 \
  do {                                                                     \
    Ort::Status _tmp_status = (function);                                  \
    ASSERT_FALSE(_tmp_status.IsOK());                                      \
    ASSERT_THAT(_tmp_status.GetErrorMessage(), ::testing::HasSubstr(msg)); \
  } while (false)

#define EXPECT_STATUS_NOT_OK_AND_HAS_SUBSTR(function, msg)                 \
  do {                                                                     \
    Ort::Status _tmp_status = (function);                                  \
    EXPECT_FALSE(_tmp_status.IsOK());                                      \
    EXPECT_THAT(_tmp_status.GetErrorMessage(), ::testing::HasSubstr(msg)); \
  } while (false)
