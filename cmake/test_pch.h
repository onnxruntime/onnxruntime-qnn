// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Test framework headers (highest compilation time impact)
#include "gtest/gtest.h"
#include "gtest/gtest-assertion-result.h"
#include "gtest/gtest-message.h"
#include "gtest/internal/gtest-port.h"

// ONNX and Protocol Buffer headers
#include "onnx/defs/schema.h"

// Windows-specific headers (if applicable)
#ifdef _WIN32
#include <windows.h>
#endif
