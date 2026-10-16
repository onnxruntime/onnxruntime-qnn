// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

// Windows COM headers define `interface` as a macro. QNN tests use it as a regular C++ identifier.
#if defined(_WIN32) && defined(interface)
#undef interface
#endif
