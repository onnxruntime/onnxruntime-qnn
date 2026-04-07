// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Include header with declaration of the function to create the execution provider factory for all enabled
// execution providers.
//
// The functions are typically implemented in
// onnxruntime/core/providers/<provider name>/<provider name>_provider_factory.cc.
//
// For execution providers that are built as separate libraries (CUDA, TensorRT, MIGraphX, DNNL, OpenVINO)
// the functions are implemented in provider_bridge_ort.cc.

#include "core/providers/cpu/cpu_provider_factory_creator.h"
