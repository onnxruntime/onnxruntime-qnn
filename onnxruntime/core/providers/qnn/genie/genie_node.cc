// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/genie/genie_node.h"

// Implement the custom deleter
void GenieNodeStateDeleter::operator()(GenieNodeState* st) {
  if (!st) return;

  auto* api = st->api;
  if (api) {
    if (st->node) api->Node_free(st->node);
    if (st->genie_logger && api->Log_free) api->Log_free(st->genie_logger);
    if (st->config) api->NodeConfig_free(st->config);
#if (GENIE_API_VERSION_MAJOR > 1) || (GENIE_API_VERSION_MAJOR == 1 && GENIE_API_VERSION_MINOR >= 17)
    if (st->dlc_handle) api->Dlc_free(st->dlc_handle);
    if (st->dlc_config_handle) api->DlcConfig_free(st->dlc_config_handle);
#endif
  }
  delete st;
}
