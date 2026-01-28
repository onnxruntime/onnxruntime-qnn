// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include <stdint.h>
#include <unordered_map>
#ifdef _WIN32
#include <windows.h>
#endif

#include "core/providers/qnn/soc_utils.h"

namespace onnxruntime {
namespace qnn {
namespace soc {

// QNN-EP COPY START
// Below implementations are directly copied from QNN SDK with few modifications.

#define MAX_FADT_PPTT_SIZE 65536
#define LEVEL_ID(LV1, LV2) ((LV1 << 32) | (LV2))

// Note that the table is intentionally kept compact as Makena is the only expected usage.
// (level1_ID | level2_ID), SOC_ID
static std::unordered_map<uint64_t, int> pptt_mappings = {
    {LEVEL_ID(113ULL, 449ULL), 435},  // Makena
};

int GetSocId() {
#ifdef _WIN32
  static int cachedSocId = -1;
  if (cachedSocId == -1) {
    DWORD bufsize = 0;
    int ret = 0;
    PPPTT pptt;
    FADT* fadt = NULL;
    BYTE* buf = NULL;

    buf = (BYTE*)malloc(MAX_FADT_PPTT_SIZE);
    if (!buf) {
      return 0;
    }

    // try FADT fitst, since this is only valid way on 0x8180
    ret = GetSystemFirmwareTable('ACPI', 'PCAF', 0, 0);
    if (!ret) {
      free(buf);
      return 0;
    }

    bufsize = ret;
    ret = GetSystemFirmwareTable('ACPI', 'PCAF', buf, bufsize);
    if (!ret) {
      free(buf);
      return 0;
    }

    fadt = (FADT*)buf;
    // since 0x8180 is the only valid chip family for FADT method
    // check it here and don't bother to create a mapping table
    if (fadt->Header.OEMRevision == 0x8180) {
      free(buf);
      return 405;
    }

    // start to try newer approach, level 1 ID, level 2 ID in PPTT
    ret = GetSystemFirmwareTable('ACPI', 'TTPP', 0, 0);
    if (!ret) {
      free(buf);
      return 0;
    }

    bufsize = ret;
    ret = GetSystemFirmwareTable('ACPI', 'TTPP', buf, bufsize);
    if (!ret) {
      free(buf);
      return 0;
    }

    pptt = (PPPTT)buf;
    uint64_t key = 0;
    for (uint32_t i = 0; i < pptt->Header.Length; i++) {
      PPROC_TOPOLOGY_NODE ptn = (PPROC_TOPOLOGY_NODE)((BYTE*)&(pptt->HeirarchyNodes[0]) + i);
      // According to ACPI spec, type = 2 is the PPTT_ID_TABLE_TYPE
      if (ptn->Type == 2) {
        key = (ptn->IdNode.Level1 << 32) | (ptn->IdNode.Level2);
        break;
      }
    }
    free(buf);
    if (key == 0) {
      return 0;
    }

    auto it = pptt_mappings.find(key);
    if (it != pptt_mappings.end()) {
      cachedSocId = it->second;
    } else {
      cachedSocId = 0;
    }
  }

  return cachedSocId;
#else
  return 0;
#endif
}

// QNN-EP COPY END

}  // namespace soc
}  // namespace qnn
}  // namespace onnxruntime
