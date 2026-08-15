// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include <climits>
#include <stdint.h>
#include <string>
#include <string_view>
#include <unordered_map>
#ifdef _WIN32
#include <windows.h>
#endif
#if !defined(_WIN32) && defined(__aarch64__)
#include <cstring>
#include <dirent.h>
#endif
#if defined(__ANDROID__)
#include <sys/system_properties.h>
#endif

#include "core/providers/qnn/soc_utils.h"

namespace onnxruntime {
namespace qnn {
namespace soc {
namespace {

// QNN-EP COPY START
// Below implementations are directly copied from QNN SDK with few modifications.

#ifdef _WIN32
// Use 1-byte packing to match the original header
#pragma pack(push, 1)

//
// Description Header structure that appears at the beginning of each ACPI table
//
typedef struct _DESCRIPTION_HEADER {
  ULONG Signature;     // Signature used to identify the type of table
  ULONG Length;        // Length of entire table including the DESCRIPTION_HEADER
  UCHAR Revision;      // Minor version of ACPI spec to which this table conforms
  UCHAR Checksum;      // Sum of all bytes in the entire TABLE should = 0
  CHAR OEMID[6];       // String that uniquely ID's the OEM
  CHAR OEMTableID[8];  // String that uniquely ID's this table
  ULONG OEMRevision;   // OEM supplied table revision number
  CHAR CreatorID[4];   // Vendor ID of utility which created this table
  ULONG CreatorRev;    // Revision of utility that created the table
} DESCRIPTION_HEADER, *PDESCRIPTION_HEADER;

//
// Processor Properties Topology Table (PPTT) structures
//
typedef struct _PROC_TOPOLOGY_NODE {
  UCHAR Type;
  UCHAR Length;
  UCHAR Reserved[2];
  union {
    struct {
      union {
        struct {
          ULONG PhysicalPackage : 1;
          ULONG ACPIProcessorIdValid : 1;
          ULONG Reserved : 30;
        };
        ULONG AsULONG;
      } Flags;
      ULONG Parent;
      ULONG ACPIProcessorId;
      ULONG NumberPrivateResources;
      ULONG PrivateResources[1];  // Variable length array
    } HeirarchyNode;
    struct {
      union {
        struct {
          ULONG SizeValid : 1;
          ULONG SetsValid : 1;
          ULONG AssociativityValid : 1;
          ULONG AllocationTypeValid : 1;
          ULONG CacheTypeValid : 1;
          ULONG WritePolicyValid : 1;
          ULONG LineSizeValid : 1;
          ULONG Reserved : 25;
        };
        ULONG AsULONG;
      } Flags;
      ULONG NextLevelCacheOffset;
      ULONG Size;
      ULONG Sets;
      UCHAR Associativity;
      union {
        struct {
          UCHAR ReadAllocate : 1;
          UCHAR WriteAllocate : 1;
          UCHAR CacheType : 2;
          UCHAR WritePolicy : 1;
          UCHAR Reserved : 3;
        };
        UCHAR AsUCHAR;
      } Attributes;
      USHORT LineSize;
    } CacheNode;
    struct {
      ULONG Vendor;
      ULONG64 Level1;
      ULONG64 Level2;
      USHORT Major;
      USHORT Minor;
      USHORT Spin;
    } IdNode;
  };
} PROC_TOPOLOGY_NODE, *PPROC_TOPOLOGY_NODE;

typedef struct _PPTT {
  DESCRIPTION_HEADER Header;
  PROC_TOPOLOGY_NODE HierarchyNodes[1];  // Variable length array
} PPTT, *PPPTT;

// Restore default packing
#pragma pack(pop)
#endif

#define MAX_FADT_PPTT_SIZE 65536
#define LEVEL_ID(LV1, LV2) ((LV1 << 32) | (LV2))

// Note that the table is intentionally kept compact as Makena is the only expected usage.
// (level1_ID | level2_ID), SOC_ID
static std::unordered_map<uint64_t, int> pptt_mappings = {
    {LEVEL_ID(113ULL, 449ULL), 435},  // Makena
};

int getSocId() {
#ifdef _WIN32
  int socId = -1;
  DWORD bufsize = 0;
  int ret = 0;
  PPPTT pptt;
  BYTE* buf = NULL;

  buf = (BYTE*)malloc(MAX_FADT_PPTT_SIZE);
  if (!buf) {
    return 0;
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
    PPROC_TOPOLOGY_NODE ptn = (PPROC_TOPOLOGY_NODE)((BYTE*)&(pptt->HierarchyNodes[0]) + i);
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
    socId = it->second;
  } else {
    socId = 0;
  }

  return socId;
#else
  return 0;
#endif
}

// QNN-EP COPY END
}  // namespace

int GetSocId() {
  static int cached_soc_id = getSocId();
  return cached_soc_id;
}

bool HasFastRpcCdspDevice() {
#if !defined(_WIN32) && defined(__aarch64__)
#if defined(__ANDROID__)
  char manufacturer[PROP_VALUE_MAX] = {};
  __system_property_get("ro.soc.manufacturer", manufacturer);
  return strncasecmp(manufacturer, "QTI", 3) == 0;
#endif
  DIR* d = opendir("/dev");
  if (!d) {
    return false;
  }
  bool found = false;
  while (dirent* e = readdir(d)) {
    if (std::strncmp(e->d_name, "fastrpc-cdsp", 12) == 0) {
      found = true;
      break;
    }
  }
  closedir(d);
  return found;
#else
  return false;
#endif
}

uint32_t SocModelFromName(std::string_view name) {
  // Static lookup table with UPPERCASE keys matching QNN SDK Qnn_SocModel_t enum values.
  // All named entries from QNN/QnnTypes.h are included (value 0 = UNKNOWN is the default).
  static const std::unordered_map<std::string, uint32_t> kSocModelNameMap = {
      {"SDM845", 1},
      {"SDM835", 2},
      {"SDM821", 3},
      {"SDM820", 4},
      {"SDM801", 5},
      {"SDM670", 6},
      {"SDM660", 7},
      {"SDM652", 8},
      {"SDM636", 9},
      {"SDM630", 10},
      {"SDM625", 11},
      {"SDM855", 12},
      {"SDM710", 13},
      {"SDM632", 15},
      {"SM6150", 16},
      {"SM7150", 17},
      {"QCS405", 18},
      {"SM6125", 19},
      {"QCS403", 20},
      {"SDM865", 21},
      {"IPQ6018", 23},
      {"IPQ6028", 24},
      {"SM7250", 25},
      {"SA8195", 26},
      {"SM6250", 27},
      {"SM4250", 28},
      {"SM6350", 29},
      {"SM8350", 30},
      {"SM4350", 31},
      {"SM7350", 32},
      {"QCS410", 33},
      {"SM8325", 34},
      {"SM7325", 35},
      {"SM8450", 36},
      {"SC8280X", 37},
      {"SM7315", 38},
      {"SA8295", 39},
      {"SM6225", 40},
      {"SM7450", 41},
      {"SM8475", 42},
      {"SM8550", 43},
      {"SXR1230P", 45},
      {"SSG2115P", 46},
      {"STP6225P", 47},
      {"QCS6125", 48},
      {"QRB4210", 49},
      {"SM6450", 50},
      {"QCS7230", 51},
      {"SA8255", 52},
      {"SXR2230P", 53},
      {"SM7475", 54},
      {"SM4375", 55},
      {"QCM4325", 56},
      {"SM8650", 57},
      {"SSG2125P", 58},
      {"SM4450", 59},
      {"SC8380XP", 60},
      {"SM7435", 61},
      {"SA8540", 62},
      {"AIC100", 63},
      {"SM7550", 64},
      {"SM6450Q", 65},
      {"QCS8550", 66},
      {"SA8620P", 67},
      {"SM8635", 68},
      {"SM8750", 69},
      {"SM7675", 70},
      {"SM4635", 71},
      {"SA8797", 72},
      {"SM7635", 73},
      {"SM6650", 74},
      {"SXR2330P", 75},
      {"SM6475", 76},
      {"QCS9100", 77},
      {"QCM6690", 78},
      {"IPQ9574", 79},
      {"IPQ5404", 80},
      {"IPQ5424", 81},
      {"QCS8300", 82},
      {"QCS2290", 83},
      {"SA525M", 84},
      {"SM8735", 85},
      {"SM7750", 86},
      {"SM8850", 87},
      {"DYNAMIC_SDM", static_cast<uint32_t>(INT_MAX)},
  };

  // Uppercase input for case-insensitive lookup (ASCII-only, no locale dependency).
  std::string upper;
  upper.reserve(name.size());
  for (char c : name) {
    upper += static_cast<char>(
        (c >= 'a' && c <= 'z') ? (c - 'a' + 'A') : c);
  }

  auto it = kSocModelNameMap.find(upper);
  if (it != kSocModelNameMap.end()) {
    return it->second;
  }
  return 0;
}

}  // namespace soc
}  // namespace qnn
}  // namespace onnxruntime
