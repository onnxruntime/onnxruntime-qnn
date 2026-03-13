// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#ifdef _WIN32
#include <windows.h>
#endif

namespace onnxruntime {
namespace qnn {
namespace soc {

// QNN-EP COPY START
// Below implementations are directly copied from QNN SDK.

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
// Fixed ACPI Description Table (FADT)
//
typedef struct _FADT {
  DESCRIPTION_HEADER Header;
  ULONG facs;                   // Physical address of the Firmware ACPI Control Structure
  ULONG dsdt;                   // Physical address of the Differentiated System Description Table
  UCHAR int_model;              // System's Interrupt mode
  UCHAR pm_profile;             // System's preferred power profile
  USHORT sci_int_vector;        // Vector of SCI interrupt
  ULONG smi_cmd_io_port;        // Address in System I/O Space of the SMI Command port
  UCHAR acpi_on_value;          // Value out'd to smi_cmd_port to activate ACPI
  UCHAR acpi_off_value;         // Value out'd to smi_cmd_port to deactivate ACPI
  UCHAR s4bios_req;             // Value to write to SMI_CMD to enter the S4 state
  UCHAR pstate_control;         // Value to write to SMI_CMD to assume control of
                                // processor performance states
  ULONG pm1a_evt_blk_io_port;   // Address in System I/O Space of the PM1a_EVT_BLK
                                // register block
  ULONG pm1b_evt_blk_io_port;   // Address in System I/O Space of the PM1b_EVT_BLK
                                // register block
  ULONG pm1a_ctrl_blk_io_port;  // Address in System I/O Space of the
                                // PM1a_CNT_BLK register block
  ULONG pm1b_ctrl_blk_io_port;  // Address in System I/O Space of the
                                // PM1b_CNT_BLK register block
  ULONG pm2_ctrl_blk_io_port;   // Address in System I/O Space of the PM2_CNT_BLK
                                // register block
  ULONG pm_tmr_blk_io_port;     // Address in System I/O Space of the PM_TMR
                                // register block
  ULONG
  gp0_blk_io_port;  // Address in System I/O Space of the GP0 register block
  ULONG
  gp1_blk_io_port;       // Address in System I/O Space of the GP1 register block
  UCHAR pm1_evt_len;     // Number of bytes decoded for PM1_BLK
  UCHAR pm1_ctrl_len;    // Number of bytes decoded for PM1_CNT
  UCHAR pm2_ctrl_len;    // Number of bytes decoded for PM1a_CNT
  UCHAR pm_tmr_len;      // Number of bytes decoded for PM_TMR
  UCHAR gp0_blk_len;     // Number of bytes decoded for GP0_BLK
  UCHAR gp1_blk_len;     // Number of bytes decoded for GP1_BLK
  UCHAR gp1_base;        // Index at which GP1 based events start
  UCHAR cstate_control;  // Value to write to SMI_CMD to assume control of _CST
                         // states
  USHORT lvl2_latency;   // Worst case latency to enter/leave C2 state
  USHORT lvl3_latency;   // Worst case latency to enter/leave C3 state
  USHORT flush_size;     // Size of memory to flush
  USHORT flush_stride;   // Memory stride width for flushing
  UCHAR duty_offset;     // Index of duty cycle setting
  UCHAR duty_width;      // Bit width of duty cycle setting
  UCHAR day_alarm_index;
  UCHAR month_alarm_index;
  UCHAR century_alarm_index;
  USHORT boot_arch;
  UCHAR reserved3[1];
  ULONG flags;
  // Additional fields omitted as they're not needed for the code
} FADT, *PFADT;

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
  PROC_TOPOLOGY_NODE HeirarchyNodes[1];  // Variable length array
} PPTT, *PPPTT;

// Restore default packing
#pragma pack(pop)
#endif

int GetSocId();

// QNN-EP COPY END

}  // namespace soc
}  // namespace qnn
}  // namespace onnxruntime
