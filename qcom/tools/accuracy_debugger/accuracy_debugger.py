# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Entry point for Accuracy Debugger tool (the ORT/QNN-EP snooper CLI)."""

from __future__ import annotations

from ort_qnn_snooper import main

if __name__ == "__main__":
    raise SystemExit(main())
