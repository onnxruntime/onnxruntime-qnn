# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""pytest configuration: make qcom/scripts/all importable from this sub-directory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
