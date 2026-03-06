#!/bin/bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

REPO_ROOT=$(git rev-parse --show-toplevel)

python3 -m venv uplevel_venv
source uplevel_venv/bin/activate
pip install uv
uv pip install -r ${REPO_ROOT}/qcom/scripts/upleveling/requirements.txt
