# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import pytest
from qdc_helpers import TestBase


@pytest.fixture(scope="session", autouse=True)
def qdc_device_session():
    base = TestBase()
    base.prepare_ort_tests()
    yield
    base.copy_logs()
