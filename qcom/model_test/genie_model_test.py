# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
from typing import cast, get_args

import pytest
from model_test import BackendT, ModelTestCase, ModelTestDef, ModelTestSuite

GENIE_MODEL_ROOTS = [Path(p) for p in os.getenv("ORT_MODEL_ZOO_TEST_ROOTS", "").split(os.pathsep) if len(p) > 0]
GENIE_BACKEND = cast(BackendT, os.getenv("ORT_MODEL_ZOO_BACKEND", "genie"))
assert GENIE_BACKEND in get_args(BackendT)

for genie_model_root in GENIE_MODEL_ROOTS:
    TEST_DEFS = list(
        ModelTestSuite(
            genie_model_root,
            backend_type=GENIE_BACKEND,
            rtol=None,
            atol=None,
            cosine_similarity=None,
            enable_context=False,
            enable_cpu_fallback=False,
        ).tests
    )

    TEST_IDS = [str(st) for st in TEST_DEFS]

    @pytest.mark.parametrize("test_def", TEST_DEFS, ids=TEST_IDS)
    def test_genie_models(test_def: ModelTestDef) -> None:
        ModelTestCase(test_def).run_and_check_nonempty()
