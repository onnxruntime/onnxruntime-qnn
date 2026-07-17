# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import logging
import os
import tempfile
from pathlib import Path
from typing import cast, get_args

import pytest
from graph_gate import check_gate
from model_test import BackendT, ModelTestCase, ModelTestDef, ModelTestSuite

MODEL_ZOO_ROOTS = [Path(p) for p in os.getenv("ORT_MODEL_ZOO_TEST_ROOTS", "").split(os.pathsep) if len(p) > 0]
MODEL_ZOO_BACKEND = cast(BackendT, os.getenv("ORT_MODEL_ZOO_BACKEND", "htp"))
assert MODEL_ZOO_BACKEND in get_args(BackendT)
MODEL_ZOO_ENABLE_CONTEXT = os.getenv("ORT_MODEL_ZOO_ENABLE_CONTEXT", "1") == "1"
MODEL_ZOO_ENABLE_CPU_FALLBACK = os.getenv("ORT_MODEL_ZOO_ENABLE_CPU_FALLBACK", "0") == "1"

# Graph-diff gating: if golden graphs exist in model_root/goldens/, skip accuracy when graph is unchanged.
ENABLE_GRAPH_GATE = os.getenv("ORT_MODEL_ZOO_DISABLE_GRAPH_GATE", "0") != "1"


def get_xfails(env_var: str) -> dict[str, str]:
    xfails_def = [s.split("=") for s in os.environ.get(env_var, "").split(";") if len(s) != 0]
    assert all(len(xd) == 2 for xd in xfails_def), f"{env_var} must be of format MODEL=REASON[;MODEL=REASON;...]"
    return {xd[0]: xd[1] for xd in xfails_def}


for model_zoo_root in MODEL_ZOO_ROOTS:
    TEST_DEFS = list(
        ModelTestSuite(
            model_zoo_root,
            backend_type=MODEL_ZOO_BACKEND,
            rtol=None,
            atol=None,
            cosine_similarity=None,
            enable_context=MODEL_ZOO_ENABLE_CONTEXT,
            enable_cpu_fallback=MODEL_ZOO_ENABLE_CPU_FALLBACK,
        ).tests
    )

    TEST_IDS = [str(st) for st in TEST_DEFS]

    @pytest.mark.parametrize("test_def", TEST_DEFS, ids=TEST_IDS)
    def test_models(test_def: ModelTestDef) -> None:
        xfails = get_xfails("ORT_MODEL_ZOO_TEST_XFAILS")
        if test_def.model_root.name in xfails:
            pytest.xfail(xfails[test_def.model_root.name])

        # Graph-diff gate: dump graph and compare with golden to decide
        # whether full accuracy verification is needed.
        if ENABLE_GRAPH_GATE and (test_def.model_root / "goldens").is_dir():
            with tempfile.TemporaryDirectory(prefix="graph-gate-") as tmpdir:
                dump_dir = Path(tmpdir)
                ModelTestCase(test_def, json_dump_dir=dump_dir).dump_graph_only()

                gate_result = check_gate(
                    model_root=test_def.model_root,
                    dumped_dir=dump_dir,
                )
                if gate_result.skip_accuracy:
                    pytest.skip(f"Graph gate: {gate_result.reason}")
                else:
                    logging.info(f"Graph gate: {gate_result.reason} - running accuracy check")

        ModelTestCase(test_def).run()
