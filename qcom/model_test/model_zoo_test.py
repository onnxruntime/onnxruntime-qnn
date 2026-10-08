# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import cast, get_args

import onnxruntime_qnn
import pytest
from graph_snapshot import normalize_qnn_graph_dump_dir
from graph_snapshot_gate import check_snapshot_gate
from model_test import BackendT, ModelTestCase, ModelTestDef, ModelTestSuite

MODEL_ZOO_ROOTS = [Path(p) for p in os.getenv("ORT_MODEL_ZOO_TEST_ROOTS", "").split(os.pathsep) if len(p) > 0]
MODEL_ZOO_BACKEND = cast(BackendT, os.getenv("ORT_MODEL_ZOO_BACKEND", "htp"))
assert MODEL_ZOO_BACKEND in get_args(BackendT)
MODEL_ZOO_ENABLE_CONTEXT = os.getenv("ORT_MODEL_ZOO_ENABLE_CONTEXT", "1") == "1"
MODEL_ZOO_ENABLE_CPU_FALLBACK = os.getenv("ORT_MODEL_ZOO_ENABLE_CPU_FALLBACK", "0") == "1"
_model_zoo_snapshot_dir = os.getenv("ORT_MODEL_ZOO_SNAPSHOT_DIR", "")
MODEL_ZOO_SNAPSHOT_DIR = Path(_model_zoo_snapshot_dir) if _model_zoo_snapshot_dir else None
_model_zoo_snapshot_golden_dir = os.getenv("ORT_MODEL_ZOO_SNAPSHOT_GOLDEN_DIR", "")
MODEL_ZOO_SNAPSHOT_GOLDEN_DIR = Path(_model_zoo_snapshot_golden_dir) if _model_zoo_snapshot_golden_dir else None
MODEL_ZOO_PLATFORM = os.getenv("ORT_MODEL_ZOO_PLATFORM", "")
MODEL_ZOO_HTP_ARCH = os.getenv("ORT_MODEL_ZOO_HTP_ARCH", "")


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

        if MODEL_ZOO_SNAPSHOT_DIR is None or MODEL_ZOO_BACKEND != "htp":
            if (
                MODEL_ZOO_SNAPSHOT_GOLDEN_DIR is not None
                and MODEL_ZOO_BACKEND == "htp"
                and MODEL_ZOO_PLATFORM
                and MODEL_ZOO_HTP_ARCH
            ):
                relative_model_dir = Path(test_def.model_root.parent.name) / test_def.model_root.name
                with tempfile.TemporaryDirectory(prefix="modelzoo-snapshot-gate-") as temporary_dir:
                    snapshot_dir = Path(temporary_dir)
                    test_case = ModelTestCase(test_def, json_dump_dir=snapshot_dir)
                    test_case.dump_graph_only()
                    normalize_qnn_graph_dump_dir(snapshot_dir)
                    package_info = onnxruntime_qnn.build_and_package_info
                    gate_result = check_snapshot_gate(
                        snapshot_dir,
                        MODEL_ZOO_SNAPSHOT_GOLDEN_DIR,
                        relative_model_dir,
                        MODEL_ZOO_PLATFORM,
                        MODEL_ZOO_HTP_ARCH,
                        package_info.__version__,
                        package_info.qnn_version,
                    )
                if gate_result.skip_real_execution:
                    logging.info("ModelZoo snapshot gate passed for %s: %s", test_def.model_root.name, gate_result.reason)
                    return
                logging.info("ModelZoo snapshot gate is unverified for %s: %s", test_def.model_root.name, gate_result.reason)
            ModelTestCase(test_def).run()
            return

        # The snapshot producer is opt-in and HTP-only. Each successful test
        # still performs its normal accuracy check before its dump is accepted.
        snapshot_dir = MODEL_ZOO_SNAPSHOT_DIR / test_def.model_root.parent.name / test_def.model_root.name
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        test_case = ModelTestCase(test_def, json_dump_dir=snapshot_dir)
        test_case.run()
        graph_files = normalize_qnn_graph_dump_dir(snapshot_dir)
        logging.info("Wrote %d normalized QNN graph snapshot(s) for %s", len(graph_files), test_def.model_root.name)
