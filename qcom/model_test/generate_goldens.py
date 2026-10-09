#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Generate golden graph JSONs for the model_zoo graph-diff gate.

Iterates all models in a test suite, creates an InferenceSession with
dump_json_qnn_graph=1 to produce the compiled QNN graph JSON, and saves
the results into each model's goldens/ subdirectory alongside a version manifest.

Usage:
    python generate_goldens.py --suite <model_zoo_root>/winml-cert

Produces (in-place):
    <suite>/<model_name>/goldens/*.json             -- per-subgraph QNN graph JSON
    <suite>/<model_name>/goldens/golden_manifest.json -- version metadata
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import get_args

from graph_gate import GOLDEN_MANIFEST_FILENAME, get_current_versions, normalize_graph_json
from model_test import BackendT, ModelTestCase, ModelTestDef, ModelTestSuite, initialize_logging


def generate_golden_for_model(test_def: ModelTestDef) -> bool:
    """Generate golden graph JSON for a single model.

    Outputs to <model_root>/goldens/. Returns True on success, False on failure.
    """
    model_name = test_def.model_root.name
    model_output_dir = test_def.model_root / "goldens"

    # Clean previous golden if exists
    if model_output_dir.exists():
        shutil.rmtree(model_output_dir)
    model_output_dir.mkdir(parents=True)

    try:
        # Create session with graph dump enabled — no inference needed
        ModelTestCase(test_def, json_dump_dir=model_output_dir).dump_graph_only()
    except Exception as e:
        logging.error(f"Failed to generate golden for {model_name}: {e}")
        shutil.rmtree(model_output_dir, ignore_errors=True)
        return False

    # Normalize all dumped JSONs (strip tensor IDs) and rewrite
    json_files = list(model_output_dir.glob("*.json"))
    if not json_files:
        logging.warning(f"No graph JSONs dumped for {model_name} — skipping")
        shutil.rmtree(model_output_dir, ignore_errors=True)
        return False

    for json_file in json_files:
        graph = json.loads(json_file.read_text())
        normalized = normalize_graph_json(graph)
        json_file.write_text(json.dumps(normalized, indent=2))

    # Write manifest
    manifest = {
        **get_current_versions(),
        "htp_arch": None,  # null = shared golden, valid for any arch
        "graphs": sorted(f.name for f in json_files),
    }
    (model_output_dir / GOLDEN_MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))

    logging.info(f"Generated golden for {model_name} ({len(json_files)} subgraph(s))")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", type=Path, required=True, help="Path to model zoo suite directory (e.g. winml-cert)")
    parser.add_argument("--backend", default="htp", choices=get_args(BackendT), help="QNN backend (default: htp)")
    parser.add_argument(
        "--enable-context", action="store_true", default=True, help="Enable context caching (default: True)"
    )
    parser.add_argument("--enable-cpu-fallback", action="store_true", help="Allow CPU fallback")
    args = parser.parse_args()

    initialize_logging("generate_goldens")

    suite = ModelTestSuite(
        args.suite,
        backend_type=args.backend,
        rtol=None,
        atol=None,
        cosine_similarity=None,
        enable_context=args.enable_context,
        enable_cpu_fallback=args.enable_cpu_fallback,
    )

    success_count = 0
    fail_count = 0

    for test_def in suite.tests:
        if generate_golden_for_model(test_def):
            success_count += 1
        else:
            fail_count += 1

    logging.info(f"Done: {success_count} succeeded, {fail_count} failed")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
