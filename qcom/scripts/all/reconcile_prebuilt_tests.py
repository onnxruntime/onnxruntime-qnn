# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Stage selected assets for tests that consume an existing instrumented build."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from ort_build_identity import build_identity, validate_test_inputs


def reconcile(build_dir: Path, config: str, sdk: str, assets: str) -> None:
    if not sdk:
        return
    validate_test_inputs(sdk, assets)
    selected = json.loads(build_identity(sdk, assets))
    marker = build_dir / f"ort-build-identity-{config}.txt"
    if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8"))["sdk"] != selected["sdk"]:
        raise ValueError(
            "Existing test build does not match the selected prebuilt ORT SDK; "
            "rebuild with the required coverage/ASan instrumentation before testing."
        )

    config_dir = build_dir / config
    cache = {}
    for line in (config_dir / "CMakeCache.txt").read_text(encoding="utf-8").splitlines():
        if "=" in line and ":" in line.split("=", 1)[0]:
            key, value = line.split("=", 1)
            cache[key.split(":", 1)[0]] = value
    if Path(cache["CMAKE_CACHEFILE_DIR"]).resolve() != config_dir.resolve():
        raise ValueError("Existing prebuilt test build has moved; rebuild at its current path before testing.")
    cmake = cache["CMAKE_COMMAND"]
    # Preserve instrumentation and all other cached build options. Only update inputs
    # after the SDK identity check, so a report task cannot silently rebuild against B.
    subprocess.run(
        [
            cmake,
            "-S",
            cache["CMAKE_HOME_DIRECTORY"],
            "-B",
            str(config_dir),
            f"-Donnxruntime_ORT_HOME:PATH={sdk}",
            f"-Donnxruntime_ORT_TEST_ASSETS_HOME:PATH={assets}",
        ],
        check=True,
    )
    subprocess.run(
        [cmake, "--build", str(config_dir), "--config", config, "--target", "ort_test_assets_target"], check=True
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sdk", required=True)
    parser.add_argument("--assets", default=os.environ.get("ORT_TEST_ASSETS_ROOT", ""))
    args = parser.parse_args()
    try:
        reconcile(args.build_dir, args.config, args.sdk, args.assets)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"Prebuilt test input reconciliation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
