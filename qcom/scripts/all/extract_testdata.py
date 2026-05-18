#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Extract a global testdata archive into per-platform on-disk locations.

The archive has 4 top-level handles; each is re-mapped to where run_tests.{ps1,sh}
and the test binaries expect to find it.
"""

import argparse
import logging
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

QCOM_ROOT = Path(__file__).parent.parent.parent
REPO_ROOT = QCOM_ROOT.parent

# Each value is a format-string with placeholders {plat} and {config}.
# {plat}   = "<os>-<arch>",  e.g. "linux-x86_64" or "windows-arm64"
# {config} = build config dir, normally "Release" but "Release/Release" for
#            Windows cross-compiled with a multi-config generator (VS).
# Handle names must match _HANDLES in archive_testdata.py — keep in sync.
MAPPING: dict[str, str] = {
    "testdata": "build/{plat}/{config}/testdata",
    "pytorch-converted": "build/{plat}/{config}/_deps/onnx-src/onnx/backend/test/data/pytorch-converted",
    "pytorch-operator": "build/{plat}/{config}/_deps/onnx-src/onnx/backend/test/data/pytorch-operator",
    # The `node` handle is REPO_ROOT-relative — {plat}/{config} placeholders are unused.
    "node": "cmake/external/onnx/onnx/backend/test/data/node",
}


def _detect_config_dir(repo_root: Path, plat: str) -> str:
    """Return 'Release/Release' for multi-config Windows builds, 'Release' otherwise.

    Visual Studio generators create build/<plat>/Release/Release/ for cross-compiled
    Windows targets (e.g. ARM64 built from an X64 host).  Detect by probing whether
    that second-level directory already exists — which it will after the per-arch
    test archive has been extracted (the caller must ensure that ordering).
    """
    if (repo_root / "build" / plat / "Release" / "Release").is_dir():
        return "Release/Release"
    return "Release"


def _extract_to_staging(archive: Path, staging: Path) -> None:
    """Extract archive into staging dir. Supports .zip and .tar.bz2."""
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(staging)
    elif archive.name.endswith(".tar.bz2") or archive.suffix in (".tbz2", ".bz2"):
        with tarfile.open(archive, "r:bz2") as tf:
            try:
                tf.extractall(staging, filter="data")
            except TypeError:
                # filter= parameter added in Python 3.12; fall back for older runtimes.
                tf.extractall(staging)
    else:
        raise ValueError(f"Unsupported archive format: {archive}")


def extract(archive: Path, target_platform: str, repo_root: Path = REPO_ROOT) -> None:
    """Extract testdata archive and re-map its 4 handles to expected on-disk paths.

    The per-arch test archive must already be extracted into repo_root before calling
    this function so that _detect_config_dir() can probe for the multi-config layout.
    """
    config_dir = _detect_config_dir(repo_root, target_platform)
    with tempfile.TemporaryDirectory(prefix="extract-testdata-") as tmp:
        staging = Path(tmp)
        _extract_to_staging(archive, staging)

        present_handles = {p.name for p in staging.iterdir() if p.is_dir()}
        unknown = present_handles - set(MAPPING.keys())
        if unknown:
            raise ValueError(f"Archive contains unexpected top-level entries: {sorted(unknown)}")

        for handle, rel_template in MAPPING.items():
            src = staging / handle
            if not src.exists():
                logging.warning("Handle %r missing from archive — skipping.", handle)
                continue
            dest = repo_root / rel_template.format(plat=target_platform, config=config_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(src), str(dest))


def main() -> int:
    log_format = "[%(asctime)s] [extract_testdata.py] [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-platform", required=True, help='e.g., "linux-x86_64", "windows-arm64"')
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root; defaults to REPO_ROOT inferred from script location.",
    )
    args = parser.parse_args()
    extract(archive=args.archive, target_platform=args.target_platform, repo_root=args.repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
