#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Produce a single global testdata archive consumed by every test job.

Output: build/onnxruntime-testdata.zip and build/onnxruntime-testdata.tar.bz2

The archive contains four arch-neutral top-level directories:
    testdata/            (from upstream ORT source — fetched via cmake/deps.txt)
    pytorch-converted/   (from cmake/external/onnx submodule)
    pytorch-operator/    (from cmake/external/onnx submodule)
    node/                (from cmake/external/onnx submodule)

extract_testdata.py knows how to re-map these handles into the on-disk locations
expected by run_tests.{ps1,sh} and the test binaries.
"""

import argparse
import hashlib
import http.client
import logging
import os
import re
import shutil
import tarfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse
from urllib.request import urlretrieve

QCOM_ROOT = Path(__file__).parent.parent.parent
REPO_ROOT = QCOM_ROOT.parent

# Persistent download cache root, matching package_manager.py's FileCache so we reuse the same
# on-disk file (see ort_core_cache_path below). Kept outside build/ so `git clean -ffdx` can't wipe
# it. Stdlib-only: this script runs on the bare interpreter (no venv), so it can't import FileCache
# (certifi/tqdm/yaml).
DEFAULT_CACHE_ROOT = Path(
    os.environ.get("ORT_BUILD_PACKAGE_CACHE_PATH", str((Path("~") / ".ort-package-cache").expanduser()))
)

# GitHub's codeload endpoint drops large transfers mid-stream (http.client.IncompleteRead) under
# throttling; retry with exponential backoff before giving up.
DOWNLOAD_ATTEMPTS = int(os.environ.get("ORT_BUILD_DOWNLOAD_ATTEMPTS", "3"))
DOWNLOAD_BACKOFF_BASE_SECONDS = float(os.environ.get("ORT_BUILD_DOWNLOAD_BACKOFF_SECONDS", "2"))

__all__ = [
    "OrtCoreDep",
    "download_and_verify",
    "parse_deps_txt",
    "stage_sources",
    "write_archives",
]


@dataclass(frozen=True)
class OrtCoreDep:
    url: str
    sha1: str


_ORT_CORE_LINE_RE = re.compile(r"^ort_core;([^;]+);([0-9a-fA-F]+)\s*$")


def parse_deps_txt(deps_file: Path) -> OrtCoreDep:
    """Parse cmake/deps.txt and return the ort_core URL + SHA1."""
    for line in deps_file.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = _ORT_CORE_LINE_RE.match(stripped)
        if m:
            return OrtCoreDep(url=m.group(1), sha1=m.group(2))
    raise ValueError(f"ort_core entry not found in {deps_file}")


def ort_core_cache_path(cache_root: Path, url: str) -> Path:
    """Location of the cached ort_core zip, matching package_manager.py's FileCache layout
    (<root>/ort_core/<url-basename>, e.g. <root>/ort_core/v1.27.0.zip) so a zip already fetched by
    fetch_cmake_deps.py on a persistent runner is reused instead of re-downloaded."""
    return cache_root / "ort_core" / PurePosixPath(urlparse(url).path).name


def _sha1_of(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_and_verify(url: str, sha1: str, cache_path: Path) -> Path:
    """Download `url` to `cache_path`. Skip fetch when cache exists with matching SHA1.
    Removes and re-downloads when a stale cache (mismatched SHA1) is found, so persistent
    CI workspaces recover automatically after a QAIRT uplevel changes cmake/deps.txt."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        actual = _sha1_of(cache_path)
        if actual == sha1.lower():
            logging.info("ort_core zip cache hit: %s", cache_path)
            return cache_path
        # Stale cache — delete and re-download rather than raising, so CI runners with
        # persistent build/ directories recover automatically after an ORT version bump.
        logging.warning(
            "SHA1 mismatch on cached %s (expected %s, got %s) — removing and re-downloading",
            cache_path,
            sha1,
            actual,
        )
        cache_path.unlink()

    last_error: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            logging.info("Downloading %s -> %s (attempt %d/%d)", url, cache_path, attempt, DOWNLOAD_ATTEMPTS)
            urlretrieve(url, cache_path)
            actual = _sha1_of(cache_path)
            if actual != sha1.lower():
                raise ValueError(f"SHA1 mismatch on freshly downloaded {cache_path}: expected {sha1}, got {actual}")
            return cache_path
        except (OSError, http.client.HTTPException, ValueError) as e:
            # OSError (urllib/socket), HTTPException (IncompleteRead — not an OSError), and our own
            # SHA1 ValueError all mean a bad/partial body; drop it so the next attempt starts clean.
            last_error = e
            cache_path.unlink(missing_ok=True)
            if attempt < DOWNLOAD_ATTEMPTS:
                delay = DOWNLOAD_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                logging.warning(
                    "Download attempt %d/%d for %s failed: %s. Retrying in %.0fs.",
                    attempt,
                    DOWNLOAD_ATTEMPTS,
                    url,
                    e,
                    delay,
                )
                time.sleep(delay)
    raise RuntimeError(f"Failed to download {url} after {DOWNLOAD_ATTEMPTS} attempt(s).") from last_error


# Maps each handle name to its source path. Handle names must match MAPPING in extract_testdata.py.
_HANDLES = {
    "testdata": "ort_core_src/onnxruntime/test/testdata",
    "pytorch-converted": "onnx_src/backend/test/data/pytorch-converted",
    "pytorch-operator": "onnx_src/backend/test/data/pytorch-operator",
    "node": "onnx_src/backend/test/data/node",
}


def stage_sources(stage_root: Path, ort_core_src: Path, onnx_src: Path) -> None:
    """Copy the 4 source trees into stage_root under their handle names.
    Existing stage_root contents are removed first."""
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True)
    for handle, rel in _HANDLES.items():
        if rel.startswith("ort_core_src/"):
            src = ort_core_src / rel[len("ort_core_src/") :]
        elif rel.startswith("onnx_src/"):
            src = onnx_src / rel[len("onnx_src/") :]
        else:
            raise AssertionError(f"unknown handle root in {rel}")
        if not src.is_dir():
            raise FileNotFoundError(f"Source for handle {handle!r} missing: {src}")
        shutil.copytree(src, stage_root / handle)


def write_archives(stage_root: Path, output_dir: Path) -> list[Path]:
    """Write both onnxruntime-testdata.zip and onnxruntime-testdata.tar.bz2 from stage_root."""
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "onnxruntime-testdata.zip"
    tar_path = output_dir / "onnxruntime-testdata.tar.bz2"
    zip_path.unlink(missing_ok=True)
    tar_path.unlink(missing_ok=True)

    files = sorted(p for p in stage_root.glob("**/*") if p.is_file())

    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, f.relative_to(stage_root).as_posix())

    with tarfile.open(tar_path, "w:bz2") as tf:
        for f in files:
            tf.add(f, str(f.relative_to(stage_root).as_posix()))

    return [zip_path, tar_path]


def main() -> int:
    log_format = "[%(asctime)s] [archive_testdata.py] [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=REPO_ROOT / "build" / "testdata-stage",
        help="Working directory for the ort_core download + extract.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "build",
        help="Directory to write onnxruntime-testdata.{zip,tar.bz2} into.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help=(
            "Persistent cache root for the ort_core zip. Defaults to ORT_BUILD_PACKAGE_CACHE_PATH "
            "(the same root as package_manager.py), so the ~278 MiB zip is shared with the build's "
            "fetch_cmake_deps download and survives `git clean` across CI runs."
        ),
    )
    args = parser.parse_args()

    dep = parse_deps_txt(REPO_ROOT / "cmake" / "deps.txt")

    cache_zip = ort_core_cache_path(args.cache_dir, dep.url)
    download_and_verify(dep.url, dep.sha1, cache_zip)

    extract_root = args.build_dir / "ort_core-src"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True)
    with zipfile.ZipFile(cache_zip) as zf:
        zf.extractall(extract_root)
    # The zip extracts to onnxruntime-<version>/, hop down one level.
    children = [p for p in extract_root.iterdir() if p.is_dir()]
    if len(children) != 1:
        raise RuntimeError(f"Unexpected layout in ort_core extraction: {children}")
    ort_core_src = children[0]

    onnx_src = REPO_ROOT / "cmake" / "external" / "onnx" / "onnx"

    stage_root = args.build_dir / "stage"
    stage_sources(stage_root=stage_root, ort_core_src=ort_core_src, onnx_src=onnx_src)
    written = write_archives(stage_root=stage_root, output_dir=args.output_dir)
    for p in written:
        logging.info("Wrote %s (%.1f MiB)", p, p.stat().st_size / (1 << 20))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
