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
import logging
import re
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

QCOM_ROOT = Path(__file__).parent.parent.parent
REPO_ROOT = QCOM_ROOT.parent

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
    logging.info("Downloading %s -> %s", url, cache_path)
    urlretrieve(url, cache_path)
    actual = _sha1_of(cache_path)
    if actual != sha1.lower():
        raise ValueError(f"SHA1 mismatch on freshly downloaded {cache_path}: expected {sha1}, got {actual}")
    return cache_path


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
    args = parser.parse_args()

    dep = parse_deps_txt(REPO_ROOT / "cmake" / "deps.txt")

    cache_zip = args.build_dir / "ort_core.zip"
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
