#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Package and publish HTP ModelZoo graph snapshots.

The artifact contract intentionally matches the unit-test golden publisher:
an immutable archive is uploaded first, followed by mutable ``latest``
manifest and archive pointers. HTP architecture is part of the path so goldens
from different devices cannot be compared accidentally.
"""

import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path


def _wheel_versions(wheel: Path) -> tuple[str, str]:
    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode("utf-8")
        ort_version = re.search(r"^Version: (.+)$", metadata, re.MULTILINE)
        package_info_name = next(
            name for name in archive.namelist() if name.endswith("onnxruntime_qnn/build_and_package_info.py")
        )
        package_info = archive.read(package_info_name).decode("utf-8")
        qairt_version = re.search(r"^qnn_version\s*=\s*['\"]([^'\"]+)['\"]", package_info, re.MULTILINE)
    if ort_version is None or qairt_version is None:
        raise ValueError("Wheel does not contain ORT and QAIRT version metadata.")
    return ort_version.group(1), qairt_version.group(1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_repo_subpath(value: str) -> str:
    if not re.fullmatch(r"ci/qnn-ep-test-store/[A-Za-z0-9._/-]+", value) or ".." in value or value.endswith("/"):
        raise argparse.ArgumentTypeError("repo subpath must stay below ci/qnn-ep-test-store")
    return value


def _upload(source: Path, destination: str) -> None:
    subprocess.run(["jf", "rt", "upload", "--flat", str(source), destination], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--wheel", required=True, help="Wheel path or glob; newest match is used.")
    parser.add_argument("--modelzoo-platform", required=True, help="ModelZoo test platform, for example windows-arm64.")
    parser.add_argument("--htp-arch", required=True, choices=["v73", "v81"])
    parser.add_argument("--repo-subpath", type=_validate_repo_subpath, default="ci/qnn-ep-test-store/modelzoo-snapshot-goldens")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--git-sha", default=os.getenv("GITHUB_SHA", "unknown"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--publish", action="store_true")
    args = parser.parse_args()

    if not args.snapshot_dir.is_dir():
        raise ValueError(f"Snapshot directory does not exist: {args.snapshot_dir}")
    snapshots = sorted(path for path in args.snapshot_dir.rglob("*.json") if path.is_file())
    if not snapshots:
        raise ValueError("Snapshot directory contains no graph JSON files.")
    wheel_matches = sorted((Path(path) for path in glob.glob(args.wheel)), key=lambda path: path.stat().st_mtime)
    if not wheel_matches:
        raise ValueError(f"Wheel does not exist: {args.wheel}")
    wheel = wheel_matches[-1]

    ort_version, qairt_version = _wheel_versions(wheel)
    artifact_key = f"{args.modelzoo_platform}-htp-{args.htp_arch}"
    generated_utc = datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    archive_id = f"{args.git_sha[:12]}-{args.htp_arch}-{generated_utc.replace(':', '').replace('-', '')}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=args.output_dir, prefix="modelzoo-snapshot-") as temp:
        staging = Path(temp)
        files = []
        for source in snapshots:
            relative = source.relative_to(args.snapshot_dir).as_posix()
            files.append({"path": relative, "sha256": _sha256(source)})
        manifest = {
            "archive_id": archive_id,
            "files": files,
            "generated_utc": generated_utc,
            "generator": "publish_modelzoo_snapshots.py",
            "git_sha": args.git_sha,
            "htp_arch": args.htp_arch,
            "modelzoo_platform": args.modelzoo_platform,
            "ort_version": ort_version,
            "qairt_version": qairt_version,
            "snapshot_schema_version": 1,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        archive_path = args.output_dir / "snapshots.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(manifest_path, "manifest.json")
            for source in snapshots:
                archive.write(source, source.relative_to(args.snapshot_dir).as_posix())
        final_manifest = args.output_dir / "manifest.json"
        final_manifest.write_bytes(manifest_path.read_bytes())

    if args.publish:
        repository = os.environ["BUILD_ARTIFACTORY_REPO"]
        base = f"{repository}/{args.repo_subpath}/{artifact_key}"
        archive_base = f"{base}/archive/{archive_id}"
        _upload(final_manifest, f"{archive_base}/manifest.json")
        _upload(archive_path, f"{archive_base}/snapshots.zip")
        _upload(final_manifest, f"{base}/latest/manifest.json")
        _upload(archive_path, f"{base}/latest/snapshots.zip")

    print(json.dumps({"archive": str(archive_path), "manifest": str(final_manifest), "mode": "publish" if args.publish else "dry-run"}))


if __name__ == "__main__":
    main()
