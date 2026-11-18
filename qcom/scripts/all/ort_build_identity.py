# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Describe ORT inputs that require regenerating an incremental build."""

import argparse
import hashlib
import json
import os
from pathlib import Path

TESTDATA_IDENTITY_FILE = "ort-testdata-identity.json"
RUNTIME_VERSION_FILE = "ort-runtime-version.txt"


def build_identity(sdk: str, assets: str) -> str:
    def describe(root: str) -> dict[str, str | None] | None:
        if not root:
            return None
        path = Path(root).resolve()
        try:
            version = (path / "VERSION_NUMBER").read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            # Let CMake report the missing version file during configuration.
            version = None
        return {"path": path.as_posix(), "version": version}

    return json.dumps({"sdk": describe(sdk), "assets": describe(assets) if sdk else None}, sort_keys=True)


def _read_version(root: str, description: str) -> str:
    version_file = Path(root) / "VERSION_NUMBER"
    version = version_file.read_text(encoding="utf-8").strip()
    if not version:
        raise ValueError(f"{description} version file is empty: {version_file}")
    return version


def validate_test_inputs(sdk: str, assets: str, *, runtime_version: str | None = None) -> str | None:
    """Validate common Python test inputs; task-specific runners stay with callers."""
    version = _read_version(sdk, "Prebuilt ORT") if sdk else runtime_version
    if not assets:
        if version is not None:
            raise ValueError("Prebuilt ORT tests require ORT test assets.")
        return None
    testdata = Path(assets) / "onnxruntime/test/testdata"
    if not testdata.is_dir():
        raise FileNotFoundError(f"ORT provider testdata does not exist: {testdata}")
    if version is not None:
        assets_version = _read_version(assets, "ORT test-assets")
        if assets_version != version:
            raise ValueError(f"ORT test-assets version {assets_version} does not match prebuilt ORT {version}.")
    return version


def testdata_digest(root: Path) -> str:
    """Hash sorted relative file names and contents, independent of paths/timestamps."""
    if not root.is_dir():
        raise FileNotFoundError(f"ORT provider testdata does not exist: {root}")
    digest = hashlib.sha256()
    files = sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix())
    for path in files:
        name = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        content = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                content.update(chunk)
        digest.update(content.digest())
    return digest.hexdigest()


def testdata_identity(sdk: str, assets: str, *, runtime_version: str | None = None) -> str:
    """Describe portable ORT model compatibility, separately from local build identity."""
    version = validate_test_inputs(sdk, assets, runtime_version=runtime_version)
    return json.dumps(
        {"version": version, "testdata_sha256": testdata_digest(Path(assets) / "onnxruntime/test/testdata")},
        sort_keys=True,
    )


def validate_testdata_archive(staging: Path, runtime_version: str | None = None) -> None:
    """Reject an archive prepared for different inputs before replacing test models."""
    sdk = os.environ.get("ORT_PREBUILT_ROOT", "")
    assets = os.environ.get("ORT_TEST_ASSETS_ROOT", "")
    identity_file = staging / TESTDATA_IDENTITY_FILE
    if not sdk and runtime_version is None and not identity_file.exists():
        return  # Existing source-build archives have no prebuilt identity.
    error = "Testdata archive does not match the selected ORT SDK and test assets; rerun archive_testdata."
    if not identity_file.is_file():
        raise ValueError(error)
    actual = json.loads(identity_file.read_text(encoding="utf-8"))
    if not isinstance(actual, dict) or set(actual) != {"version", "testdata_sha256"}:
        raise ValueError(error)
    if sdk or assets:
        expected = json.loads(testdata_identity(sdk, assets, runtime_version=runtime_version))
        if actual != expected:
            raise ValueError(error)
    elif runtime_version is None:
        raise ValueError(
            "Prebuilt testdata requires a selected SDK/assets or a runtime version from the per-arch test archive."
        )
    if runtime_version is not None and actual.get("version") != runtime_version:
        raise ValueError(error)
    if actual.get("testdata_sha256") != testdata_digest(staging / "testdata"):
        raise ValueError(error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", default="")
    parser.add_argument("--assets", default="")
    args = parser.parse_args()
    print(build_identity(args.sdk, args.assets))
