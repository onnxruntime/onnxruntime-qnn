# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Describe ORT inputs that require regenerating an incremental build."""

import argparse
import json
import os
from pathlib import Path

TESTDATA_IDENTITY_FILE = "ort-testdata-identity.json"


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


def testdata_identity(sdk: str, assets: str) -> str:
    """Validate the selected prebuilt inputs before archiving or extracting models."""
    identity = build_identity(sdk, assets)
    if sdk:
        selected = json.loads(identity)
        for name in ("sdk", "assets"):
            if not selected[name] or not selected[name]["version"]:
                raise ValueError(f"Prebuilt testdata requires {name} with a VERSION_NUMBER file.")
        if selected["sdk"]["version"] != selected["assets"]["version"]:
            raise ValueError("ORT test-assets version does not match prebuilt ORT.")
        if not (Path(assets) / "onnxruntime/test/testdata").is_dir():
            raise FileNotFoundError(f"ORT provider testdata not found under {assets}")
    return identity


def validate_testdata_archive(staging: Path) -> None:
    """Reject an archive prepared for different inputs before replacing test models."""
    sdk = os.environ.get("ORT_PREBUILT_ROOT", "")
    assets = os.environ.get("ORT_TEST_ASSETS_ROOT", "")
    identity_file = staging / TESTDATA_IDENTITY_FILE
    if not sdk and not identity_file.exists():
        return  # Existing source-build archives have no prebuilt identity.
    expected = json.loads(testdata_identity(sdk, assets))
    if not identity_file.is_file() or json.loads(identity_file.read_text(encoding="utf-8")) != expected:
        raise ValueError(
            "Testdata archive does not match the selected ORT SDK and test assets; rerun archive_testdata."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", default="")
    parser.add_argument("--assets", default="")
    args = parser.parse_args()
    print(build_identity(args.sdk, args.assets))
