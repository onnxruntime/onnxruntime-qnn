# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Describe ORT inputs that require regenerating an incremental build."""

import argparse
import json
from pathlib import Path


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", default="")
    parser.add_argument("--assets", default="")
    args = parser.parse_args()
    print(build_identity(args.sdk, args.assets))
