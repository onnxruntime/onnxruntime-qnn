#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path

import yaml

from artifactory import ToolArtifactory, initialize_logging, valid_artifact_name

REPO_ROOT = Path(__file__).parent.parent.parent.parent.absolute()
PACKAGES_YML = REPO_ROOT / "qcom" / "packages.yml"


def default_qairt_version() -> str:
    """The qnn-udo-test-package artifact is locked 1:1 to the pinned QAIRT SDK version -- a
    prebuilt op package cannot float across QAIRT SDK versions (see build_udo_test_package.py)."""
    with PACKAGES_YML.open() as f:
        config = yaml.safe_load(f)
    return config["qairt"]["version"]


class PublishToolArtifact:
    def __init__(self, name: str, version: str, build_dir: Path, src_pattern: str) -> None:
        self.__build_dir = build_dir
        self.__src_pattern = src_pattern
        self.__client = ToolArtifactory(name, version)

    @property
    def destination(self) -> str:
        return f"{self.__client.artifact_root}/"

    def run(self) -> None:
        self.__client.upload(self.__build_dir, self.__src_pattern, self.destination)


if __name__ == "__main__":
    initialize_logging("publish_tool.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=valid_artifact_name, required=True, help="Artifact name")
    parser.add_argument(
        "--version",
        default=None,
        help="Artifact version. Defaults to the QAIRT SDK version pinned in qcom/packages.yml.",
    )
    parser.add_argument("--src-root", type=Path, default=REPO_ROOT / "build", help="Root directory of artifacts")
    parser.add_argument("--src-pattern", type=str, required=True, help="Artifact file pattern, relative to SRC_ROOT.")

    args = parser.parse_args()

    PublishToolArtifact(
        args.name,
        args.version if args.version is not None else default_qairt_version(),
        args.src_root,
        args.src_pattern,
    ).run()
