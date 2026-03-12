#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path

from artifactory import CiArtifactory, initialize_logging, valid_artifact_name

REPO_ROOT = Path(__file__).parent.parent.parent.parent.absolute()


class PublishCiArtifact:
    def __init__(self, name: str, build_dir: Path, src_pattern: str) -> None:
        self.__name = name
        self.__build_dir = build_dir
        self.__src_pattern = src_pattern
        self.__client = CiArtifactory()

    @property
    def destination(self) -> str:
        return f"{self.__client.artifact_root}/{self.__name}/"

    def run(self) -> None:
        self.__client.upload(self.__build_dir, self.__src_pattern, self.destination)


if __name__ == "__main__":
    initialize_logging("publish_ci.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=valid_artifact_name, required=True, help="Artifact name")
    parser.add_argument(
        "--src-root",
        type=Path,
        default=REPO_ROOT / "build",
        help="Root directory of artifacts"
    )
    parser.add_argument("--src-pattern", type=str, required=True, help="Artifact file pattern, relative to SRC_ROOT.")

    args = parser.parse_args()

    PublishCiArtifact(
        args.name,
        args.src_root,
        args.src_pattern,
    ).run()
