#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path

from artifactory import QaArtifactory, initialize_logging, valid_artifact_name

REPO_ROOT = Path(__file__).parent.parent.parent.parent.absolute()


class PublishQaArtifact:
    def __init__(self, tag: str, build_dir: Path, src_pattern: str) -> None:
        self.__build_dir = build_dir
        self.__src_pattern = src_pattern
        self.__client = QaArtifactory(tag)

    @property
    def destination(self) -> str:
        return f"{self.__client.artifact_root}/"

    def run(self) -> None:
        self.__client.upload(self.__build_dir, self.__src_pattern, self.destination, flat=True)


if __name__ == "__main__":
    initialize_logging("publish_qa.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=valid_artifact_name, required=True, help="QA tag")
    parser.add_argument("--src-root", type=Path, default=REPO_ROOT / "build", help="Root directory of artifacts")
    parser.add_argument("--src-pattern", type=str, required=True, help="Artifact file pattern, relative to SRC_ROOT.")

    args = parser.parse_args()

    PublishQaArtifact(
        args.tag,
        args.src_root,
        args.src_pattern,
    ).run()
