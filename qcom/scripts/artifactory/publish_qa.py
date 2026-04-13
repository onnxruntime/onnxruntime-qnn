#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path

from artifactory import QaArtifactory, initialize_logging

REPO_ROOT = Path(__file__).parent.parent.parent.parent.absolute()


class PublishQaArtifact:
    def __init__(
        self,
        date: str,
        branch: str,
        commit: str,
        build_dir: Path,
        src_pattern: str,
        dest_subpath: str = "",
    ) -> None:
        self.__build_dir = build_dir
        self.__src_pattern = src_pattern
        self.__dest_subpath = dest_subpath
        self.__client = QaArtifactory(date, branch, commit)

    @property
    def destination(self) -> str:
        if self.__dest_subpath:
            return f"{self.__client.artifact_root}/{self.__dest_subpath}/"
        return f"{self.__client.artifact_root}/"

    def run(self) -> None:
        self.__client.upload(self.__build_dir, self.__src_pattern, self.destination, flat=True)


if __name__ == "__main__":
    initialize_logging("publish_qa.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True, help="Date in DD_MM_YY_HH_MM_SS format")
    parser.add_argument("--branch", type=str, required=True, help="Branch name (e.g. main)")
    parser.add_argument("--commit", type=str, required=True, help="First 10 characters of commit SHA")
    parser.add_argument("--src-root", type=Path, default=REPO_ROOT / "build", help="Root directory of artifacts")
    parser.add_argument("--src-pattern", type=str, required=True, help="Artifact file pattern, relative to SRC_ROOT.")
    parser.add_argument(
        "--dest-subpath",
        type=str,
        default="",
        help="Subdirectory within the QA root to upload into (e.g. wheels/windows-arm64).",
    )

    args = parser.parse_args()

    PublishQaArtifact(
        args.date,
        args.branch,
        args.commit,
        args.src_root,
        args.src_pattern,
        args.dest_subpath,
    ).run()
