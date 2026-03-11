#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path

from artifactory import CiArtifactory, initialize_logging, valid_artifact_name

REPO_ROOT = Path(__file__).parent.parent.parent.parent.absolute()


class DownloadCiArtifact:
    def __init__(self, name: str, destination: Path) -> None:
        self.__name = name
        self.__destination = destination
        self.__client = CiArtifactory()

    @property
    def source(self) -> str:
        return f"{self.__client.artifact_root}/{self.__name}/"

    def run(self) -> None:
        self.__client.download(self.source, self.__destination)


if __name__ == "__main__":
    initialize_logging("download_ci.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=valid_artifact_name, required=True, help="Artifact name")
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=REPO_ROOT / "build",
        help="Root directory of artifacts"
    )

    args = parser.parse_args()

    DownloadCiArtifact(
        args.name,
        args.dest_root,
    ).run()
