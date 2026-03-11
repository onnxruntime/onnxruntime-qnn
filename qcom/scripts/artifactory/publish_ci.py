#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import argparse
import logging
import os
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.parent.absolute()


class Artifactory:
    def __init__(self) -> None:
        self.__repo = os.environ["BUILD_ARTIFACTORY_REPO"]

    @property
    def repo_path(self) -> str:
        return self.__repo

    def upload(self, cwd: Path, src_pattern: str, destination: str) -> None:
        self.__run([
            "rt",
            "upload",
            "--insecure-tls=true",  # TODO-jrk
            src_pattern,
            destination,
        ], cwd)

    def __run(self, args: Iterable[str], cwd: Path | None) -> None:
        cmd = ["jf", *args]
        logging.debug(f"Running {cmd} from {cwd}")
        subprocess.run(args=cmd, cwd=cwd, check=True)


class CiArtifactory(Artifactory):
    def __init__(self) -> None:
        super().__init__()
        self.__commit_hash = os.environ["GITHUB_SHA"]
        actor = os.environ["GITHUB_ACTOR"] if os.environ["GITHUB_ACTOR"] != "" else "main"
        run_id = os.environ["GITHUB_RUN_ID"]
        self.__run_attempt = os.environ["GITHUB_RUN_ATTEMPT"]
        self.__ref = os.environ.get("GITHUB_REF", f"{actor}/{run_id}")

    @property
    def artifact_root(self) -> str:
        return f"{super().repo_path}/ci/{self.__ref}/{self.__commit_hash[:10]}-{self.__run_attempt}"


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


def initialize_logging() -> None:
    log_format = "[%(asctime)s] [publish_ci.py] [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_format, force=True)


def valid_artifact_name(proposed_name: str) -> str:
    if not re.match(r"^[a-z][a-z0-9-\._]*$", proposed_name):
        raise ValueError(f"{proposed_name} is not a valid artifact name.")
    return proposed_name


if __name__ == "__main__":
    initialize_logging()

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
