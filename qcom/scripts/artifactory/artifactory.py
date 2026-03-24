# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path


class Artifactory:
    def __init__(self) -> None:
        self.__repo = os.environ["BUILD_ARTIFACTORY_REPO"]
        if len(self.__repo) == 0:
            raise ValueError("BUILD_ARTIFACTORY_REPO not set")

    @property
    def repo_path(self) -> str:
        return self.__repo

    def download(self, src_pattern: str, destination: Path) -> None:
        # jf rt download preserves all the extra bookkeeping stuff we put into the path.
        # Download everything into a tempdir and copy into the final destination.
        with tempfile.TemporaryDirectory(prefix="ArtifactoryDownload-") as tmpdir:
            self.__run(
                [
                    "rt",
                    "download",
                    src_pattern,
                    f"{tmpdir}/",
                ],
                cwd=None,
            )
            relpath = "/".join(src_pattern.split("/")[1:])
            logging.info(f"Copying everything under {Path(tmpdir) / relpath} to {destination}.")
            shutil.copytree(Path(tmpdir) / relpath, destination, dirs_exist_ok=True)

    def upload(self, cwd: Path, src_pattern: str, destination: str) -> None:
        self.__run(
            [
                "rt",
                "upload",
                src_pattern,
                destination,
            ],
            cwd,
        )

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
        self.__ref = os.environ.get("GITHUB_REF", f"{actor}/{run_id}")

    @property
    def artifact_root(self) -> str:
        return f"{super().repo_path}/ci/{self.__ref}/{self.__commit_hash[:10]}"


def initialize_logging(name: str) -> None:
    log_format = f"[%(asctime)s] [{name}] [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_format, force=True)


def valid_artifact_name(proposed_name: str) -> str:
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9-\._]*$", proposed_name):
        raise ValueError(f"{proposed_name} is not a valid artifact name.")
    return proposed_name
