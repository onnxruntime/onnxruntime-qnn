#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import argparse
import subprocess
import sys
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


class DownloadToolArtifact:
    def __init__(self, name: str, version: str, destination: Path) -> None:
        self.__name = name
        self.__version = version
        self.__destination = destination
        self.__client = ToolArtifactory(name, version)

    @property
    def source(self) -> str:
        return f"{self.__client.artifact_root}/"

    def run(self) -> None:
        try:
            self.__client.download(self.source, self.__destination)
        except subprocess.CalledProcessError:
            print(
                f"{self.__name} {self.__version} not found on Artifactory. Run the "
                "qualcomm-internal-publish-udo-package workflow against this QAIRT SDK version, or "
                "build it locally with qcom/scripts/linux/build_udo_test_package.py.",
                file=sys.stderr,
            )
            raise


if __name__ == "__main__":
    initialize_logging("download_tool.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=valid_artifact_name, required=True, help="Artifact name")
    parser.add_argument(
        "--version",
        default=None,
        help="Artifact version. Defaults to the QAIRT SDK version pinned in qcom/packages.yml.",
    )
    parser.add_argument("--dest-root", type=Path, default=REPO_ROOT / "build", help="Root directory of artifacts")

    args = parser.parse_args()

    DownloadToolArtifact(
        args.name,
        args.version if args.version is not None else default_qairt_version(),
        args.dest_root,
    ).run()
