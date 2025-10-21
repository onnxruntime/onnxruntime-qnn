#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import argparse
import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.parent

DEFAULT_LOCAL_WHEELS_DIR = REPO_ROOT / "local_wheels"


def find_requirements_files() -> list[Path]:
    ci_build_req_dir = REPO_ROOT / "tools" / "ci_build" / "requirements"
    return list(ci_build_req_dir.rglob("requirements.txt"))


def main(local_wheels_dir: Path) -> None:
    local_wheels_dir.mkdir(parents=True, exist_ok=True)
    requirements_files = find_requirements_files()

    if not requirements_files:
        logging.warning("No requirements.txt files found under tools/ci_build")
        return

    for req_file in requirements_files:
        logging.info(f"Processing {req_file}")
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--only-binary=:all:",
            "-r",
            str(req_file),
            "--dest",
            str(local_wheels_dir),
        ]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            logging.error(f"Failed to download from {req_file}: {result.stderr}")
        else:
            logging.debug(f"Successfully downloaded from {req_file}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download Python wheels from requirements.txt files")

    parser.add_argument(
        "--local-wheels-dir",
        "-d",
        type=Path,
        default=DEFAULT_LOCAL_WHEELS_DIR,
        help="Directory to download wheels into",
    )

    return parser


if __name__ == "__main__":
    log_format = "[%(asctime)s] [fetch_python_wheels.py] [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, force=True)

    parser = make_parser()
    args = parser.parse_args()

    main(args.local_wheels_dir)
