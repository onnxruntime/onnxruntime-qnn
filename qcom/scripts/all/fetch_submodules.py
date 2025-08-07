#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import os
import argparse
import logging
import subprocess
from pathlib import Path

from package_manager import DEFAULT_SUBMODULE_CACHE_DIR, FileCache

from git import Repo

# Assuming REPO_ROOT is the top-level directory of your main Git repository
# where .gitmodules resides.
REPO_ROOT = Path(__file__).parent.parent.parent.parent # Adjust based on your actual script location

GITMODULES_FILE = REPO_ROOT / ".gitmodules"

class SubmoduleConfig:
    """Represents a single submodule's configuration from .gitmodules."""
    def __init__(self, name: str, path: str, url: str):
        self.name = name
        self.path = Path(path)
        self.url = url

    def __repr__(self):
        return f"SubmoduleConfig(name='{self.name}', path='{self.path}', url='{self.url}')"


def _parse_gitmodules() -> list[SubmoduleConfig]:
    """
    Parses the .gitmodules file to extract submodule information.
    """
    if not GITMODULES_FILE.exists():
        logging.warning(f"'.gitmodules' file not found at {GITMODULES_FILE}. No submodules to fetch.")
        return []

    all_submodules = []
    def recurse(submodules, base_path):
        for sm in submodules:
            sub_path = os.path.join(base_path, sm.path)
            all_submodules.append(SubmoduleConfig(sm.name, sub_path, sm.url))
            if sm.module_exists():
                with sm.module() as mod:
                    recurse(mod.submodules, sub_path)
    with Repo(REPO_ROOT) as repo:
        recurse(repo.submodules, ".")
    return all_submodules


def _create_gitmodules_cache(cache_dir, sm_configs: list[SubmoduleConfig]) -> list[str]:
    all_cache_dests = []
    for sm_config in sm_configs:
        dest_path = os.path.join(cache_dir / sm_config.path)
        all_cache_dests.append(dest_path)
        logging.info(f"Cloning '{sm_config.url}' to '{dest_path}' for external cache...")
        try:
            # Use GitPython's Repo.clone_from to clone the submodule's repository
            # This creates a full working tree clone.
            # We clone the default branch (usually 'main' or 'master').
            if os.path.exists(dest_path):
                logging.info(f"{sm_config.path} already exists in external cache {dest_path}")
            else:
                Repo.clone_from(sm_config.url, dest_path)
                logging.info(f"Successfully cloned {sm_config.name} to external cache {dest_path}")
        except Exception as e:
            logging.error(f"Failed to clone {sm_config.name} to external cache: {e.stderr}")

    return all_cache_dests


def _update_gitmodules_from_local(cache_dests: list[str]) -> None:
    # submodule sync
    cmd = ["git", "submodule", "sync", "--recursive"]
    logging.info(f"Syncing submodule...")
    logging.info(f" ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    logging.info(res.stdout)

    # submodule update
    cmd = ["git", "submodule", "update", "--init", "--force", "--recursive"]
    for cache_dest in cache_dests:
        cmd += ["--reference", cache_dest]
    logging.info(f"Updating submodule with local cache...")
    logging.info(f" ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    logging.info(res.stdout)

    return

def main(cache_dir: Path) -> None:
    """
    Main function to fetch and potentially 'cache' submodules.
    """
    submodules = _parse_gitmodules()

    if not submodules:
        logging.info("No submodules found or .gitmodules file is empty.")
        return

    # Clone submodules to local
    cache_dests = _create_gitmodules_cache(cache_dir, submodules)

    if not cache_dests:
        logging.info("No submodules local cache created.")
        return

    _update_gitmodules_from_local(cache_dests)

    return

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch and update Git submodules, optionally creating a flat mirror of their working trees."
    )

    parser.add_argument(
        "--cache-dir",
        "-d",
        type=Path,
        default=DEFAULT_SUBMODULE_CACHE_DIR
    )

    return parser


if __name__ == "__main__":
    log_format = "[%(asctime)s] [fetch_submodule.py] [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, force=True)

    parser = make_parser()
    args = parser.parse_args()

    main(args.cache_dir)
