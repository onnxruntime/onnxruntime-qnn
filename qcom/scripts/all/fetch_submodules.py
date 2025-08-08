#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import os
import re
import argparse
import logging
import subprocess
from pathlib import Path
import tempfile

from package_manager import DEFAULT_SUBMODULE_BUNDLE_DIR, FileCache

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

def _create_gitmodules_bundle(cache_dir: Path, sm_configs: list[SubmoduleConfig]) -> list[Path]:
    """
    Clones each submodule to a temporary directory and creates a Git bundle (.bundle)
    from it, storing the bundle in the specified cache_dir.
    """
    cache_dir.mkdir(parents=True, exist_ok=True) # Ensure cache directory exists
    all_bundle_paths = []

    for sm_config in sm_configs:
        # Use a hash of the URL and path to create a stable, unique bundle filename
        # This helps avoid issues if multiple submodules have the same 'name' but different sources/paths.
        sanitized_name = re.sub(r'[^\w\-_.]', '_', sm_config.name)
        bundle_filename = f"{sanitized_name}.bundle"
        bundle_path = cache_dir / bundle_filename

        # Create a temporary directory for cloning
        with tempfile.TemporaryDirectory() as temp_clone_dir:
            temp_clone_path = Path(temp_clone_dir) / sm_config.name
            logging.info(f"Cloning '{sm_config.url}' to temporary location '{temp_clone_path}' to create bundle...")
            try:
                # Use subprocess to clone the repository to the temporary location
                # `git clone --bare` is ideal for bundling as it's a "bare" repo, not a working copy, which is smaller.
                # However, `git bundle create` generally expects a full repo or it can create from remote directly.
                # Let's clone a full working copy for simplicity and robust bundling.
                clone_cmd = [
                    "git", "clone", "--depth", "1", # Use --depth 1 if you only need the latest state
                    sm_config.url,
                    str(temp_clone_path)
                ]
                subprocess.run(clone_cmd, capture_output=True, text=True, check=True)
                logging.info(f"Successfully cloned '{sm_config.name}' to '{temp_clone_path}'.")

                # Now, create the bundle from the temporary clone
                bundle_create_cmd = [
                    "git", "bundle", "create",
                    str(bundle_path),
                    "HEAD" # Bundle the HEAD of the cloned repository
                ]
                logging.info(f"Creating bundle for '{sm_config.name}' to '{bundle_path}'...")
                subprocess.run(bundle_create_cmd, capture_output=True, text=True, check=True, cwd=str(temp_clone_path))
                logging.info(f"Successfully created bundle for '{sm_config.name}' at '{bundle_path}'.")
                all_bundle_paths.append(bundle_path)

            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to process '{sm_config.name}' (URL: {sm_config.url}):")
                logging.error(f"  Command: {' '.join(e.cmd)}")
                logging.error(f"  Stdout: {e.stdout.strip()}")
                logging.error(f"  Stderr: {e.stderr.strip()}")
            except Exception as e:
                logging.error(f"An unexpected error occurred for '{sm_config.name}': {e}")

    return all_bundle_paths


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
    all_bundle_paths = _create_gitmodules_bundle(cache_dir, submodules)

    if not all_bundle_paths:
        logging.info("No submodules bundle created.")
        return

    # _update_gitmodules_from_local(cache_dests)

    return

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch and update Git submodules, optionally creating a flat mirror of their working trees."
    )

    parser.add_argument(
        "--cache-dir",
        "-d",
        type=Path,
        default=DEFAULT_SUBMODULE_BUNDLE_DIR
    )

    return parser


if __name__ == "__main__":
    log_format = "[%(asctime)s] [fetch_submodule.py] [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, force=True)

    parser = make_parser()
    args = parser.parse_args()

    main(args.cache_dir)
