# fetch_submodule_files_segment.py

import os
import shutil
import subprocess
import logging
from pathlib import Path
from typing import List, Optional
import configparser
import tempfile
import argparse

# --- Configuration ---
MS_REMOTE = "https://github.com/microsoft/onnxruntime.git"

# Assuming REPO_ROOT is the top-level directory of your main Git repository
REPO_ROOT = Path(__file__).parent.parent.parent.parent
GITMODULE_FILE = REPO_ROOT / ".gitmodules"
QCOM_ROOT = Path(__file__).parent
TMP_ORT = Path(__file__).parent / "tmp_ort"

# --- Helper Function for Running Git Commands ---
def _run_cmd(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    """
    Helper to run a subprocess command (specifically Git commands) with logging.

    Args:
        cmd: A list of strings representing the command and its arguments.
        cwd: The working directory for the command. Defaults to REPO_ROOT.
        check: If True, raise a CalledProcessError if the command returns a non-zero exit code.

    Returns:
        A CompletedProcess object.
    """
    if cwd is None:
        cwd = REPO_ROOT

    logging.info(f"Running command: {' '.join(cmd)} in {cwd}")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=check, cwd=cwd)

        if res.stdout:
            for line in res.stdout.strip().splitlines():
                logging.debug(f"STDOUT: {line}")
        if res.stderr:
            for line in res.stderr.strip().splitlines():
                logging.debug(f"STDERR: {line}") # Git often sends warnings to stderr

        return res
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}: {' '.join(e.cmd)}")
        if e.stdout:
            logging.error(f"STDOUT:\n{e.stdout.strip()}")
        if e.stderr:
            logging.error(f"STDERR:\n{e.stderr.strip()}")
        raise # Re-raise the exception to indicate failure

def is_submodule(path: Path):
    logging.info("Check submodule")
    res = _run_cmd(f"git ls-files --stage {path}".split())
    if "160000" in res.stdout:
        return True
    return False

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--upstream-branch",
        type=str,
        default="origin/main",
        help="Branch or sha to acquire submodules",
    )

    return parser

# --- Main Logic (Translation of CMD segment) ---
def main(args):
    UPSTREAM_BRANCH = args.upstream_branch
    logging.info(f"Starting Git operations for repository at: {REPO_ROOT}")
    logging.info(f"UPSTREAM_BRANCH: {UPSTREAM_BRANCH}")

    try:
        # Verify REPO_ROOT is a Git repository
        _run_cmd(["git", "rev-parse", "--is-inside-work-tree"], check=True)
        logging.info("Repository confirmed.")
    except subprocess.CalledProcessError:
        logging.critical(f"Error: '{REPO_ROOT}' is not a valid Git repository. Exiting.")
        return

    tmp_ort_str = str(TMP_ORT)
    try:
        if TMP_ORT.exists():
            logging.info(f"Found {tmp_ort_str} already exists, remove...")
            shutil.rmtree(TMP_ORT)
        # git clone
        logging.info(f"Running: git clone {MS_REMOTE} {tmp_ort_str}")
        _run_cmd(["git", "clone", MS_REMOTE, tmp_ort_str])

        # git -C temp_ort checkout %UPSTREAM_BRANCH% -f --recurse-submodules
        logging.info(f"Checking out: {UPSTREAM_BRANCH}")
        _run_cmd(["git", "-C", tmp_ort_str, "checkout", UPSTREAM_BRANCH])

        # git -C temp_ort submodule sync --recursive
        cmd = f"git -C {tmp_ort_str} submodule sync --recursive"
        logging.info(f"Running: {cmd}")
        _run_cmd(cmd.split())

        # git submodule update --init --force --recursive
        cmd = f"git -C {tmp_ort_str} submodule update --init --force --recursive"
        logging.info(f"Running: {cmd}")
        _run_cmd(cmd.split())

        config = configparser.ConfigParser()
        config.read(GITMODULE_FILE)
        def ignore_git(directory, files):
            return {f for f in files if f == ".git"}
        for section in config.sections():
            tgt_sm_path = Path(REPO_ROOT) / Path(config[section]["path"])
            src_sm_path = TMP_ORT / Path(config[section]["path"])
            if os.path.exists(tgt_sm_path):
                if is_submodule(tgt_sm_path):
                    # If the tgt_sm_path is a submodule, turn it into regular file
                    cmd = f"git submodule deinit -f {tgt_sm_path}"
                    _run_cmd(cmd.split())
                    cmd = f"git rm --cached {tgt_sm_path}"
                    _run_cmd(cmd.split())
                else:
                    logging.info(f"Remove the path: {tgt_sm_path}")
                    shutil.rmtree(tgt_sm_path)
            logging.info(f"Make directories: {tgt_sm_path}")
            os.makedirs(tgt_sm_path, exist_ok=True)
            logging.info(f"Copy from {src_sm_path} to {tgt_sm_path}")
            shutil.copytree(
                src_sm_path, tgt_sm_path,
                ignore=ignore_git, dirs_exist_ok=True
            )
        logging.info("Script segment completed successfully.")

    except subprocess.CalledProcessError:
        logging.error("A Git command failed. Please check the logs above for details.")
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}")

    if TMP_ORT.exists():
        logging.info(f"Remove {tmp_ort_str} ...")
        shutil.rmtree(TMP_ORT)

if __name__ == "__main__":
    # Configure logging to show INFO messages and above by default.
    # For more detailed output, set level to logging.DEBUG.
    log_format = "[%(asctime)s] [fetch_submodule_files.py] [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    parser = make_parser()
    args = parser.parse_args()

    main(args)
