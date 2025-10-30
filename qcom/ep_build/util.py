# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import logging
import os
import platform
import shlex
import signal
import subprocess
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from types import FrameType


def default_parallelism() -> int:
    """A conservative number of processes across which to spread pytests desiring parallelism."""
    from .github import is_host_github_runner  # noqa: PLC0415

    cpu_count = os.cpu_count()
    if not cpu_count:
        return 1

    # In CI, saturate the machine
    if is_host_github_runner():
        return cpu_count

    # When running locally, leave a little CPU for other uses
    return max(1, int(cpu_count - 2))


# Convenience function for printing to the logger.
def echo(value: str) -> None:
    logging.info(value)


def get_env_bool(key: str, default: bool | None = None) -> bool | None:
    val = os.environ.get(key, None)
    if val is None:
        return default
    return str_to_bool(val)


def get_env_int(key: str, default: int | None = None) -> int | None:
    val = os.environ.get(key, None)
    if val is None:
        return default
    int_val = int(val)
    assert str(int_val) == val, f"Environment variable '{key}' is not a well-formed int."
    return int_val


def git_head_sha() -> str:
    return run_and_get_output(["git", "rev-parse", "HEAD"], quiet=True)


def have_root() -> bool:
    # mypy/pyright are generally unhappy here because these calls aren't always available.
    if is_host_windows():
        import ctypes  # noqa: PLC0415

        return ctypes.windll.shell32.IsUserAnAdmin() != 0  # type: ignore[attr-defined]
    return os.geteuid() == 0  # type:ignore[attr-defined]


def is_host_arm64() -> bool:
    return platform.machine().lower() == "arm64"


def is_host_in_ci():
    from .github import is_host_github_runner  # noqa: PLC0415

    return is_host_github_runner()


def is_host_user_linux():
    return is_host_linux() and not is_host_in_ci()


def is_host_linux():
    return platform.uname().system == "Linux"


def is_host_mac():
    return platform.uname().system == "Darwin"


def is_host_windows():
    return platform.uname().system == "Windows"


def is_host_x86_64():
    machine = platform.machine()
    return machine == "AMD64" or machine == "x86_64"


def process_output(process: subprocess.CompletedProcess):
    return process.stdout.decode("utf-8").strip()


def run(
    command: str | list[str],
    check: bool = True,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    stdout: int | None = None,
    capture_output: bool = False,
    quiet: bool = False,
) -> subprocess.CompletedProcess:
    return run_with_venv(
        venv=None,
        command=command,
        check=check,
        env=env,
        cwd=cwd,
        stdout=stdout,
        capture_output=capture_output,
        quiet=quiet,
    )


def run_and_get_output(
    command: str | list[str],
    check: bool = True,
    cwd: Path | None = None,
    capture_stderr: bool = False,
    quiet: bool = False,
) -> str:
    return run_with_venv_and_get_output(
        venv=None,
        command=command,
        check=check,
        cwd=cwd,
        capture_stderr=capture_stderr,
        quiet=quiet,
    )


def run_with_venv(
    venv: Path | None,
    command: str | list[str],
    check: bool = True,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    stdout: int | None = None,
    capture_output: bool = False,
    quiet: bool = False,
) -> subprocess.CompletedProcess:
    if venv is not None:
        env = dict(env) if env is not None else dict(os.environ)
        if is_host_windows():
            pathvars = [k for k in env if k.lower() == "path"]
            assert len(pathvars) == 1, f"Got wrong number of PATH-like variables in the environment: {pathvars}."
            pathvar = pathvars[0]
        else:
            pathvar = "PATH"
        env[pathvar] = f"{venv / VENV_BIN_RELPATH}{os.pathsep}{env[pathvar]}"
        env["VIRTUAL_ENV"] = str(venv.absolute())
        env["VIRTUAL_ENV_PROMPT"] = f"({venv.name})"
        command = command if isinstance(command, str) else shlex.join(command)
        prompt = f"{env['VIRTUAL_ENV_PROMPT']} $ "
    else:
        prompt = "$ "

    if not quiet:
        echo(f"{prompt}{command if isinstance(command, str) else shlex.join(command)}")

    if capture_output:
        stdout = subprocess.PIPE
        stderr = subprocess.PIPE
    else:
        stdout = stdout if stdout is not None else None
        stderr = None

    proc = subprocess.Popen(
        command,
        stdout=stdout,
        stderr=stderr,
        shell=isinstance(command, str),
        executable=SHELL_EXECUTABLE if isinstance(command, str) else None,
        env=env,
        cwd=cwd,
    )
    with TemporarySignalHandler(lambda sig, frame: proc.send_signal(sig)):
        outs, errs = proc.communicate()
        stdout_out = outs if stdout is not None else None
        stderr_out = errs if stderr is not None else None
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, command, stdout_out, stderr_out)
    return subprocess.CompletedProcess(command, proc.returncode, stdout_out, stderr_out)


def run_with_venv_and_get_output(
    venv: Path | None,
    command: str | list[str],
    check: bool = True,
    cwd: Path | None = None,
    capture_stderr: bool = False,
    quiet: bool = False,
) -> str:
    return process_output(
        run_with_venv(
            venv,
            command,
            stdout=subprocess.PIPE if not capture_stderr else None,
            check=check,
            cwd=cwd,
            capture_output=capture_stderr,
            quiet=quiet,
        )
    )


def str_to_bool(word: str) -> bool:
    return word.lower() in ["1", "true", "yes"]


def timestamp_brief() -> str:
    return datetime.now().strftime("%Y%m%d.%H%M%S")


class Colors:
    GREEN = "\033[0;32m" if not is_host_windows() else ""
    GREY = "\033[0;37m" if not is_host_windows() else ""
    RED = "\033[0;31m" if not is_host_windows() else ""
    RED_BOLD = "\033[0;1;31m" if not is_host_windows() else ""
    RED_REVERSED_VIDEO = "\033[0;7;31m" if not is_host_windows() else ""
    YELLOW = "\033[0;33m" if not is_host_windows() else ""
    OFF = "\033[0m" if not is_host_windows() else ""


# Caution: this class can also be found in qcom/scripts/all/qdc_runner.py. Consider copying any edits.
class TemporarySignalHandler:
    def __init__(self, handler: Callable[[int, FrameType | None], None], signum: int = signal.SIGINT) -> None:
        self.__signum = signum
        self.__handler = handler

    def __enter__(self) -> None:
        self.__prev_handler = signal.getsignal(self.__signum)
        signal.signal(self.__signum, self.__handler)

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        try:
            signal.signal(self.__signum, self.__prev_handler)
        except Exception as e:
            logging.warning(f"Failed to restore signal handler: {e}")


if is_host_windows():
    POWERSHELL_EXECUTABLE = run_and_get_output(["cmd", "/c", "where", "powershell.exe"], quiet=True)
    SHELL_EXECUTABLE = POWERSHELL_EXECUTABLE
else:
    BASH_EXECUTABLE = run_and_get_output(["which", "bash"], quiet=True)
    SHELL_EXECUTABLE = BASH_EXECUTABLE


DEFAULT_PYTHON_LINUX = Path("python3.10")

DEFAULT_PYTHON_WINDOWS = Path("python.exe")

DEFAULT_PYTHON = DEFAULT_PYTHON_WINDOWS if is_host_windows() else DEFAULT_PYTHON_LINUX
"""Different python distributions have different executable names. Use this for a reasonable default."""

MSFT_CI_REQUIREMENTS_RELPATH = (
    f"tools/ci_build/github/{'windows' if is_host_windows() else 'linux'}/python/requirements.txt"
)

REPO_ROOT = Path(__file__).parent.parent.parent

VENV_BIN_RELPATH = "Scripts" if is_host_windows() else "bin"

VENV_ACTIVATE_RELPATH = f"{VENV_BIN_RELPATH}/activate"
"""Where to find the bash script to source to activate a virtual environment."""
