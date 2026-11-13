# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Exercise cancellation through the real venv activation shell and child process."""

import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

import pytest

QCOM_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.skipif(os.name == "nt", reason="POSIX process signal forwarding")
@pytest.mark.parametrize("signum", (signal.SIGINT, signal.SIGTERM))
def test_cancel_reaches_executable_after_venv_activation(tmp_path, signum):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/activate").write_text("")
    ready = tmp_path / "ready"
    child = (
        "import os,signal,sys,time; from pathlib import Path; "
        "assert sys.argv[1] == 'argument with spaces'; "
        "signal.signal(signal.SIGINT, lambda *_: sys.exit(42)); "
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(42)); "
        f"Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(30)"
    )
    driver = (
        f"import sys; from pathlib import Path; sys.path.insert(0, {str(QCOM_ROOT)!r}); "
        "from ep_build.util import run_with_venv; "
        f"result=run_with_venv(Path({str(venv)!r}), "
        f"[sys.executable, '-c', {child!r}, 'argument with spaces'], check=False, capture_output=True); "
        "print(result.returncode)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", driver], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), "Child did not start"
        process.send_signal(signum)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
        assert stdout.strip() == "42", stdout + stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if ready.exists():
            with suppress(ProcessLookupError):
                os.kill(int(ready.read_text()), signal.SIGKILL)
