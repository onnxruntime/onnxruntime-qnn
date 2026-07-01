# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import sys

from . import build_and_package_info  # noqa: F401
from .build_and_package_info import __version__  # noqa: F401

EP_NAME = "QNNExecutionProvider"

LIB_DIR_FULL_PATH = os.path.dirname(os.path.abspath(__file__))

# Platform-aware library loading
try:
    from .platform_loader import setup_library_path

    _lib_dir_path = setup_library_path()
    LIB_DIR_FULL_PATH = os.path.abspath(_lib_dir_path)
except ImportError:
    # Silently fall back to default LIB_DIR_FULL_PATH if platform loader is unavailable
    pass


def _configure_dsp_skel_search_path():
    """Ensure the QNN HTP DSP skel (libQnnHtpV*Skel.so) can be found by the
    fastRPC loader on Windows.

    The DSP-side fastRPC loader searches ``ADSP_LIBRARY_PATH`` (plus a few fixed
    system paths) for the skel. Unlike the Windows DLL loader, it does NOT
    automatically include the directory a DLL was loaded from -- so when
    ``QnnHtp.dll`` is loaded dynamically from deep under ``site-packages``, the
    skel that sits next to it is not on the search path. Native executables
    (qnn-net-run, onnxruntime_perf_test) work without this because they run from
    the same directory as the skel; the Python wheel does not.

    Without this, skel load fails (``qnn_open 0x80000406`` / error 1002), QNN
    falls back to the HNRD user-driver path, and newer ops (e.g. GroupQueryAttention,
    MatMulNBits) fail op validation and land on CPU.

    We prepend the package directory to ``ADSP_LIBRARY_PATH`` (idempotently, and
    preserving any user-provided value).
    """
    if sys.platform != "win32":
        # Linux/Android package and load the skel via different mechanisms.
        return
    skel_dir = LIB_DIR_FULL_PATH
    existing = os.environ.get("ADSP_LIBRARY_PATH", "")
    parts = existing.split(";") if existing else []
    if skel_dir not in parts:
        os.environ["ADSP_LIBRARY_PATH"] = ";".join([skel_dir, *parts])


_configure_dsp_skel_search_path()


def _lib_name(base):
    if sys.platform == "win32":
        return f"{base}.dll"
    else:  # linux / android
        return f"lib{base}.so"


def get_ep_names():
    return [EP_NAME]


def get_ep_name():
    return EP_NAME


def get_library_path():
    return os.path.join(LIB_DIR_FULL_PATH, _lib_name("onnxruntime_providers_qnn"))


def get_qnn_cpu_path():
    return os.path.join(LIB_DIR_FULL_PATH, _lib_name("QnnCpu"))


def get_qnn_gpu_path():
    return os.path.join(LIB_DIR_FULL_PATH, _lib_name("QnnGpu"))


def get_qnn_htp_path():
    return os.path.join(LIB_DIR_FULL_PATH, _lib_name("QnnHtp"))
