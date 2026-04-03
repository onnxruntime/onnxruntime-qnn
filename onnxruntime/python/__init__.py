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
