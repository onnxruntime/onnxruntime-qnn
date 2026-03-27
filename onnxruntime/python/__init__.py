# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os

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


def get_ep_names():
    return [EP_NAME]


def get_ep_name():
    return EP_NAME


def get_library_path():
    return os.path.join(LIB_DIR_FULL_PATH, "onnxruntime_providers_qnn.dll")


def get_qnn_cpu_path():
    return os.path.join(LIB_DIR_FULL_PATH, "QnnCpu.dll")


def get_qnn_gpu_path():
    return os.path.join(LIB_DIR_FULL_PATH, "QnnGpu.dll")


def get_qnn_htp_path():
    return os.path.join(LIB_DIR_FULL_PATH, "QnnHtp.dll")
