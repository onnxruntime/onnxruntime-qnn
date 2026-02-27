# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from .logger import get_logger
from .platform_helpers import is_linux, is_macOS, is_windows  # noqa: F401
from .qnn_helpers import parse_qnn_version_from_sdk_yaml  # noqa: F401
from .run import run  # noqa: F401
from .vcpkg_helpers import (  # noqa: F401
    generate_android_triplets,
    generate_linux_triplets,
    generate_macos_triplets,
    generate_vcpkg_triplets_for_emscripten,
    generate_windows_triplets,
)

# see if we can make the pytorch helpers available.
import importlib.util

have_torch = importlib.util.find_spec("torch")
if have_torch:
    from .pytorch_export_helpers import infer_input_info  # noqa: F401
