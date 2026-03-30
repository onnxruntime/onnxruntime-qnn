# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Shared utilities for QNN EP Python test scripts."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import argparse
    import types

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUALCOMM_EP_REGISTRATION_NAME = "QNNExecutionProvider"

BackendT = Literal["cpu", "gpu", "htp"]

HTP_PERFORMANCE_MODES: tuple[str, ...] = (
    "burst",
    "balanced",
    "default",
    "high_performance",
    "high_power_saver",
    "low_balanced",
    "low_power_saver",
    "power_saver",
    "extreme_power_saver",
    "sustained_high_performance",
)

PROFILING_LEVELS: tuple[str, ...] = ("off", "basic", "detailed", "optrace")

HTP_FINALIZATION_MODES: tuple[str, ...] = ("0", "1", "2", "3")

# ---------------------------------------------------------------------------
# Optional onnx import
# ---------------------------------------------------------------------------

try:
    import onnx
    import onnx.numpy_helper

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# ---------------------------------------------------------------------------
# Module-level sentinel for EP registration
# ---------------------------------------------------------------------------

_ep_registered = False

# ---------------------------------------------------------------------------
# EP loading and registration
# ---------------------------------------------------------------------------


def load_qnn_ep() -> tuple[types.ModuleType, types.ModuleType]:
    """Import onnxruntime_qnn and onnxruntime, exiting with a clear message if unavailable."""
    try:
        import onnxruntime_qnn as qnn_ep
    except ImportError:
        print(
            "ERROR: onnxruntime_qnn is not installed.\nInstall the QNN EP wheel:\n  pip install onnxruntime_qnn-*.whl",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        import onnxruntime as ort
    except ImportError:
        print(
            "ERROR: onnxruntime is not installed.\nInstall it with:\n  pip install onnxruntime",
            file=sys.stderr,
        )
        sys.exit(1)

    return qnn_ep, ort


def register_qnn_ep(ort: types.ModuleType, qnn_ep: types.ModuleType) -> None:
    """Register the QNN EP plugin library (idempotent)."""
    global _ep_registered  # noqa: PLW0603
    if not _ep_registered:
        ort.register_execution_provider_library(
            QUALCOMM_EP_REGISTRATION_NAME,
            qnn_ep.get_library_path(),
        )
        _ep_registered = True


def unregister_qnn_ep(ort: types.ModuleType) -> None:
    """Unregister the QNN EP plugin library."""
    global _ep_registered  # noqa: PLW0603
    if _ep_registered:
        ort.unregister_execution_provider_library(QUALCOMM_EP_REGISTRATION_NAME)
        _ep_registered = False


def get_backend_path(qnn_ep: types.ModuleType, backend: BackendT) -> str:
    """Return the path to the QNN backend DLL for the given backend type."""
    match backend:
        case "cpu":
            return qnn_ep.get_qnn_cpu_path()
        case "gpu":
            return qnn_ep.get_qnn_gpu_path()
        case "htp":
            return qnn_ep.get_qnn_htp_path()
        case _:
            raise ValueError(f"Unknown backend: {backend!r}. Expected one of: cpu, gpu, htp")


def select_ep_device(ort: types.ModuleType, qnn_ep: types.ModuleType, backend: BackendT):
    """Select the OrtEpDevice matching the requested backend type."""
    ep_name = qnn_ep.get_ep_names()[0]
    all_devices = ort.get_ep_devices()
    qnn_devices = [d for d in all_devices if d.ep_name == ep_name]

    if not qnn_devices:
        raise RuntimeError(
            "No QNN EP devices found. Ensure the QNN EP library is registered and the device is available."
        )

    # Map backend type to OrtHardwareDeviceType
    target_hw_type = {
        "cpu": ort.OrtHardwareDeviceType.CPU,
        "gpu": ort.OrtHardwareDeviceType.GPU,
        "htp": ort.OrtHardwareDeviceType.NPU,
    }[backend]

    matching = [d for d in qnn_devices if d.device.type == target_hw_type]
    if not matching:
        available = [d.device.type for d in qnn_devices]
        raise RuntimeError(f"No QNN EP device found for backend '{backend}'. Available device types: {available}")

    return matching[0]


def build_session_options(
    ort: types.ModuleType,
    qnn_ep: types.ModuleType,
    args: argparse.Namespace,
    provider_options: dict[str, str] | None = None,
):
    """Build a fully configured SessionOptions from an argparse Namespace.

    The caller may pass additional provider_options to merge in.
    """
    sess_options = ort.SessionOptions()

    # Enable ORT verbose logging when --verbose is set
    if getattr(args, "verbose", False):
        sess_options.log_severity_level = 0  # 0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR

    # Thread counts
    if getattr(args, "intra_op_threads", None) is not None:
        sess_options.intra_op_num_threads = args.intra_op_threads
    if getattr(args, "inter_op_threads", None) is not None:
        sess_options.inter_op_num_threads = args.inter_op_threads

    # Build provider options dict
    ep_options: dict[str, str] = {"backend_path": get_backend_path(qnn_ep, args.backend)}

    if getattr(args, "htp_performance_mode", None) is not None:
        ep_options["htp_performance_mode"] = args.htp_performance_mode

    if getattr(args, "htp_graph_finalization_optimization_mode", None) is not None:
        ep_options["htp_graph_finalization_optimization_mode"] = args.htp_graph_finalization_optimization_mode

    if getattr(args, "enable_htp_fp16_precision", False):
        ep_options["enable_htp_fp16_precision"] = "1"

    if getattr(args, "vtcm_mb", None) is not None:
        ep_options["vtcm_mb"] = str(args.vtcm_mb)

    if getattr(args, "rpc_control_latency", None) is not None:
        ep_options["rpc_control_latency"] = str(args.rpc_control_latency)

    if getattr(args, "profiling_level", None) is not None:
        ep_options["profiling_level"] = args.profiling_level

    if getattr(args, "profiling_file_path", None) is not None:
        ep_options["profiling_file_path"] = str(args.profiling_file_path)

    # Merge any extra options from caller
    if provider_options:
        ep_options.update(provider_options)

    # Select device and add provider
    ep_device = select_ep_device(ort, qnn_ep, args.backend)
    sess_options.add_provider_for_devices([ep_device], ep_options)

    # Session config entries
    disable_fallback = getattr(args, "disable_cpu_fallback", False)
    if disable_fallback and args.backend != "cpu":
        sess_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    # Context caching (perf_test uses --context_cache flag; provider_test uses --enable_context)
    context_enabled = getattr(args, "context_cache", False) or getattr(args, "enable_context", False)
    if context_enabled:
        sess_options.add_session_config_entry("ep.context_enable", "1")

        ctx_path = getattr(args, "context_file_path", None)
        if ctx_path is None:
            # Derive default path beside the model
            model_path = Path(args.model)
            ctx_path = model_path.parent / f"{model_path.stem}_ctx.onnx"
        sess_options.add_session_config_entry("ep.context_file_path", str(ctx_path))

    return sess_options


# ---------------------------------------------------------------------------
# ONNX dtype mapping
# ---------------------------------------------------------------------------

# Maps ONNX TensorProto.DataType integer values to numpy dtypes
_ONNX_DTYPE_MAP: dict[int, np.dtype] = {
    1: np.dtype("float32"),  # FLOAT
    2: np.dtype("uint8"),  # UINT8
    3: np.dtype("int8"),  # INT8
    4: np.dtype("uint16"),  # UINT16
    5: np.dtype("int16"),  # INT16
    6: np.dtype("int32"),  # INT32
    7: np.dtype("int64"),  # INT64
    9: np.dtype("bool"),  # BOOL
    10: np.dtype("float16"),  # FLOAT16
    11: np.dtype("float64"),  # DOUBLE
    12: np.dtype("uint32"),  # UINT32
    13: np.dtype("uint64"),  # UINT64
}

# ORT type string to numpy dtype (used when reading from InferenceSession.get_inputs())
_ORT_TYPE_STR_MAP: dict[str, np.dtype] = {
    "tensor(float)": np.dtype("float32"),
    "tensor(float16)": np.dtype("float16"),
    "tensor(double)": np.dtype("float64"),
    "tensor(int8)": np.dtype("int8"),
    "tensor(int16)": np.dtype("int16"),
    "tensor(int32)": np.dtype("int32"),
    "tensor(int64)": np.dtype("int64"),
    "tensor(uint8)": np.dtype("uint8"),
    "tensor(uint16)": np.dtype("uint16"),
    "tensor(uint32)": np.dtype("uint32"),
    "tensor(uint64)": np.dtype("uint64"),
    "tensor(bool)": np.dtype("bool"),
    "tensor(string)": np.dtype("object"),
}


def onnx_dtype_to_numpy(elem_type: int) -> np.dtype:
    """Map an ONNX TensorProto.DataType integer to a numpy dtype."""
    if elem_type not in _ONNX_DTYPE_MAP:
        raise ValueError(
            f"Unsupported ONNX element type: {elem_type}. Supported types: {sorted(_ONNX_DTYPE_MAP.keys())}"
        )
    return _ONNX_DTYPE_MAP[elem_type]


def ort_type_str_to_numpy(type_str: str) -> np.dtype:
    """Map an ORT type string (e.g. 'tensor(float)') to a numpy dtype."""
    if type_str not in _ORT_TYPE_STR_MAP:
        raise ValueError(f"Unsupported ORT type string: {type_str!r}. Supported: {sorted(_ORT_TYPE_STR_MAP.keys())}")
    return _ORT_TYPE_STR_MAP[type_str]


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------


def generate_random_inputs(session) -> dict[str, np.ndarray]:
    """Generate random inputs matching the model's input shapes and dtypes.

    Dynamic dimensions (None or symbolic strings) are replaced with 1.
    """
    inputs: dict[str, np.ndarray] = {}
    for inp in session.get_inputs():
        name = inp.name
        dtype = ort_type_str_to_numpy(inp.type)

        # Resolve concrete shape: replace None / symbolic strings with 1
        shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in inp.shape]

        if dtype == np.dtype("bool"):
            arr = np.random.randint(0, 2, shape).astype(np.bool_)
        elif np.issubdtype(dtype, np.integer):
            arr = np.random.randint(0, 10, shape).astype(dtype)
        elif np.issubdtype(dtype, np.floating):
            arr = np.random.randn(*shape).astype(dtype)
        else:
            raise ValueError(f"Cannot generate random data for dtype {dtype} (input '{name}')")

        inputs[name] = arr

    return inputs


# ---------------------------------------------------------------------------
# Test data loading (.pb format)
# ---------------------------------------------------------------------------


def load_pb_tensor(path: Path) -> tuple[str, np.ndarray]:
    """Load a single .pb tensor file (ONNX TensorProto format).

    Returns (tensor_name, numpy_array).
    """
    if not ONNX_AVAILABLE:
        raise ImportError(
            "The 'onnx' package is required to load .pb test data files.\nInstall it with: pip install onnx"
        )
    proto = onnx.TensorProto.FromString(path.read_bytes())
    return proto.name, onnx.numpy_helper.to_array(proto)


def load_test_data_set(ds_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    """Load inputs and (optionally) expected outputs from a test_data_set_N directory.

    Returns (inputs_dict, outputs_dict). outputs_dict is None if no output_*.pb files exist.
    """
    input_files = sorted(ds_dir.glob("input_*.pb"))
    output_files = sorted(ds_dir.glob("output_*.pb"))

    inputs = dict(load_pb_tensor(f) for f in input_files)
    outputs = dict(load_pb_tensor(f) for f in output_files) if output_files else None

    return inputs, outputs


def find_test_data_sets(root: Path) -> list[Path]:
    """Return sorted list of test_data_set_* directories under root."""
    if not root.exists():
        return []
    return sorted(root.glob("test_data_set_*"))


# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------


def cosine_similarity(actual: np.ndarray, expected: np.ndarray) -> float:
    """Compute cosine similarity between two arrays (flattened)."""
    a = actual.flatten().astype(np.float64)
    b = expected.flatten().astype(np.float64)
    a = np.nan_to_num(a)
    b = np.nan_to_num(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0 if norm_a == norm_b else 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def configure_logging(verbose: bool) -> None:
    """Configure root logger level and format."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "[%(asctime)s] [%(levelname)s] %(message)s"
    logging.basicConfig(level=level, format=fmt, force=True)
