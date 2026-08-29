#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""QNN EP performance benchmarking script.

Replicates the functionality of onnxruntime_perf_test.exe for the QNN Execution Provider.
Measures latency (min/max/mean/median/p99) and throughput for an ONNX model on QNN backends.

Usage examples:
    # Basic HTP benchmark with random inputs
    python onnxruntime_perf_test.py model.onnx

    # Run for 10 seconds on HTP with burst performance mode
    python onnxruntime_perf_test.py model.onnx --backend htp --duration_secs 10 \\
        --htp_performance_mode burst

    # Use specific input data files
    python onnxruntime_perf_test.py model.onnx --input_data input_0.pb input_1.pb

    # Enable context caching and save outputs
    python onnxruntime_perf_test.py model.onnx --context_cache --output_dir ./outputs
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Allow running this script directly from any working directory
sys.path.insert(0, str(Path(__file__).parent))
import qnn_ep_utils as utils

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="onnxruntime_perf_test.py",
        description="Benchmark an ONNX model on the QNN Execution Provider.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Positional
    parser.add_argument("model", type=Path, metavar="MODEL", help="Path to the ONNX model file.")

    # Backend
    parser.add_argument(
        "--backend",
        "-b",
        default="htp",
        choices=["cpu", "gpu", "htp"],
        help="QNN backend to use. (default: htp)",
    )

    # Run counts / duration
    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument(
        "--num_runs",
        "-r",
        type=int,
        default=10,
        metavar="N",
        help="Number of timed inference runs. Ignored if --duration_secs is set. (default: 10)",
    )
    run_group.add_argument(
        "--duration_secs",
        "-d",
        type=float,
        default=None,
        metavar="SECS",
        help="Run inference for this many seconds instead of a fixed count.",
    )
    parser.add_argument(
        "--warmup_runs",
        "-w",
        type=int,
        default=3,
        metavar="N",
        help="Number of warmup runs before timing begins. (default: 3)",
    )

    # Threading
    parser.add_argument(
        "--intra_op_threads",
        type=int,
        default=None,
        metavar="N",
        help="Number of intra-op threads for ORT. (default: ORT default)",
    )
    parser.add_argument(
        "--inter_op_threads",
        type=int,
        default=None,
        metavar="N",
        help="Number of inter-op threads for ORT. (default: ORT default)",
    )

    # HTP-specific options
    htp_group = parser.add_argument_group("HTP options")
    htp_group.add_argument(
        "--htp_performance_mode",
        choices=list(utils.HTP_PERFORMANCE_MODES),
        default=None,
        metavar="MODE",
        help=f"HTP performance mode. Choices: {', '.join(utils.HTP_PERFORMANCE_MODES)}",
    )
    htp_group.add_argument(
        "--htp_graph_finalization_optimization_mode",
        choices=list(utils.HTP_FINALIZATION_MODES),
        default=None,
        metavar="MODE",
        help="HTP graph finalization optimization level (0=default, 1=fast, 2=optimal, 3=most optimal).",
    )
    htp_group.add_argument(
        "--enable_htp_fp16_precision",
        action="store_true",
        help="Enable FP16 precision on HTP backend.",
    )
    htp_group.add_argument(
        "--vtcm_mb",
        type=str,
        default=None,
        metavar="MB",
        help="VTCM size in MB for HTP.",
    )
    htp_group.add_argument(
        "--rpc_control_latency",
        type=str,
        default=None,
        metavar="US",
        help="RPC control latency in microseconds.",
    )

    # Profiling
    prof_group = parser.add_argument_group("Profiling options")
    prof_group.add_argument(
        "--profiling_level",
        choices=list(utils.PROFILING_LEVELS),
        default=None,
        metavar="LEVEL",
        help=f"QNN profiling level. Choices: {', '.join(utils.PROFILING_LEVELS)}",
    )
    prof_group.add_argument(
        "--profiling_file_path",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to write QNN profiling output.",
    )

    # Context caching
    ctx_group = parser.add_argument_group("Context cache options")
    ctx_group.add_argument(
        "--context_cache",
        action="store_true",
        help="Enable QNN context caching (HTP only). Speeds up subsequent session creation.",
    )
    ctx_group.add_argument(
        "--context_file_path",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path for the context cache file. Defaults to <model_stem>_ctx.onnx beside the model.",
    )

    # CPU fallback
    parser.add_argument(
        "--disable_cpu_fallback",
        action="store_true",
        help="Disable CPU fallback for unsupported ops (non-CPU backends only).",
    )

    # Input / output
    io_group = parser.add_argument_group("Input/output options")
    io_group.add_argument(
        "--input_data",
        type=Path,
        nargs="*",
        default=None,
        metavar="PATH",
        help=(
            "Input data as .pb files (ONNX TensorProto), .npy files, or a directory "
            "containing input_*.pb files. If not provided, random inputs are generated."
        ),
    )
    io_group.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory to save inference outputs as output_N.npy files.",
    )
    io_group.add_argument(
        "--provider_options",
        "-i",
        nargs="*",
        default=None,
        metavar="KEY|VALUE",
        help=(
            "Extra QNN provider options as KEY|VALUE pairs (space-separated). "
            "Example: --provider_options dump_json_qnn_graph|1 json_qnn_graph_dir|output"
        ),
    )

    # Verbosity
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )

    return parser


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def load_inputs_from_args(args: argparse.Namespace, session) -> dict[str, np.ndarray]:
    """Load inputs from --input_data paths, or generate random inputs if not provided."""
    if not args.input_data:
        logging.warning(
            "No --input_data provided. Generating random inputs from model metadata. "
            "Results may not reflect real-world performance."
        )
        return utils.generate_random_inputs(session)

    paths = args.input_data

    # Single directory: look for input_*.pb inside it
    if len(paths) == 1 and paths[0].is_dir():
        pb_files = sorted(paths[0].glob("input_*.pb"))
        if pb_files:
            logging.info(f"Loading {len(pb_files)} input .pb files from {paths[0]}")
            return dict(utils.load_pb_tensor(f) for f in pb_files)
        # Fall through to try as a single file
        logging.warning(f"No input_*.pb files found in {paths[0]}. Generating random inputs.")
        return utils.generate_random_inputs(session)

    # .npy files: zip with session input names in order
    if all(p.suffix == ".npy" for p in paths):
        input_names = [inp.name for inp in session.get_inputs()]
        if len(paths) != len(input_names):
            logging.warning(
                f"Got {len(paths)} .npy files but model has {len(input_names)} inputs. Mapping by position."
            )
        inputs = {}
        for name, path in zip(input_names, paths, strict=False):
            logging.debug(f"Loading input '{name}' from {path}")
            inputs[name] = np.load(path)
        return inputs

    # .pb files: load each and build dict by tensor name
    if all(p.suffix == ".pb" for p in paths):
        logging.info(f"Loading {len(paths)} input .pb files")
        return dict(utils.load_pb_tensor(f) for f in paths)

    # Mixed or unknown: try .pb first, then .npy
    logging.warning("Mixed or unrecognized input file types. Attempting .pb load.")
    try:
        return dict(utils.load_pb_tensor(f) for f in paths)
    except Exception as e:
        logging.error(f"Failed to load inputs: {e}. Falling back to random inputs.")
        return utils.generate_random_inputs(session)


# ---------------------------------------------------------------------------
# Performance stats
# ---------------------------------------------------------------------------


@dataclass
class PerfStats:
    """Collected latency measurements from a benchmark run."""

    latencies_ms: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.latencies_ms)

    @property
    def min_ms(self) -> float:
        return min(self.latencies_ms)

    @property
    def max_ms(self) -> float:
        return max(self.latencies_ms)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.latencies_ms)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.latencies_ms)

    @property
    def p99_ms(self) -> float:
        sorted_lats = sorted(self.latencies_ms)
        idx = max(0, int(len(sorted_lats) * 0.99) - 1)
        return sorted_lats[idx]

    @property
    def throughput(self) -> float:
        """Runs per second."""
        return 1000.0 / self.mean_ms if self.mean_ms > 0 else 0.0


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_perf_test(session, inputs: dict[str, np.ndarray], args: argparse.Namespace) -> PerfStats:
    """Run warmup iterations then timed inference runs. Returns collected PerfStats."""
    logging.info(f"Running {args.warmup_runs} warmup iteration(s)...")
    for _ in range(args.warmup_runs):
        session.run(None, inputs)

    latencies: list[float] = []

    if args.duration_secs is not None:
        logging.info(f"Running timed benchmark for {args.duration_secs:.1f} seconds...")
        deadline = time.perf_counter() + args.duration_secs
        while time.perf_counter() < deadline:
            t0 = time.perf_counter()
            session.run(None, inputs)
            latencies.append((time.perf_counter() - t0) * 1000.0)
    else:
        logging.info(f"Running {args.num_runs} timed iteration(s)...")
        for _ in range(args.num_runs):
            t0 = time.perf_counter()
            session.run(None, inputs)
            latencies.append((time.perf_counter() - t0) * 1000.0)

    if not latencies:
        raise RuntimeError("No timed runs completed. Try increasing --duration_secs or --num_runs.")

    return PerfStats(latencies_ms=latencies)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_perf_results(stats: PerfStats, model_path: Path, backend: str) -> None:
    """Print a formatted performance results table."""
    sep = "=" * 50
    print(f"\n{sep}")
    print(f"  Performance Results: {model_path.name}")
    print(sep)
    print(f"  Backend   : {backend}")
    print(f"  Runs      : {stats.count}")
    print(f"  Min (ms)  : {stats.min_ms:.3f}")
    print(f"  Max (ms)  : {stats.max_ms:.3f}")
    print(f"  Mean (ms) : {stats.mean_ms:.3f}")
    print(f"  Median(ms): {stats.median_ms:.3f}")
    print(f"  P99 (ms)  : {stats.p99_ms:.3f}")
    print(f"  Throughput: {stats.throughput:.2f} runs/sec")
    print(sep)


def save_outputs(outputs: list[np.ndarray], output_names: list[str], output_dir: Path) -> None:
    """Save inference outputs as .npy files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, (name, arr) in enumerate(zip(output_names, outputs, strict=False)):
        out_path = output_dir / f"output_{i}.npy"
        np.save(out_path, arr)
        logging.info(f"Saved output '{name}' -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    utils.configure_logging(args.verbose)

    if not args.model.exists():
        logging.error(f"Model file not found: {args.model}")
        sys.exit(1)

    qnn_ep, ort = utils.load_qnn_ep()
    utils.register_qnn_ep(ort, qnn_ep)

    # Parse --provider_options KEY|VALUE pairs
    extra_provider_options: dict[str, str] = {}
    if args.provider_options:
        for item in args.provider_options:
            if "|" not in item:
                logging.error(f"Invalid --provider_options entry (expected KEY|VALUE): {item!r}")
                sys.exit(1)
            k, v = item.split("|", 1)
            extra_provider_options[k.strip()] = v.strip()

    # Auto-create any output directories referenced in provider options
    for dir_key in ("json_qnn_graph_dir", "profiling_file_path", "qnn_saver_path", "dump_qnn_ir_dlc_dir"):
        if dir_key in extra_provider_options:
            dir_path = Path(extra_provider_options[dir_key])
            dir_path.mkdir(parents=True, exist_ok=True)
            logging.debug(f"Created directory for {dir_key}: {dir_path}")

    try:
        sess_options = utils.build_session_options(ort, qnn_ep, args, provider_options=extra_provider_options)

        logging.info(f"Loading model: {args.model}")
        session = ort.InferenceSession(str(args.model), sess_options=sess_options)

        inputs = load_inputs_from_args(args, session)
        logging.info(f"Starting benchmark on backend='{args.backend}'")

        stats = run_perf_test(session, inputs, args)
        print_perf_results(stats, args.model, args.backend)

        if args.output_dir is not None:
            output_names = [o.name for o in session.get_outputs()]
            outputs = session.run(None, inputs)
            save_outputs(outputs, output_names, args.output_dir)
            logging.info(f"Outputs saved to {args.output_dir}")

        del session

    finally:
        utils.unregister_qnn_ep(ort)


if __name__ == "__main__":
    main()
