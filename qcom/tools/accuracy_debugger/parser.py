# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Command-line parser for the ORT/QNN-EP snooper.

Defines the ``python accuracy_debugger.py`` flags, parses them and validates/normalizes.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from typing import Final

from common import Algorithm
from comparators import ComparatorName, get_comparator

__all__ = ["PROG", "OrtQnnSnooperParser", "build_parser"]

PROG: Final = "python accuracy_debugger.py"
LOG_LEVEL_CHOICES: Final = ("ERROR", "WARN", "INFO", "DEBUG", "VERBOSE")


def _comma_list(value: str) -> list[str]:
    """Splits a comma separated option value into stripped, non-empty items.

    Args:
        value: Raw option value, e.g. ``"conv1, Gemm,,MatMul"``.

    Returns:
        The stripped items with empty entries dropped, e.g. ``["conv1", "Gemm", "MatMul"]``.
    """
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_elementwise_stats(value: str) -> dict[str, int]:
    """Parses a ``--dump_elementwise_stats`` value such as ``"conv1=2;Conv=0"``.

    Args:
        value: ``;`` separated ``<layer name or op type>=<non-negative int>`` pairs. Empty pieces are ignored.

    Returns:
        Mapping of layer name / op type to the allowed step value.

    Raises:
        ValueError: If a piece has no ``=``, an empty key, a non-integer step or a negative step.
    """
    stats: dict[str, int] = {}
    for raw_piece in value.split(";"):
        piece = raw_piece.strip()
        if not piece:
            continue
        layer, step = piece.rsplit("=", 1)
        layer = layer.strip()
        if not layer:
            raise ValueError(f"empty layer name in '{piece}'")
        steps = int(step)
        if steps < 0:
            raise ValueError(f"step value for '{layer}' must be >= 0, got {steps}")
        stats[layer] = steps
    return stats


class OrtQnnSnooperParser:
    """Argument parser for the ORT/QNN-EP snooper.

    Attributes:
        parser: The underlying ``argparse.ArgumentParser`` (all arguments are registered once at construction).
        required: Argument group titled "required arguments".
        optional: Argument group titled "optional arguments".
    """

    def __init__(self) -> None:
        """Creates the parser and registers every snooper flag."""
        self.parser = argparse.ArgumentParser(prog=PROG, formatter_class=argparse.RawDescriptionHelpFormatter)
        self.required = self.parser.add_argument_group("Required arguments")
        self.optional = self.parser.add_argument_group("Optional arguments")

        """Register the snooper flags on the required argument group."""
        self.required.add_argument("--qdq_model", type=str, required=True, help="Path to QDQ ONNX model.")
        self.required.add_argument(
            "--input_list", type=str, required=True, default=None, help="Path to text file containing input list."
        )

        """Register the snooper flags on the optional argument group."""
        self.optional.add_argument(
            "--op_trace_file",
            help="QNN framework op trace JSON, optionally enriched with original_sources.",
        )
        self.optional.add_argument(
            "--enable_framework_op_trace",
            action="store_true",
            help="Generate a QNN framework op trace for every target compilation, under the output directory.",
        )
        self.optional.add_argument(
            "--dump_qnn_ep_input_graph",
            action="store_true",
            help="Also dump the EP input graph next to each trace for offline name recovery "
            "(requires --enable_framework_op_trace).",
        )
        self.optional.add_argument(
            "--reference_model",
            type=str,
            required=False,
            default=None,
            help="Path to reference ONNX model. If not provided, given qdq_model will be run on "
            "ORT-CPU to generate reference outputs. "
            "Note: This argument is allowed only with oneshot algorithm (for layerwise algorithm "
            "qdq_model itself will be used to generate ORT-CPU reference outputs).",
        )
        self.optional.add_argument(
            "--algorithm",
            type=Algorithm,
            required=False,
            choices=[Algorithm.ONESHOT, Algorithm.LAYERWISE],
            default=Algorithm.ONESHOT,
            help="Algorithm to use to debug the model.",
        )
        self.optional.add_argument(
            "--comparator",
            type=ComparatorName,
            nargs="+",
            required=False,
            choices=list(ComparatorName),
            default=[ComparatorName.MSE],
            help="Comparator to use to compare tensors. For multiple comparators, "
            "specify as follows: --comparator mse cosine",
        )
        self.optional.add_argument(
            "--working_directory",
            type=str,
            required=False,
            default=None,
            help="Path to working directory. If not specified, a directory with name "
            "working_directory will be created in the current directory.",
        )
        self.optional.add_argument(
            "--output_directory",
            type=str,
            required=False,
            default=None,
            help="Name of the output directory. If not specified, a directory with name "
            "<timestamp> will be created in the working directory.",
        )
        self.optional.add_argument(
            "--golden_reference",
            type=str,
            required=False,
            default=None,
            help="The path of directory where golden reference tensor files are saved.",
        )
        self.optional.add_argument(
            "--set_intermediate_layers",
            type=_comma_list,
            required=False,
            default=None,
            help="Pass comma separated layer names/types which needs to be debugged. "
            "Can specify layer names like conv1_output, fc_layer_output or layer types like Conv, Gemm, MatMul. "
            "e.g., --set_intermediate_layers conv1_output,Gemm,MatMul,fc_layer_output",
        )
        self.optional.add_argument(
            "--set_cpu_layers",
            type=_comma_list,
            required=False,
            default=None,
            help="Pass comma separated layer names/types which needs to be executed on CPU Execution-Provider. "
            "This option is supported only with layerwise algorithm. "
            "Can specify layer names like conv1_output, fc_layer_output or layer types like Conv, Gemm, MatMul. "
            "e.g., --set_cpu_layers conv1_output,Gemm,MatMul,fc_layer_output",
        )
        self.optional.add_argument(
            "--dump_elementwise_stats",
            type=str,
            required=False,
            default=None,
            help="Dumps elementwise comparison statistics for the specified layer names/types "
            "along with their corresponding allowed step value. "
            "Can specify layer names like conv1_output, fc_layer_output or layer types like Conv, MatMul. "
            "Step value has to be an integer which represents allowed steps while comparing elementwise. "
            "e.g., --dump_elementwise_stats 'conv1_output=3;Gemm=1;MatMul=2;fc_layer_output=5'",
        )
        self.optional.add_argument(
            "--log_level",
            type=str.upper,
            required=False,
            default="INFO",
            choices=list(LOG_LEVEL_CHOICES),
            help="Enable verbose logging.",
        )

    def parse(self, args: Sequence[str] | None = None) -> argparse.Namespace:
        """Parses and validates the command-line arguments.

        Args:
            args: Arguments to parse; ``None`` parses ``sys.argv[1:]``.

        Returns:
            The verified and normalised namespace.

        Raises:
            SystemExit: For unknown flags, invalid choices or ``--help`` (argparse behaviour).
            argparse.ArgumentTypeError: If the post-parse validation fails.
        """
        parsed_args = self.parser.parse_args(None if args is None else list(args))
        return self._verify_and_update_parsed_args(parsed_args)

    def _verify_and_update_parsed_args(self, args: argparse.Namespace) -> argparse.Namespace:
        """Validates the parsed arguments and converts them to their runtime representation.

        Args:
            args: Namespace returned by ``parse_args``.

        Returns:
            The same namespace with ``input_list`` resolved, ``comparator`` holding comparator instances
            (deduplicated, in order) and ``dump_elementwise_stats`` parsed into ``dict[str, int]`` (or ``None``).

        Raises:
            argparse.ArgumentTypeError: If no input list is given, a file does not exist or
                ``--dump_elementwise_stats`` is malformed.
        """

        if not args.input_list:
            raise argparse.ArgumentTypeError("Argument --input_list must be supplied.")

        if not os.path.exists(args.qdq_model):
            raise argparse.ArgumentTypeError(f"QDQ model file '{args.qdq_model}' does not exist.")

        if not os.path.exists(args.input_list):
            raise argparse.ArgumentTypeError(f"Input list file '{args.input_list}' does not exist.")

        if args.op_trace_file and not os.path.isfile(args.op_trace_file):
            raise argparse.ArgumentTypeError(f"Op trace file '{args.op_trace_file}' does not exist.")
        if args.dump_qnn_ep_input_graph and not args.enable_framework_op_trace:
            raise argparse.ArgumentTypeError("--dump_qnn_ep_input_graph requires --enable_framework_op_trace.")

        args.comparator = [get_comparator(comparator) for comparator in dict.fromkeys(args.comparator)]

        if args.dump_elementwise_stats is not None:
            try:
                args.dump_elementwise_stats = _parse_elementwise_stats(args.dump_elementwise_stats) or None
            except Exception as e:
                raise argparse.ArgumentTypeError(
                    f"--dump_elementwise_stats value is not in valid format, parsing error: {e}"
                ) from e

        return args


def build_parser() -> argparse.ArgumentParser:
    """Returns the bare ``argparse`` parser with every snooper flag registered.

    Returns:
        ``OrtQnnSnooperParser().parser``.
    """
    return OrtQnnSnooperParser().parser
