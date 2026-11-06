# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Verification of target outputs against reference outputs with a set of comparators."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np
from comparators import Comparator, MSEComparator

__all__ = ["VerifierResult", "verify"]

VerifierResult: TypeAlias = dict[tuple[str, str], dict[str, Any]]
"""``{(target_name, reference_name): {<comparator name>: score, "op_type": None, "dimensions": shape}}``."""


def verify(
    reference_output: Mapping[str, np.ndarray] | None,
    inference_output: Mapping[str, np.ndarray] | None,
    logger: logging.Logger | None = None,
    comparators: Sequence[Comparator] | None = None,
    tensor_mapping: Mapping[str, str | None] | None = None,
) -> VerifierResult:
    """Compares every inference tensor with its reference tensor using the given comparators.

    Args:
        reference_output: Reference tensors by name.
        inference_output: Inference (target) tensors by name; the result follows this order.
        logger: Logger; defaults to this module's logger.
        comparators: Comparators to apply; defaults to ``[MSEComparator()]``.
        tensor_mapping: Maps target names to reference names. Unmapped names (and an empty or ``None`` mapping)
            use the target name itself.

    Returns:
        The scores per ``(target_name, reference_name)`` pair. Target tensors without a reference are skipped. A
        comparator that fails stores ``nan`` for that pair and logs a warning.

    Raises:
        ValueError: If either output dictionary is ``None``.
    """
    if reference_output is None or inference_output is None:
        raise ValueError("No input tensors found to perform verification!")
    logger = logger or logging.getLogger(__name__)
    if comparators is None:
        comparators = [MSEComparator()]
    mapping = tensor_mapping or {}

    result: VerifierResult = {}
    for inf_name, inf_tensor in inference_output.items():
        ref_name = mapping.get(inf_name) or inf_name
        if ref_name not in reference_output:
            logger.debug(f"Skipping target tensor '{inf_name}': reference tensor '{ref_name}' not found.")
            continue

        ref_flat = np.asarray(reference_output[ref_name]).flatten().astype(np.float32)
        inf_flat = np.asarray(inf_tensor).flatten().astype(np.float32)

        scores: dict[str, Any] = {}
        for comparator in comparators:
            try:
                scores[comparator.name] = comparator.compare([ref_flat], [inf_flat])[0]
            except Exception as e:
                logger.warning(f"Tensor comparison failed with error '{e}'")
                scores[comparator.name] = float("nan")

        scores["op_type"] = None
        scores["dimensions"] = np.shape(inf_tensor)
        result[(inf_name, ref_name)] = scores
    return result
