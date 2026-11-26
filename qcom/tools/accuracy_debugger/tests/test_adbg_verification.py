# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Unit tests for verification."""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest
from comparators import (
    Comparator,
    CosineComparator,
    MSEComparator,
    TopKComparator,
)
from verification import verify


@pytest.fixture
def ref_outputs() -> dict[str, np.ndarray]:
    return {
        "a": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "b": np.array([0.5, 0.25], dtype=np.float64),
    }


@pytest.fixture
def inf_outputs() -> dict[str, np.ndarray]:
    return {
        "a": np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32),
        "b": np.array([0.5, 0.25], dtype=np.float16),
    }


def test_result_shape_and_keys(ref_outputs: dict, inf_outputs: dict, logger: logging.Logger) -> None:
    result = verify(ref_outputs, inf_outputs, logger, [MSEComparator(), CosineComparator()])
    assert list(result) == [("a", "a"), ("b", "b")]
    scores = result[("a", "a")]
    assert set(scores) == {"mse", "cosine", "op_type", "dimensions"}
    assert scores["mse"] == 0.25
    assert type(scores["mse"]) is float
    assert scores["op_type"] is None
    assert scores["dimensions"] == (2, 2)
    assert result[("b", "b")]["mse"] == 0.0


def test_default_comparator_is_mse(ref_outputs: dict, inf_outputs: dict) -> None:
    result = verify(ref_outputs, inf_outputs)
    assert set(result[("a", "a")]) == {"mse", "op_type", "dimensions"}


def test_result_follows_inference_order(ref_outputs: dict, inf_outputs: dict) -> None:
    reordered = {"b": inf_outputs["b"], "a": inf_outputs["a"]}
    assert list(verify(ref_outputs, reordered)) == [("b", "b"), ("a", "a")]


def test_mapping_is_applied(ref_outputs: dict) -> None:
    inference = {"tgt_a": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), "b": ref_outputs["b"]}
    result = verify(ref_outputs, inference, tensor_mapping={"tgt_a": "a", "b": None})
    assert list(result) == [("tgt_a", "a"), ("b", "b")]
    assert result[("tgt_a", "a")]["mse"] == 0.0


@pytest.mark.parametrize("mapping", [None, {}])
def test_empty_mapping_is_identity(ref_outputs: dict, inf_outputs: dict, mapping: dict | None) -> None:
    result = verify(ref_outputs, inf_outputs, tensor_mapping=mapping)
    assert list(result) == [("a", "a"), ("b", "b")]


def test_missing_reference_is_skipped(ref_outputs: dict, mock_logger) -> None:
    inference = {"a": ref_outputs["a"], "unknown": np.ones(2, np.float32)}
    result = verify(ref_outputs, inference, mock_logger, tensor_mapping={"a": "a"})
    assert list(result) == [("a", "a")]
    mock_logger.debug.assert_called_once()
    assert "unknown" in mock_logger.debug.call_args[0][0]
    mock_logger.warning.assert_not_called()


def test_mapping_to_missing_reference_is_skipped(ref_outputs: dict) -> None:
    result = verify(ref_outputs, {"x": np.ones(2, np.float32)}, tensor_mapping={"x": "nope"})
    assert result == {}


def test_comparator_failure_gives_nan_and_warning(mock_logger) -> None:
    reference = {"t": np.ones(4, np.float32)}
    inference = {"t": np.ones(5, np.float32)}
    result = verify(reference, inference, mock_logger, [MSEComparator(), TopKComparator()])
    scores = result[("t", "t")]
    assert math.isnan(scores["mse"])
    assert math.isnan(scores["topk"])
    assert scores["dimensions"] == (5,)
    assert mock_logger.warning.call_count == 2
    message = mock_logger.warning.call_args_list[0][0][0]
    assert message == "Tensor comparison failed with error 'tensor1 and tensor2 are of different size.'"


def test_generic_comparator_exception_gives_nan(caplog: pytest.LogCaptureFixture) -> None:
    class _Boom(Comparator):
        def __init__(self) -> None:
            super().__init__("boom")

        def _compare(self, reference: np.ndarray, inference: np.ndarray) -> float:
            raise RuntimeError("kaput")

    caplog.set_level(logging.WARNING, logger="verification")
    result = verify({"t": np.ones(2)}, {"t": np.ones(2)}, comparators=[_Boom(), MSEComparator()])
    assert math.isnan(result[("t", "t")]["boom"])
    assert result[("t", "t")]["mse"] == 0.0
    assert "Tensor comparison failed with error 'Failed to compare tensors: kaput'" in caplog.text


def test_tensors_are_flattened_and_cast_to_float32() -> None:
    seen: list[tuple[np.ndarray, np.ndarray]] = []

    class _Spy(Comparator):
        def __init__(self) -> None:
            super().__init__("spy")

        def _compare(self, reference: np.ndarray, inference: np.ndarray) -> float:
            seen.append((reference, inference))
            return 0.0

    verify({"t": np.arange(6, dtype=np.int64).reshape(2, 3)}, {"t": np.arange(6, dtype=np.uint8)}, comparators=[_Spy()])
    (ref, inf) = seen[0]
    assert ref.dtype == np.float32
    assert inf.dtype == np.float32
    assert ref.shape == (6,)
    assert inf.shape == (6,)


@pytest.mark.parametrize(("reference", "inference"), [(None, {}), ({}, None), (None, None)])
def test_none_dict_raises(reference: dict | None, inference: dict | None) -> None:
    with pytest.raises(ValueError, match="No input tensors found to perform verification!"):
        verify(reference, inference)


def test_empty_dicts_give_empty_result() -> None:
    assert verify({}, {}) == {}
