#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


# ---------- Base interface ----------
class Metric(ABC):
    """
    Base metric interface.

    Each metric must implement:
    - compute(actual, expected): returns the numeric metric value
    - is_pass(value): returns True/False if the metric meets the configured criteria (e.g., threshold)
    """

    name: str

    @abstractmethod
    def compute(self, actual: np.ndarray, expected: np.ndarray) -> float: ...

    @abstractmethod
    def is_pass(self, value: float) -> bool: ...

    def evaluate(self, actual: np.ndarray, expected: np.ndarray) -> tuple[float, bool]:
        """Convenience method: compute + pass/fail in one call."""
        value = self.compute(actual, expected)
        return value, self.is_pass(value)


# ---------- Concrete metrics ----------
@dataclass
class MSEMetric(Metric):
    """
    Mean Squared Error (lower is better).
    threshold: maximum allowable MSE (pass if mse <= threshold).
    """

    threshold: float
    name: str = "mse"

    def compute(self, actual: np.ndarray, expected: np.ndarray) -> float:
        actual = actual.astype(float)
        expected = expected.astype(float)
        if actual.shape != expected.shape:
            raise ValueError(f"MSE: shape mismatch {actual.shape} vs {expected.shape}")
        diff = actual - expected
        mse = np.mean(np.square(diff))
        return float(mse)

    def is_pass(self, value: float) -> bool:
        if self.threshold is None:
            return True
        print("threshold:", self.threshold)
        return value <= self.threshold


@dataclass
class AssertAllCloseMetric(Metric):
    """
    Asserts predictions are close to targets element-wise (NumPy-style).

    rtol: relative tolerance
    atol: absolute tolerance
    """

    rtol: float
    atol: float
    name: str = "assert_allclose"

    def compute(self, actual: np.ndarray, expected: np.ndarray) -> float:
        # We return a summary scalar—fraction of elements within tolerance (1.0 means all close).
        actual = actual.astype(float)
        expected = expected.astype(float)
        if actual.shape != expected.shape:
            raise ValueError(f"AssertAllClose: shape mismatch {actual.shape} vs {expected.shape}")
        close = np.allclose(actual, expected, rtol=self.rtol, atol=self.atol, equal_nan=True)
        return float(np.mean(close))

    def is_pass(self, value: float) -> bool:
        # Pass only if all elements are within tolerance.
        print("rtol:", self.rtol, " atol:", self.atol)
        return value == 1.0


@dataclass
class CosineSimilarityMetric(Metric):
    """
    Cosine similarity between actual and expected (higher is better).

    threshold: minimum allowable cosine similarity (pass if cos_sim >= threshold).
    """

    threshold: float
    name: str = "cosine_similarity"

    def compute(self, actual: np.ndarray, expected: np.ndarray) -> float:
        # Flatten the values for actual and expected
        actual, expected = actual.flatten(), expected.flatten()
        actual = actual.astype(float)
        expected = expected.astype(float)

        if actual.shape != expected.shape:
            raise ValueError(f"CosineSimilarity: shape mismatch {actual.shape} vs {expected.shape}")

        # Replace NaN or Inf values with 0.
        actual = np.nan_to_num(actual, nan=0.0, posinf=0.0, neginf=0.0)
        expected = np.nan_to_num(expected, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize the vectors to avoid overflow.
        actual_norm = np.linalg.norm(actual)
        expected_norm = np.linalg.norm(expected)

        if actual_norm == 0 or expected_norm == 0:
            # Zero norm encountered: cosine similarity is undefined; use 0.0 by convention.
            return 0.0

        actual_normalized = actual / actual_norm
        expected_normalized = expected / expected_norm

        # Calculate dot product and norms.
        num = np.dot(actual_normalized, expected_normalized.T)
        denom = np.linalg.norm(actual_normalized) * np.linalg.norm(expected_normalized)

        if denom == 0.0:
            # Zero denominator encountered: cosine similarity is undefined; use 0.0 by convention.
            return 0.0

        similarity_score = num / denom

        if np.isnan(similarity_score):
            return 0.0

        return similarity_score

    def is_pass(self, value: float) -> bool:
        print("threshold:", self.threshold)
        return value >= self.threshold


@dataclass
class SNRMetric(Metric):
    """
    Signal-to-Noise Ratio between actual and expected (higher is better).

    threshold: minimum allowable SNR (pass if snr >= threshold).
    """

    threshold: float
    batch_size: int
    name: str = "snr"

    def compute(self, actual: np.ndarray, expected: np.ndarray) -> float:
        # Flatten the values for actual and expected
        actual, expected = actual.flatten(), expected.flatten()
        actual = actual.astype(float)
        expected = expected.astype(float)
        scores = []

        if actual.shape != expected.shape:
            raise ValueError(f"CosineSimilarity: shape mismatch {actual.shape} vs {expected.shape}")
        if len(actual) % self.batch_size != 0:
            raise ValueError(
                f"Length of actual {len(actual)} is not perfectly divisible by batch_size {self.batch_size}"
            )

        non_batched_output_length = len(actual) // self.batch_size
        for i in range(self.batch_size):
            start = i * non_batched_output_length
            signal = expected[start : start + non_batched_output_length]
            signal = signal * signal
            noise = (
                expected[start : start + non_batched_output_length] - actual[start : start + non_batched_output_length]
            )
            noise = noise * noise
            snr = 10 * (np.log10(np.sum(signal) / (np.sum(noise) + np.finfo(float).eps)))
            scores.append(snr)

        return min(scores)

    def is_pass(self, value: float) -> bool:
        print("threshold:", self.threshold)
        return value >= self.threshold
