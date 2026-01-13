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
        return value >= self.threshold
