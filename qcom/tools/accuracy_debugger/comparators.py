# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Tensor comparators used to score target outputs against reference outputs.

Each comparator reduces a (reference, inference) pair of numpy arrays to a single Python float. The per-comparator
math, including the sentinel values returned on degenerate inputs, is a faithful port of the legacy QAIRT comparator
package. Comparators that legacy converted with ``float()`` keep that conversion; ``mae`` (which legacy returned as a
numpy float32) goes through :func:`~.common.to_py_scalar` so its report text keeps the float32 shortest repr.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from common import to_py_scalar

__all__ = [
    "COMPARATORS",
    "COMPARATOR_REGISTRY",
    "AdjustedRtolAtolComparator",
    "AverageComparator",
    "Comparator",
    "ComparatorName",
    "ComparisonError",
    "CosineComparator",
    "KLDComparator",
    "L1ErrorComparator",
    "L1NormComparator",
    "L2NormComparator",
    "MAEComparator",
    "MSEComparator",
    "MSERelComparator",
    "RtolAtolComparator",
    "SNRComparator",
    "STDComparator",
    "TensorShapeError",
    "TopKComparator",
    "get_comparator",
]

_MIN_TOL: Final = 1e-10
_MAX_TOL: Final = 1e-2


class ComparatorName(str, Enum):
    """Names of the supported comparators (also the CSV column names)."""

    L1_NORM = "l1_norm"
    L2_NORM = "l2_norm"
    AVERAGE = "average"
    COSINE = "cosine"
    STANDARD_DEVIATION = "standard_deviation"
    MSE = "mse"
    SNR = "snr"
    KLD = "kl_divergence"
    RTOL_ATOL = "rtol_atol"
    L1_ERROR = "l1_error"
    MSE_REL = "mse_rel"
    TOPK = "topk"
    ADJUSTED_RTOL_ATOL = "adjusted_rtol_atol"
    MAE = "mae"

    def __str__(self) -> str:
        """Returns the plain value so f-strings and argparse help print ``mse`` and not ``ComparatorName.MSE``."""
        return self.value


COMPARATORS = ComparatorName
"""Legacy alias of :class:`ComparatorName`."""


class TensorShapeError(Exception):
    """Raised when the two tensors of a pair do not have the same size or shape."""


class ComparisonError(Exception):
    """Raised when a comparison fails for any reason."""


class Comparator(ABC):
    """Interface to compare two lists of tensors pair by pair.

    Subclasses implement :meth:`_compare`. By convention the first tensor of a pair is the reference and the second
    one is the inference (target) output.
    """

    def __init__(self, name: str, tol: float = 1e-5) -> None:
        """Initialises the comparator.

        Args:
            name: Name of the comparator.
            tol: Tolerance; the tensors are rounded to ``int(abs(round(log10(tol))))`` decimals before comparison.

        Raises:
            ValueError: If ``tol`` is outside ``[1e-10, 1e-2]``.
        """
        if not _MIN_TOL <= tol <= _MAX_TOL:
            raise ValueError(f"tol must be between {_MIN_TOL} and {_MAX_TOL}, got {tol}.")
        self.name = name
        self._tol = tol
        self._decimals = int(abs(round(math.log10(tol))))

    @property
    def name(self) -> str:
        """Name of the comparator."""
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    def compare(self, tensors1: Sequence[np.ndarray], tensors2: Sequence[np.ndarray]) -> list[float]:
        """Compares two lists of tensors pair by pair.

        Args:
            tensors1: Reference tensors.
            tensors2: Inference tensors, in the same order as ``tensors1``.

        Returns:
            One comparison score per pair.

        Raises:
            ValueError: If either list is empty or the lists have different lengths.
            TensorShapeError: If the two tensors of a pair have different sizes.
            ComparisonError: If the comparison of a pair fails (including a shape mismatch at equal size).
        """
        if not (len(tensors1) and len(tensors2)):
            raise ValueError("Empty data list.")
        if len(tensors1) != len(tensors2):
            raise ValueError("Number of tensors in both lists must match.")

        errors: list[float] = []
        for t1, t2 in zip(tensors1, tensors2, strict=True):
            if t1.size != t2.size:
                raise TensorShapeError("tensor1 and tensor2 are of different size.")
            try:
                errors.append(self._compare(t1, t2))
            except Exception as e:
                raise ComparisonError(f"Failed to compare tensors: {e}") from e
        return errors

    def _check_shape(self, t1: np.ndarray, t2: np.ndarray) -> None:
        """Validates that both tensors have the same shape.

        Raises:
            TensorShapeError: If the shapes differ.
        """
        if t1.shape != t2.shape:
            raise TensorShapeError("Mismatch in shape of the input arrays")

    def _flatten_round(self, t1: np.ndarray, t2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Flattens both tensors and rounds them to the comparator's number of decimals."""
        return np.around(t1.flatten(), decimals=self._decimals), np.around(t2.flatten(), decimals=self._decimals)

    @abstractmethod
    def _compare(self, reference: np.ndarray, inference: np.ndarray) -> float:
        """Computes the score of one pair of tensors.

        Args:
            reference: Reference tensor.
            inference: Inference tensor.

        Returns:
            The comparison score as a Python float.
        """


class L1NormComparator(Comparator):
    """Relative L1 norm of the difference: ``||I - R||_1 / ||R||_1``."""

    def __init__(self, tol: float = 1e-5) -> None:
        """Initialises the comparator; see :class:`Comparator`."""
        super().__init__(ComparatorName.L1_NORM.value, tol)

    def _compare(self, reference: np.ndarray, inference: np.ndarray) -> float:
        self._check_shape(reference, inference)
        ref, inf = self._flatten_round(reference, inference)
        ref_l1 = np.linalg.norm(ref, ord=1)
        if float(ref_l1) == 0.0:
            return 0.0 if np.all(inf == 0.0) else 1.0
        return float(np.linalg.norm(inf - ref, ord=1) / ref_l1)


class L2NormComparator(Comparator):
    """Relative L2 norm of the difference: ``||I - R||_2 / ||R||_2``."""

    def __init__(self, tol: float = 1e-5) -> None:
        """Initialises the comparator; see :class:`Comparator`."""
        super().__init__(ComparatorName.L2_NORM.value, tol)

    def _compare(self, reference: np.ndarray, inference: np.ndarray) -> float:
        self._check_shape(reference, inference)
        ref, inf = self._flatten_round(reference, inference)
        ref_l2 = np.linalg.norm(ref, ord=2)
        if float(ref_l2) == 0.0:
            return 0.0 if np.all(inf == 0.0) else 1.0
        return float(np.linalg.norm(inf - ref, ord=2) / ref_l2)


class AverageComparator(Comparator):
    """Mean absolute difference relative to ``max(mean|R|, mean|I|)``."""

    def __init__(self, tol: float = 1e-5) -> None:
        """Initialises the comparator; see :class:`Comparator`."""
        super().__init__(ComparatorName.AVERAGE.value, tol)

    def _compare(self, reference: np.ndarray, inference: np.ndarray) -> float:
        self._check_shape(reference, inference)
        ref, inf = self._flatten_round(reference, inference)
        ref_avg = max(np.average(np.absolute(ref)), np.average(np.absolute(inf)))
        if float(ref_avg) == 0.0:
            return 0.0 if np.all(inf == 0) else 1.0
        return float(np.average(np.absolute(ref - inf)) / ref_avg)


class CosineComparator(Comparator):
    """Cosine similarity ``dot(R, I) / (||R||_2 ||I||_2)``; 0 when either norm is 0."""

    def __init__(self, tol: float = 1e-5) -> None:
        """Initialises the comparator; see :class:`Comparator`."""
        super().__init__(ComparatorName.COSINE.value, tol)

    def _compare(self, reference: np.ndarray, inference: np.ndarray) -> float:
        self._check_shape(reference, inference)
        ref, inf = self._flatten_round(reference, inference)
        ref_l2 = np.linalg.norm(ref, ord=2)
        inf_l2 = np.linalg.norm(inf, ord=2)
        if float(ref_l2) == 0.0 or float(inf_l2) == 0.0:
            return 0.0
        return float(np.dot(ref, inf) / (ref_l2 * inf_l2))


class STDComparator(Comparator):
    """Relative difference of the population standard deviations: ``|std(I) - std(R)| / |std(R)|``."""

    def __init__(self, tol: float = 1e-5) -> None:
        """Initialises the comparator; see :class:`Comparator`."""
        super().__init__(ComparatorName.STANDARD_DEVIATION.value, tol)

    def _compare(self, reference: np.ndarray, inference: np.ndarray) -> float:
        self._check_shape(reference, inference)
        ref, inf = self._flatten_round(reference, inference)
        if np.all(ref == inf):
            return 0.0
        std_ref = np.std(ref)
        std_inf = np.std(inf)
        if float(std_ref) == 0:
            return 100.0
        return float(abs(std_inf - std_ref) / abs(std_ref))


class MSEComparator(Comparator):
    """Mean squared error ``mean((R - I)^2)``; 100.0 when the reference is all zeros and the inference is not."""

    def __init__(self, tol: float = 1e-5) -> None:
        """Initialises the comparator; see :class:`Comparator`."""
        super().__init__(ComparatorName.MSE.value, tol)

    def _compare(self, reference: np.ndarray, inference: np.ndarray) -> float:
        self._check_shape(reference, inference)
        ref, inf = self._flatten_round(reference, inference)
        if float(np.mean(np.square(ref))) == 0.0:
            return 0.0 if np.all(inf == 0) else 100.0
        return float(np.mean(np.square(ref - inf)))


class SNRComparator(Comparator):
    """Signal-to-noise ratio in dB ``10 log10(mean(R^2) / mean((R - I)^2))``; 100.0 on a perfect match."""

    def __init__(self, tol: float = 1e-5) -> None:
        """Initialises the comparator; see :class:`Comparator`."""
        super().__init__(ComparatorName.SNR.value, tol)

    def _compare(self, reference: np.ndarray, inference: np.ndarray) -> float:
        self._check_shape(reference, inference)
        ref, inf = self._flatten_round(reference, inference)
        mse_diff = np.mean(np.square(ref - inf))
        if float(mse_diff) == 0.0:
            return 100.0
        mse_ref = np.mean(np.square(ref))
        with np.errstate(divide="ignore"):
            return float(10.0 * np.log10(mse_ref / mse_diff))


def _kl_divergence_base2(p: np.ndarray, q: np.ndarray) -> float:
    """Numpy equivalent of ``scipy.stats.entropy(p, q, base=2)`` (scipy 1.10 semantics).

    scipy normalises both distributions in the input dtype, evaluates ``rel_entr`` in double precision, rounds each
    element back to the input dtype and sums. This function reproduces those steps bit for bit.
    """
    with np.errstate(all="ignore"):
        p = 1.0 * p / np.sum(p)
        q = 1.0 * q / np.sum(q)
        p64 = p.astype(np.float64)
        q64 = q.astype(np.float64)
        rel64 = np.where(
            (p64 > 0) & (q64 > 0),
            p64 * np.log(p64 / q64),
            np.where((p64 == 0) & (q64 >= 0), 0.0, np.inf),
        )
        rel64 = np.where(np.isnan(p64) | np.isnan(q64), np.nan, rel64)
        rel = rel64.astype(p.dtype)
        return float(np.sum(rel) / np.log(2))


class KLDComparator(Comparator):
    """Kullback-Leibler divergence (base 2) of the L2-normalised tensors.

    Returns 0.0 on a perfect match and 1.0 when either L2 norm is 0. Tensors with negative values usually give
    ``inf`` or ``nan`` (legacy behaviour).
    """

    def __init__(self, tol: float = 1e-5) -> None:
        """Initialises the comparator; see :class:`Comparator`."""
        super().__init__(ComparatorName.KLD.value, tol)

    def _compare(self, reference: np.ndarray, inference: np.ndarray) -> float:
        self._check_shape(reference, inference)
        ref, inf = self._flatten_round(reference, inference)
        ref_norm = np.linalg.norm(ref, ord=2)
        inf_norm = np.linalg.norm(inf, ord=2)
        if np.all(ref == inf):
            return 0.0
        if float(ref_norm) == 0.0 or float(inf_norm) == 0.0:
            return 1.0
        return _kl_divergence_base2(ref / ref_norm, inf / inf_norm)


class RtolAtolComparator(Comparator):
    """Percentage (0-100) of elements for which ``np.isclose(R, I, atol, rtol)`` is False.

    Unlike the other comparators the tensors are neither flattened nor rounded (legacy behaviour).
    """

    def __init__(self, tol: float = 1e-5, rtol_margin: float = 1e-2, atol_margin: float = 1e-2) -> None:
        """Initialises the comparator.

        Args:
            tol: See :class:`Comparator`.
            rtol_margin: Relative tolerance passed to ``np.isclose``.
            atol_margin: Absolute tolerance passed to ``np.isclose``.
        """
        super().__init__(ComparatorName.RTOL_ATOL.value, tol)
        self._rtol_margin = rtol_margin
        self._atol_margin = atol_margin

    def _compare(self, reference: np.ndarray, inference: np.ndarray) -> float:
        self._check_shape(reference, inference)
        match_array = np.isclose(reference, inference, atol=self._atol_margin, rtol=self._rtol_margin)
        percent_not_close = (len(match_array) - np.count_nonzero(match_array)) / len(match_array)
        return float(percent_not_close * 100)


class L1ErrorComparator(Comparator):
    """Sum of absolute differences scaled by ``multiplier * scale``."""

    def __init__(self, tol: float = 1e-5, multiplier: float = 1.0, scale: float = 1.0) -> None:
        """Initialises the comparator.

        Args:
            tol: See :class:`Comparator`.
            multiplier: Factor applied to the L1 sum.
            scale: Second factor applied to the L1 sum.
        """
        super().__init__(ComparatorName.L1_ERROR.value, tol)
        self._multiplier = multiplier
        self._scale = scale

    def _compare(self, reference: np.ndarray, inference: np.ndarray) -> float:
        self._check_shape(reference, inference)
        ref, inf = self._flatten_round(reference, inference)
        l1_sum = np.sum(np.absolute(ref - inf))
        # Legacy (numpy 1.x) promoted the float32 sum to float64 when multiplying by the Python floats; doing the
        # product in Python floats keeps that result under numpy 2 (NEP 50) as well.
        return float(l1_sum) * float(self._multiplier) * float(self._scale)


class MSERelComparator(Comparator):
    """Relative mean squared error ``mean((R - I)^2) / mean(R^2)``."""

    def __init__(self, tol: float = 1e-5) -> None:
        """Initialises the comparator; see :class:`Comparator`."""
        super().__init__(ComparatorName.MSE_REL.value, tol)

    def _compare(self, reference: np.ndarray, inference: np.ndarray) -> float:
        self._check_shape(reference, inference)
        ref, inf = self._flatten_round(reference, inference)
        mse_ref = np.mean(np.square(ref))
        if float(mse_ref) == 0.0:
            return 0.0 if np.all(inf == 0) else 100.0
        return float(np.mean(np.square(ref - inf)) / mse_ref)


class TopKComparator(Comparator):
    """Percentage (0-100) of the top-k indices of the inference that are not top-k indices of the reference."""

    def __init__(self, tol: float = 1e-5, k: int = 1, ordered: bool = False) -> None:
        """Initialises the comparator.

        Args:
            tol: See :class:`Comparator`.
            k: Number of top indices to compare.
            ordered: If True, indices are compared position by position; otherwise as sets.
        """
        super().__init__(ComparatorName.TOPK.value, tol)
        self._k = k
        self._ordered = ordered

    def _compare(self, reference: np.ndarray, inference: np.ndarray) -> float:
        self._check_shape(reference, inference)
        ref, inf = self._flatten_round(reference, inference)
        top_ref = np.flip(np.argsort(ref, kind="stable")[-self._k :])
        top_inf = np.flip(np.argsort(inf, kind="stable")[-self._k :])
        if not self._ordered:
            top_ref.sort()
            top_inf.sort()
        if self._ordered:
            num_diff = int(np.count_nonzero(top_ref != top_inf))
        else:
            num_diff = int(np.count_nonzero(~np.isin(top_inf, top_ref)))
        return float(num_diff / min(self._k, len(ref)) * 100)


class AdjustedRtolAtolComparator(Comparator):
    """Percentage (0-100) of elements not close, with an absolute tolerance derived from the reference range.

    ``atol = levels_num * (max(R) - min(R)) / 255`` and ``rtol = 0.1``, relative to the reference.
    """

    def __init__(self, tol: float = 1e-5, levels_num: int = 4) -> None:
        """Initialises the comparator.

        Args:
            tol: See :class:`Comparator`.
            levels_num: Number of 8-bit quantisation steps allowed as absolute tolerance.

        Raises:
            ValueError: If ``levels_num`` is smaller than 1.
        """
        if levels_num < 1:
            raise ValueError(f"levels_num must be >= 1, got {levels_num}.")
        super().__init__(ComparatorName.ADJUSTED_RTOL_ATOL.value, tol)
        self._levels_num = levels_num

    def _calculate_margins(self, tensor: np.ndarray) -> tuple[float, float]:
        """Returns the adjusted ``(atol, rtol)`` margins for the given reference tensor."""
        min_value = np.min(tensor)
        max_value = np.max(tensor)
        step = (max_value - min_value) / 255
        return self._levels_num * step, 1e-1

    def _compare(self, reference: np.ndarray, inference: np.ndarray) -> float:
        self._check_shape(reference, inference)
        ref, inf = self._flatten_round(reference, inference)
        atol_margin, rtol_margin = self._calculate_margins(ref)
        match_array = np.isclose(inf, ref, atol=atol_margin, rtol=rtol_margin)
        percent_not_close = (len(match_array) - np.count_nonzero(match_array)) / len(match_array)
        return float(percent_not_close * 100)


class MAEComparator(Comparator):
    """Mean absolute error ``mean(|R - I|)``."""

    def __init__(self, tol: float = 1e-5) -> None:
        """Initialises the comparator; see :class:`Comparator`."""
        super().__init__(ComparatorName.MAE.value, tol)

    def _compare(self, reference: np.ndarray, inference: np.ndarray) -> float:
        self._check_shape(reference, inference)
        ref, inf = self._flatten_round(reference, inference)
        return float(to_py_scalar(np.mean(np.absolute(ref - inf))))


COMPARATOR_REGISTRY: Final[Mapping[ComparatorName, type[Comparator]]] = MappingProxyType(
    {
        ComparatorName.L1_NORM: L1NormComparator,
        ComparatorName.L2_NORM: L2NormComparator,
        ComparatorName.AVERAGE: AverageComparator,
        ComparatorName.COSINE: CosineComparator,
        ComparatorName.STANDARD_DEVIATION: STDComparator,
        ComparatorName.MSE: MSEComparator,
        ComparatorName.SNR: SNRComparator,
        ComparatorName.KLD: KLDComparator,
        ComparatorName.RTOL_ATOL: RtolAtolComparator,
        ComparatorName.L1_ERROR: L1ErrorComparator,
        ComparatorName.MSE_REL: MSERelComparator,
        ComparatorName.TOPK: TopKComparator,
        ComparatorName.ADJUSTED_RTOL_ATOL: AdjustedRtolAtolComparator,
        ComparatorName.MAE: MAEComparator,
    }
)


def get_comparator(comparator: ComparatorName | str, **kwargs: Any) -> Comparator:
    """Creates a comparator by name.

    Args:
        comparator: A :class:`ComparatorName` member or its string value (e.g. ``"mse"``).
        **kwargs: Constructor keyword arguments of the comparator (e.g. ``k=5`` for topk).

    Returns:
        A new comparator instance.

    Raises:
        NotImplementedError: If the name is not a supported comparator.
    """
    try:
        name = ComparatorName(comparator)
    except ValueError as e:
        raise NotImplementedError(f"Comparator '{comparator}' is not supported.") from e
    return COMPARATOR_REGISTRY[name](**kwargs)
