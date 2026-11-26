# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Unit tests for comparators."""

from __future__ import annotations

import math

import numpy as np
import pytest
from comparators import (
    COMPARATOR_REGISTRY,
    COMPARATORS,
    AdjustedRtolAtolComparator,
    AverageComparator,
    Comparator,
    ComparatorName,
    ComparisonError,
    CosineComparator,
    KLDComparator,
    L1ErrorComparator,
    L1NormComparator,
    L2NormComparator,
    MAEComparator,
    MSEComparator,
    MSERelComparator,
    RtolAtolComparator,
    SNRComparator,
    STDComparator,
    TensorShapeError,
    TopKComparator,
    get_comparator,
)

EXPECTED_CLASSES: dict[ComparatorName, type[Comparator]] = {
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
ALL_NAMES = list(ComparatorName)

# Seeded float32 data (np.random.default_rng(42)); expected values were captured once from the legacy QAIRT
# comparators (numpy 1.26.4, scipy 1.10.1) and are compared exactly through their repr.
_RNG = np.random.default_rng(42)
GOLDEN_REF = _RNG.random((2, 5), dtype=np.float32).flatten()
GOLDEN_INF = (GOLDEN_REF + _RNG.normal(0, 0.01, GOLDEN_REF.shape)).astype(np.float32)
GOLDEN_VALUES: dict[str, str] = {
    "l1_norm": "0.013708462007343769",
    "l2_norm": "0.014165553264319897",
    "average": "0.013678308576345444",
    "cosine": "0.999902069568634",
    "standard_deviation": "0.0018238474149256945",
    "mse": "5.409729783423245e-05",
    "snr": "36.97532892227173",
    "kl_divergence": "0.000643224015672701",
    "rtol_atol": "10.0",
    "l1_error": "0.05931994318962097",
    "mse_rel": "0.0002006628637900576",
    "topk": "0.0",
    "adjusted_rtol_atol": "0.0",
    "mae": "0.0059319944",
}

# (all-zero vs all-zero, constant 2.0 vs [1, 2, 3, 4], all-zero vs [1, 2, 3, 4]); legacy values.
EDGE_VALUES: dict[str, tuple[float, float, float]] = {
    "l1_norm": (0.0, 0.5, 1.0),
    "l2_norm": (0.0, 0.6123724579811096, 1.0),
    "average": (0.0, 0.4000000059604645, 1.0),
    "cosine": (0.0, 0.9128708839416504, 0.0),
    "standard_deviation": (0.0, 100.0, 100.0),
    "mse": (0.0, 1.5, 100.0),
    "snr": (100.0, 4.259687662124634, -math.inf),
    "kl_divergence": (0.0, 0.17568757637010501, 1.0),
    "rtol_atol": (0.0, 75.0, 100.0),
    "l1_error": (0.0, 4.0, 10.0),
    "mse_rel": (0.0, 0.375, 100.0),
    "topk": (0.0, 0.0, 0.0),
    "adjusted_rtol_atol": (0.0, 75.0, 100.0),
    "mae": (0.0, 1.0, 2.5),
}

ZEROS = np.zeros(4, dtype=np.float32)
CONSTANT = np.full(4, 2.0, dtype=np.float32)
RAMP = np.array([1, 2, 3, 4], dtype=np.float32)


# --------------------------------------------------------------------------------------------- registry / factory
def test_registry_has_all_14_comparators() -> None:
    assert len(ComparatorName) == 14
    assert set(COMPARATOR_REGISTRY) == set(ComparatorName)
    assert dict(COMPARATOR_REGISTRY) == EXPECTED_CLASSES


@pytest.mark.parametrize("name", ALL_NAMES, ids=str)
def test_get_comparator_by_enum_and_string(name: ComparatorName) -> None:
    by_enum = get_comparator(name)
    by_str = get_comparator(name.value)
    assert type(by_enum) is EXPECTED_CLASSES[name]
    assert type(by_str) is EXPECTED_CLASSES[name]
    assert by_enum.name == name.value
    assert type(by_enum.name) is str


def test_get_comparator_unknown_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="Comparator 'bogus' is not supported."):
        get_comparator("bogus")


def test_get_comparator_forwards_kwargs() -> None:
    comp = get_comparator("topk", k=3, ordered=True)
    assert isinstance(comp, TopKComparator)
    assert comp._k == 3
    assert comp._ordered is True


def test_legacy_alias_and_member_names() -> None:
    assert COMPARATORS is ComparatorName
    assert COMPARATORS.STANDARD_DEVIATION.value == "standard_deviation"
    assert COMPARATORS.KLD.value == "kl_divergence"
    assert str(ComparatorName.MSE) == "mse"
    assert f"{ComparatorName.SNR}" == "snr"


def test_name_setter() -> None:
    comp = MSEComparator()
    comp.name = "custom"
    assert comp.name == "custom"


# --------------------------------------------------------------------------------------------- tol validation
@pytest.mark.parametrize("tol", [1e-11, 0.1, 0.0, -1e-5])
def test_tol_out_of_range_raises(tol: float) -> None:
    with pytest.raises(ValueError, match="tol"):
        MSEComparator(tol=tol)


@pytest.mark.parametrize(("tol", "decimals"), [(1e-5, 5), (1e-10, 10), (1e-2, 2), (3e-4, 4)])
def test_tol_sets_decimals(tol: float, decimals: int) -> None:
    assert MSEComparator(tol=tol)._decimals == decimals


def test_adjusted_rtol_atol_levels_num_validation() -> None:
    with pytest.raises(ValueError, match="levels_num"):
        AdjustedRtolAtolComparator(levels_num=0)


# --------------------------------------------------------------------------------------------- compare() contract
@pytest.mark.parametrize("name", ALL_NAMES, ids=str)
def test_empty_data_list(name: ComparatorName) -> None:
    with pytest.raises(ValueError, match="Empty data list."):
        get_comparator(name).compare([], [])


@pytest.mark.parametrize("name", ALL_NAMES, ids=str)
def test_list_length_mismatch(name: ComparatorName) -> None:
    with pytest.raises(ValueError, match="Number of tensors in both lists must match."):
        get_comparator(name).compare([np.arange(9)] * 3, [np.arange(9)] * 2)


@pytest.mark.parametrize("name", ALL_NAMES, ids=str)
def test_mismatched_size_raises_tensor_shape_error(name: ComparatorName) -> None:
    with pytest.raises(TensorShapeError, match="tensor1 and tensor2 are of different size."):
        get_comparator(name).compare([np.ones(5, np.float32)], [np.ones(6, np.float32)])


@pytest.mark.parametrize("name", ALL_NAMES, ids=str)
def test_equal_size_different_shape_raises_comparison_error(name: ComparatorName) -> None:
    comp = get_comparator(name)
    with pytest.raises(ComparisonError, match="Mismatch in shape of the input arrays") as excinfo:
        comp.compare([np.ones((2, 3), np.float32)], [np.ones((3, 2), np.float32)])
    assert isinstance(excinfo.value.__cause__, TensorShapeError)
    with pytest.raises(TensorShapeError, match="Mismatch in shape of the input arrays"):
        comp._compare(np.ones((5, 6), np.float32), np.ones(6, np.float32))


@pytest.mark.parametrize("name", ALL_NAMES, ids=str)
def test_compare_list_returns_python_floats(name: ComparatorName) -> None:
    list1 = [np.arange(i + j * 10, i + j * 10 + 10).reshape(-1, 1) for j in range(3) for i in range(10)]
    list2 = [np.arange(i + j * 10, i + j * 10 + 10).reshape(-1, 1) for j in range(3) for i in range(10)]
    result = get_comparator(name).compare(list1, list2)
    assert isinstance(result, list)
    assert len(result) == len(list1)
    assert all(type(v) is float for v in result)


@pytest.mark.parametrize("name", ALL_NAMES, ids=str)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.uint8])
def test_every_comparator_returns_python_float(name: ComparatorName, dtype: type) -> None:
    ref = np.array([1, 5, 3, 7, 2], dtype=dtype)
    inf = np.array([2, 4, 3, 6, 2], dtype=dtype)
    assert type(get_comparator(name).compare([ref], [inf])[0]) is float


# --------------------------------------------------------------------------------------------- golden values
@pytest.mark.parametrize("name", ALL_NAMES, ids=str)
def test_golden_values(name: ComparatorName) -> None:
    value = get_comparator(name).compare([GOLDEN_REF], [GOLDEN_INF])[0]
    # NumPy 2.x scalar promotion changes the final float32 rounding of SNR.
    if name == ComparatorName.SNR:
        assert value == pytest.approx(float(GOLDEN_VALUES[name.value]), rel=2e-7)
    else:
        assert repr(value) == GOLDEN_VALUES[name.value]


@pytest.mark.parametrize("name", ALL_NAMES, ids=str)
def test_edge_values(name: ComparatorName) -> None:
    comp = get_comparator(name)
    expected_zero, expected_const, expected_zero_ref = EDGE_VALUES[name.value]
    assert comp.compare([ZEROS], [ZEROS])[0] == expected_zero
    assert comp.compare([CONSTANT], [RAMP])[0] == pytest.approx(expected_const, rel=2e-7)
    with np.errstate(all="ignore"):
        assert comp.compare([ZEROS], [RAMP])[0] == expected_zero_ref


@pytest.mark.parametrize("name", ALL_NAMES, ids=str)
def test_identical_ones(name: ComparatorName) -> None:
    ones = np.ones(5, dtype=np.float32)
    value = get_comparator(name).compare([ones], [ones.copy()])[0]
    expected = {"snr": 100.0, "cosine": 1.0}.get(name.value, 0.0)
    assert value == expected


def test_mae_float32_shortest_repr() -> None:
    value = MAEComparator().compare([GOLDEN_REF], [GOLDEN_INF])[0]
    assert type(value) is float
    assert str(value) == "0.0059319944"
    ref = np.array([0.0, 0.0], dtype=np.float32)
    inf = np.array([0.00474, 0.00474], dtype=np.float32)
    assert str(MAEComparator().compare([ref], [inf])[0]) == "0.00474"


def test_l1_error_multiplier_and_scale() -> None:
    ref = np.array([1.0, 2.0], dtype=np.float32)
    inf = np.array([2.0, 4.0], dtype=np.float32)
    assert L1ErrorComparator(multiplier=2.0, scale=0.5).compare([ref], [inf])[0] == 3.0
    assert L1ErrorComparator(multiplier=3.0).compare([ref], [inf])[0] == 9.0


def test_rtol_atol_margins() -> None:
    ref = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    inf = np.array([1.05, 2.0, 3.0, 4.5], dtype=np.float32)
    assert RtolAtolComparator().compare([ref], [inf])[0] == 50.0
    assert RtolAtolComparator(rtol_margin=0.2, atol_margin=0.2).compare([ref], [inf])[0] == 0.0


# --------------------------------------------------------------------------------------------- KLD
def test_kld_matches_scipy_golden() -> None:
    rng = np.random.default_rng(7)
    ref = rng.random(64, dtype=np.float32)
    inf = rng.random(64, dtype=np.float32)
    # Captured once with scipy.stats.entropy(p, q, base=2) on the L2-normalised, rounded tensors (scipy 1.10.1).
    assert KLDComparator().compare([ref], [inf])[0] == 0.758443060348359


def test_kld_signed_data_gives_inf() -> None:
    ref = np.array([-1.0, 2.0, 3.0], dtype=np.float32)
    inf = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert KLDComparator().compare([ref], [inf])[0] == math.inf


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    ("reference", "inference", "expected"),
    [
        ([1, 1], [1, 3], 0.2075187496394219),
        ([1, 3], [1, 1], 0.18872187554086717),
        ([0, 1], [1, 1], 1.0),
        ([1, 0], [0, 1], math.inf),
        ([1, 3], [2, 6], 0.0),
    ],
    ids=["uniform-to-skewed", "skewed-to-uniform", "zero-probability", "disjoint", "scale-invariant"],
)
def test_kld_reference_values(reference, inference, expected, dtype) -> None:
    # Fixed base-2 KL values: sum(p * log2(p / q)), with each distribution summing to one.
    # The first two cases also check that KL divergence is asymmetric.
    ref = np.array(reference, dtype=dtype)
    inf = np.array(inference, dtype=dtype)
    assert KLDComparator().compare([ref], [inf])[0] == pytest.approx(expected, rel=1e-6, abs=1e-8)


# --------------------------------------------------------------------------------------------- topk
def test_topk_ordered_and_unordered() -> None:
    ref = np.array([1, 3, 2, 4, 5])
    inf = np.array([1, 2, 3, 4, 5])
    assert TopKComparator(k=3, ordered=True)._compare(ref, inf) == 33.33333333333333
    assert TopKComparator(k=3, ordered=False)._compare(ref, inf) == 33.33333333333333


def test_topk_k_larger_than_tensor() -> None:
    ref = np.array([1.0, 2.0], dtype=np.float32)
    assert TopKComparator(k=5).compare([ref], [ref.copy()])[0] == 0.0


def test_topk_ties_are_deterministic() -> None:
    ref = np.zeros(32, dtype=np.float32)
    inf = np.zeros(32, dtype=np.float32)
    inf[0] = 1.0
    values = {TopKComparator(k=1).compare([ref], [inf])[0] for _ in range(5)}
    assert values == {100.0}
    assert TopKComparator(k=1).compare([ref], [ref.copy()])[0] == 0.0
