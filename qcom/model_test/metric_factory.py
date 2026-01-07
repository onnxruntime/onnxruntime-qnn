#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

from types import MappingProxyType
from typing import Any, ClassVar

from accuracy_metrics import AssertAllCloseMetric, CosineSimilarityMetric, Metric, MSEMetric, SNRMetric

# ---------- Default metric parameters ----------
DEFAULTS: dict[str, dict[str, Any]] = {
    "mse": {"threshold": 0.001},
    "assert_allclose": {"rtol": 1e-3, "atol": 1e-5},
    "cosine_similarity": {"threshold": 0.99},
    "snr": {"threshold": 20, "batch_size": 1},
}


# ---------- Factory ----------
class MetricFactory:
    """
    Builds Metric instances from config dicts of the form:
    {
        "output_1": {"metric_type": "mse", "metric_params": {"threshold": 0.02}},
        "output_2": {"metric_type": "cosine_similarity", "metric_params": {"threshold": 0.97}},
        "output_3": {"metric_type": "assert_allclose", "metric_params": {"rtol": 1e-2, "atol": 1e-5}},
        "output_4": {"metric_type": "cosine_similarity"},
    }
    If metric_params is not provided, default parameters will be used.
    In above case, 0.99 will be used as threshold of cosine similarity.
    """

    _REGISTRY: ClassVar[dict[str, type["Metric"]]] = MappingProxyType(
        {
            "mse": MSEMetric,
            "assert_allclose": AssertAllCloseMetric,
            "cosine_similarity": CosineSimilarityMetric,
            "snr": SNRMetric,
        }
    )

    @staticmethod
    def merge_with_default_params(metric_type: str, params: dict[str, Any]) -> dict[str, Any]:
        """
        Merge user-provided params with defaults for the given metric_type.
        - Validates metric_type.
        - Validates that provided keys are known.
        - Validates value types (float/int for numeric params).
        """

        # Check metric is supported.
        if metric_type not in DEFAULTS:
            raise ValueError(f"Unsupported metric_type '{metric_type}'. Supported: {list(DEFAULTS.keys())}")

        # Start with default params.
        merged = {**DEFAULTS[metric_type]}

        # Check user-provided params are known keys.
        unknown = set(params.keys()) - set(merged.keys())
        if unknown:
            raise ValueError(f"Unknown params for {metric_type}: {sorted(unknown)}")

        # Override defaults.
        for k, v in params.items():
            if not isinstance(v, (float, int)):
                raise ValueError(
                    f"Invalid metric parameter '{k}': expected a numeric value (float/int), got {type(v).__name__}"
                )
            merged[k] = v

        return merged

    @classmethod
    def from_dict(cls, cfg: dict) -> Metric:
        # Use assert_allclose if metric_type by default.
        metric_type = cfg.get("metric_type", "assert_allclose")

        raw_params = cfg.get("metric_params", {}) or {}
        params = cls.merge_with_default_params(metric_type, raw_params)

        metric_cls = cls._REGISTRY[metric_type]

        # Instantiate with params which have merged with default values
        return metric_cls(**params)


# ---------- Utility to build metrics per output ----------
def build_metrics_map(output_names, config: dict) -> dict[str, Metric]:
    """
    Given the YAML-decoded config dict (top-level outputs), return a mapping
    from output name -> Metric instance.
    """
    metrics: dict[str, Metric] = {}
    for output_name in output_names:
        if output_name in config:
            metrics[output_name] = MetricFactory.from_dict(config[output_name])
        else:
            metrics[output_name] = MetricFactory.from_dict(dict())
    return metrics
