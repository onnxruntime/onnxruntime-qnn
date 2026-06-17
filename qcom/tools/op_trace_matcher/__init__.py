# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""op_trace_matcher: source ONNX -> optimized ONNX -> QNN op trace tooling.

Marking this directory as a Python package allows package-style imports
(`from qcom.tools.op_trace_matcher.source_to_optimized_matcher import ...`).

Public APIs are intentionally exposed only on the submodules — each submodule
defines its own ``__all__``. Import them directly:

    from qcom.tools.op_trace_matcher.source_to_optimized_matcher import (
        Matcher, build_output, join_qnn_trace,
    )
    from qcom.tools.op_trace_matcher.enrich_profiling_csv import (
        build_lookups, enrich,
    )

The package itself does NOT re-export those names; this keeps the submodule
boundary explicit and lets each tool be loaded in isolation (e.g. Mode A of
``enrich_profiling_csv`` does not need ``onnx`` and only imports the
matcher's stdlib-only schema constants on demand).
"""
