# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Trace mapping, fusion boundaries, report enrichment and real EP generation."""

from __future__ import annotations

import json
import logging
import sys
from unittest.mock import patch

import numpy as np
import onnx
import pandas as pd
import pytest
from common import FP32_REFERENCE_QDQ_OPS_INDEX, QDQ_OPS_INDEX_DEFAULT, Algorithm
from inputs import InputProcessor
from onnx import helper
from op_trace import TRACE_COLUMNS, FusionGroup, OpTraceIndex
from ort_qnn_inference_engine import (
    OrtQnnInferenceEngine,
    OrtQnnInferenceEngineInputConfig,
    OrtQnnInferenceEngineOutputConfig,
)
from ort_qnn_snooper import FORCED_CPU_REASON, HOST_DEQUANTIZE_REASON, OrtQnnSnooper, OrtQnnSnooperInputConfig
from parser import OrtQnnSnooperParser


def source(name, kind="OP"):
    return {"name": name, "type": kind}


def mapping(name="qnn_conv", sources=None, originals=None, **kwargs):
    result = {"dst_name": name, "sources": sources or [source("conv1")], **kwargs}
    if originals is not None:
        result["original_sources"] = originals
    return result


def document(ops=(), tensors=(), unsupported=()):
    return {
        "schema_version": 2,
        "subgraph_traces": [{"graph_name": "graph", "op_mappings": list(ops), "tensor_mappings": list(tensors)}],
        "unsupported_nodes": list(unsupported),
    }


def graph(*nodes):
    return helper.make_model(helper.make_graph(list(nodes), "test", [], []))


# QNN HTP fuses reshape1 -> transpose -> reshape2 of fusion_chain() as Rank6ToRank5Fusion.
FUSED_NODES = ("reshape1", "r1_q", "r1_dq", "transpose", "t_q", "t_dq", "reshape2", "r2_q")
FUSION_INTERNAL = frozenset({"r1", "r1_q", "r1_dq", "t", "t_q", "t_dq", "r2"})
FUSION_INPUT = {"input": np.linspace(-1, 1, 16, dtype=np.float32).reshape(1, 4, 2, 2)}


def fusion_chain(make_model):
    """QDQ Reshape(6D) -> Transpose -> Reshape(4D) -> Relu; every Q/DQ node is named after its output."""

    def qdq(tensor, output=None):
        quantized, dequantized = f"{tensor}_q", f"{tensor}_dq"
        return [
            helper.make_node("QuantizeLinear", [tensor, "s", "z"], [quantized], name=quantized),
            helper.make_node("DequantizeLinear", [quantized, "s", "z"], [output or dequantized], name=dequantized),
        ]

    nodes = [
        *qdq("input"),
        helper.make_node("Reshape", ["input_dq", "shape6"], ["r1"], name="reshape1"),
        *qdq("r1"),
        helper.make_node("Transpose", ["r1_dq"], ["t"], name="transpose", perm=[0, 2, 1, 3, 4, 5]),
        *qdq("t"),
        helper.make_node("Reshape", ["t_dq", "shape4"], ["r2"], name="reshape2"),
        *qdq("r2"),
        helper.make_node("Relu", ["r2_dq"], ["relu"], name="relu"),
        *qdq("relu", "output"),
    ]
    return make_model(
        nodes,
        [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 4, 2, 2])],
        [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 4, 2, 2])],
        [
            helper.make_tensor("s", onnx.TensorProto.FLOAT, [], [0.01]),
            helper.make_tensor("z", onnx.TensorProto.UINT8, [], [128]),
            helper.make_tensor("shape6", onnx.TensorProto.INT64, [6], [1, 2, 2, 2, 2, 1]),
            helper.make_tensor("shape4", onnx.TensorProto.INT64, [4], [1, 4, 2, 2]),
        ],
    )


def fusion_trace():
    """Trace of fusion_chain(): the fusion's three QNN ops share its sources, including the leading DQ."""
    sources = [source("input_dq"), *(source(name) for name in FUSED_NODES)]
    ops = [
        mapping(name, sources, dst_qnn_op_type=kind, node_group_type="Rank6ToRank5Fusion")
        for name, kind in (("reshape1_2", "Reshape"), ("transpose_2", "Transpose"), ("reshape2_2", "Reshape"))
    ]
    relu = [source("r2_dq"), source("relu"), source("relu_q")]
    return document([*ops, mapping("relu_2", relu, dst_qnn_op_type="ElementWiseNeuron", node_group_type="OrtNodeUnit")])


def fusion_config(path, tmp_path, write_input_list, **kwargs):
    inputs = write_input_list(FUSION_INPUT, tmp_path / "inputs")
    return OrtQnnSnooperInputConfig(
        qdq_model=path,
        working_directory=tmp_path,
        enable_framework_op_trace=True,
        input_processor=InputProcessor(inputs, path),
        **kwargs,
    )


def test_arbitrary_quantizer_names_use_graph_edges(qdq_chain_model, fp32_chain_model):
    target = onnx.load(qdq_chain_model)
    for node in target.graph.node:
        for names in (node.input, node.output):
            for i, name in enumerate(names):
                if name == "conv1_out_dq":
                    names[i] = "completely_unrelated_activation"
    index = OpTraceIndex(document([mapping()]), target, onnx.load(fp32_chain_model))
    assert index.resolve("completely_unrelated_activation", {"conv1_out", "output"}, False) == ("conv1_out", False)


def test_merged_tensor_overrides_exact_name_and_preserves_namespace():
    doc = document(tensors=[mapping("x", [source("optimized", "TENSOR")], [source("authored", "TENSOR")])])
    index = OpTraceIndex(doc)
    assert index.resolve("x", {"x", "optimized", "authored"}, False) == ("authored", False)
    assert index.resolve("optimized", {"authored"}, False) == ("authored", False)
    assert OpTraceIndex(document([mapping("x", [source("authored")])])).resolve("x", {"authored"}, False) == (
        None,
        False,
    )


def test_merged_sources_fall_back_to_raw_when_reference_uses_optimized_names():
    index = OpTraceIndex(document(tensors=[mapping("x", [source("raw", "TENSOR")], [source("old", "TENSOR")])]))
    assert index.resolve("x", {"raw"}, True) == ("raw", False)


def test_sanitized_golden_name():
    index = OpTraceIndex(document(tensors=[mapping("x", [source("/authored:0", "TENSOR")])]))
    assert index.resolve("x", {"_authored_0"}, True) == ("_authored_0", False)


def test_fused_matmul_add_maps_only_boundary_output():
    target = graph(helper.make_node("Gemm", ["x", "w", "b"], ["fused_output"], name="optimized_gemm"))
    reference = graph(
        helper.make_node("MatMul", ["x", "w"], ["mm_out"], name="mm"),
        helper.make_node("Add", ["mm_out", "b"], ["add_out"], name="add"),
    )
    doc = document(
        [
            mapping(
                "fc",
                [source("optimized_gemm")],
                [source("mm"), source("mm_out", "TENSOR"), source("add")],
                dst_qnn_op_type="FullyConnected",
                node_group_type="MatMulAddFusion",
            )
        ]
    )
    index = OpTraceIndex(doc, target, reference)
    assert index.resolve("fused_output", {"mm_out", "add_out"}, False) == ("add_out", False)
    assert index.metadata("fused_output") == {
        "QNN_Op_Name": "fc",
        "QNN_Op_Type": "FullyConnected",
        "Fusion_Pattern": "MatMulAddFusion",
        "Original_ONNX_Nodes": "mm;add",
        "CPU_Fallback_Reason": "",
    }


def test_multi_output_op_preserves_output_slot():
    target = graph(helper.make_node("Split", ["x"], ["target0", "target1"], name="split"))
    reference = graph(helper.make_node("Split", ["x"], ["ref0", "ref1"], name="split"))
    index = OpTraceIndex(document([mapping(sources=[source("split")])]), target, reference)
    assert index.resolve("target1", {"ref0", "ref1"}, False) == ("ref1", False)


def test_fusion_with_multiple_boundaries_is_ambiguous_even_with_partial_reference():
    target = graph(helper.make_node("Custom", ["x"], ["out"], name="fused"))
    reference = graph(helper.make_node("Split", ["x"], ["a", "b"], name="split"))
    index = OpTraceIndex(document([mapping(sources=[source("fused")], originals=[source("split")])]), target, reference)
    assert index.resolve("out", {"a"}, False) == (None, True)


def test_duplicate_destinations_in_different_subgraphs_do_not_overwrite(logger):
    doc = document(tensors=[mapping("a_suffix", [source("a", "TENSOR")])])
    doc["subgraph_traces"].append(
        document(tensors=[mapping("a_suffix", [source("b", "TENSOR")])])["subgraph_traces"][0]
    )
    snooper = OrtQnnSnooper(logger)
    snooper._op_trace_index = OpTraceIndex(doc)
    assert snooper._get_onnx_tensor_mapping(["a", "b"], ["a_suffix"], False) == {}


def test_qdq_same_reference_keeps_quantized_and_float_outputs_distinct(qdq_chain_model):
    proto = onnx.load(qdq_chain_model)
    index = OpTraceIndex(document([mapping(sources=[source("conv1"), source("conv1_q")])]), proto, proto)
    names = {"conv1_out", "conv1_out_q", "conv1_out_dq"}
    for name in names:
        assert index.resolve(name, names, True) == (name, False)


def test_output_dq_is_not_attributed_to_its_downstream_consumer(qdq_chain_model):
    proto = onnx.load(qdq_chain_model)
    index = OpTraceIndex(
        document(
            [
                mapping("conv", [source("conv1")]),
                mapping("relu", [source("relu1"), source("conv1_dq")]),
            ]
        ),
        proto,
        proto,
    )
    assert index.metadata("conv1_out_dq")["QNN_Op_Name"] == "conv"


def test_real_ep_group_provenance_does_not_alias_input_tensors(qdq_chain_model):
    proto = onnx.load(qdq_chain_model)
    # The installed EP uses the complete group as tensor provenance, with
    # TENSOR entries that can be input tensors or optimizer node labels.
    sources = [
        source("w_dq"),
        source("w_dq", "TENSOR"),
        source("conv1_token_1"),
        source("conv1", "TENSOR"),
        source("conv1_q"),
    ]
    index = OpTraceIndex(
        document(
            [mapping("conv", sources), mapping("relu", [source("conv1_dq"), source("relu1")])],
            [mapping("conv1_q", sources)],
        ),
        proto,
        proto,
    )
    assert index.metadata("conv1_out_dq")["QNN_Op_Name"] == "conv"
    assert index.metadata("conv1_q")["QNN_Op_Name"] == "conv"
    assert index.resolve("conv1_q", {"w_dq"}, False) == (None, False)


def test_ep_inserted_op_group_falls_back_to_graph_producer(qdq_chain_model):
    proto = onnx.load(qdq_chain_model)
    # Real HTP traces map Q outputs to a layout Transpose that is absent from the ONNX graph.
    index = OpTraceIndex(
        document(
            [
                mapping("Transpose_token_1", [source("Transpose_token_1")], dst_qnn_op_type="Transpose"),
                mapping("conv", [source("conv1"), source("conv1_q")], dst_qnn_op_type="Conv2d"),
            ],
            [mapping("conv1_out_q", [source("Transpose_token_1")])],
        ),
        proto,
        proto,
    )
    metadata = index.metadata("conv1_out_q")
    assert (metadata["QNN_Op_Name"], metadata["QNN_Op_Type"]) == ("conv", "Conv2d")
    assert metadata["Original_ONNX_Nodes"] == "conv1;conv1_q"
    assert "Layer Name" not in metadata


def test_renamed_unsupported_node_reason_uses_ep_input_graph(tmp_path, qdq_chain_model):
    reason = "QNN EP is configured to not take DQ nodes that generate a graph output."
    trace = tmp_path / "qnn_op_trace.json"
    trace.write_text(json.dumps(document([mapping()], unsupported=[{"node_name": "DQ_token_1", "reason": reason}])))
    # The EP renames the rejected ONNX DQ; only the input graph dump links it to its output tensor.
    dump = {"graph": {"nodes": {"DQ_token_1": {"output_names": ["conv1_out_dq"], "type": "DequantizeLinear"}}}}
    (tmp_path / "graph.1_qnn_ep_input_graph.json").write_text(json.dumps(dump))
    proto = onnx.load(qdq_chain_model)
    assert OpTraceIndex.load(trace, proto).metadata("conv1_out_dq")["CPU_Fallback_Reason"] == reason
    (tmp_path / "graph.1_qnn_ep_input_graph.json").unlink()
    assert OpTraceIndex.load(trace, proto).metadata("conv1_out_dq")["CPU_Fallback_Reason"] == ""


def test_forced_cpu_layer_ignores_supplied_trace(logger):
    snooper = OrtQnnSnooper(logger)
    snooper._trace_requested = True
    snooper._supplied_op_trace = OpTraceIndex(document([mapping(dst_qnn_op_type="Conv2d")]))
    snooper._forced_cpu = True
    assert snooper._trace_metadata("conv1_out_dq") == {
        **dict.fromkeys(TRACE_COLUMNS, ""),
        "CPU_Fallback_Reason": FORCED_CPU_REASON,
    }


def test_host_dequantized_output_reports_reason(logger):
    snooper = OrtQnnSnooper(logger)
    snooper._trace_requested = True
    retry = "ORT retried this model on CPU after a QNN HTP execution failure."
    result = OrtQnnInferenceEngineOutputConfig(host_dequantized_outputs=("conv1_out_dq",), cpu_fallback_reason=retry)
    snooper._use_generated_trace(result, onnx.ModelProto())
    assert snooper._trace_metadata("conv1_out_dq")["CPU_Fallback_Reason"] == f"{HOST_DEQUANTIZE_REASON};{retry}"
    assert snooper._trace_metadata("conv1_out_q")["CPU_Fallback_Reason"] == retry
    snooper._use_generated_trace(OrtQnnInferenceEngineOutputConfig(), onnx.ModelProto())
    assert snooper._trace_metadata("conv1_out_dq")["CPU_Fallback_Reason"] == ""


def test_graph_dump_requires_enable_framework_op_trace(tmp_path, qdq_chain_model, qdq_input_sample, write_input_list):
    with pytest.raises(ValueError, match="requires enable_framework_op_trace"):
        OrtQnnSnooperInputConfig(
            qdq_model=qdq_chain_model,
            working_directory=tmp_path,
            dump_qnn_ep_input_graph=True,
            input_processor=InputProcessor(write_input_list(qdq_input_sample, tmp_path / "input"), qdq_chain_model),
        )


@pytest.mark.parametrize("algorithm", list(Algorithm))
def test_enable_framework_op_trace_uses_working_directory(
    tmp_path, qdq_chain_model, qdq_input_sample, write_input_list, logger, ort_cpu_target, algorithm
):
    config = OrtQnnSnooperInputConfig(
        qdq_model=qdq_chain_model,
        working_directory=tmp_path,
        algorithm=algorithm,
        enable_framework_op_trace=True,
        set_intermediate_layers=["conv1"],
        input_processor=InputProcessor(write_input_list(qdq_input_sample, tmp_path / "input"), qdq_chain_model),
    )
    with patch("ort_qnn_snooper.plot_graphs"):
        output = OrtQnnSnooper(logger, isolate_inference=False).run(config)
    # Oneshot compiles once in the run directory; layerwise compiles each layer in its subgraph directory.
    base = tmp_path if algorithm == Algorithm.ONESHOT else tmp_path / "subgraph_data" / "conv1"
    assert len(list((base / "op_trace").glob("run_*"))) == 1
    # The fusion probe compiles the unmodified model once, in its own directory.
    assert len(list((tmp_path / "fusion_probe" / "op_trace").glob("run_*"))) == 1
    assert set(TRACE_COLUMNS) <= set(pd.read_csv(output.csv_snooping_report).columns)


def test_multi_soc_flat_trace_and_cpu_fallback(qdq_chain_model):
    reason = "V73/soc_model=60,V81/soc_model=88: unsupported input Q"
    doc = document([mapping()], unsupported=[{"node_name": "input_q", "reason": reason}])
    doc["soc_traces"] = [{"compilation_target": {"soc_model": 60}}, {"compilation_target": {"soc_model": 88}}]
    index = OpTraceIndex(doc, onnx.load(qdq_chain_model))
    assert index.metadata("input_q")["CPU_Fallback_Reason"] == reason
    assert index.metadata("conv1_out_dq")["QNN_Op_Name"] == "qnn_conv"


@pytest.mark.parametrize(
    "doc",
    [
        [],
        {},
        {"schema_version": True},
        {"schema_version": 3},
        {"schema_version": 2, "subgraph_traces": {}},
        document(tensors=[{"dst_name": "x", "sources": ["bad"]}]),
        document(tensors=[{"dst_name": "x", "sources": [source("x", "INVALID")]}]),
    ],
)
def test_invalid_trace_warns_and_uses_legacy_mapping(tmp_path, logger, caplog, doc):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(doc))
    snooper = OrtQnnSnooper(logger)
    assert snooper._load_op_trace(path, graph()) is None
    assert "Cannot use framework op trace" in caplog.text
    assert snooper._get_onnx_tensor_mapping(["legacy"], ["legacy_quantized"], False) == {"legacy_quantized": "legacy"}


def test_bom_trace_and_empty_originals(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(document(tensors=[mapping("x", [source("y", "TENSOR")], [])])), encoding="utf-8-sig")
    assert OpTraceIndex.load(path).resolve("x", {"y"}, True) == ("y", False)


def test_trace_selection_follows_non_adjacent_qdq_branches(logger):
    proto = graph(
        helper.make_node("Split", ["x"], ["a", "b"], name="split"),
        helper.make_node("QuantizeLinear", ["b", "s", "z"], ["bq"], name="qb"),
        helper.make_node("Identity", ["other"], ["other2"]),
        helper.make_node("QuantizeLinear", ["a", "s", "z"], ["aq"], name="qa"),
        helper.make_node("DequantizeLinear", ["bq", "s", "z"], ["bdq"], name="db"),
        helper.make_node("DequantizeLinear", ["aq", "s", "z"], ["adq"], name="da"),
    )
    snooper = OrtQnnSnooper(logger)
    snooper._trace_requested = True
    assert set(snooper._get_valid_intermediates(["split"], proto, True)) == {"aq", "bq", "adq", "bdq"}


@pytest.mark.parametrize("algorithm", list(Algorithm))
def test_cpu_snooping_reports_trace_metadata_and_resets_between_runs(
    tmp_path,
    qdq_chain_model,
    qdq_input_sample,
    write_input_list,
    logger,
    ort_cpu_target,
    algorithm,
):
    trace = tmp_path / "merged.json"
    trace.write_text(
        json.dumps(
            document(
                [mapping(dst_qnn_op_type="Conv2d", node_group_type="DQQFusion")],
                unsupported=[{"node_name": "input_q", "reason": "input Q stays on CPU"}],
            )
        )
    )
    config = OrtQnnSnooperInputConfig(
        qdq_model=qdq_chain_model,
        working_directory=tmp_path,
        algorithm=algorithm,
        op_trace_file=trace,
        input_processor=InputProcessor(write_input_list(qdq_input_sample, tmp_path / "input"), qdq_chain_model),
    )
    snooper = OrtQnnSnooper(logger, isolate_inference=False)
    with patch("ort_qnn_snooper.plot_graphs"):
        output = snooper.run(config)
        csv = pd.read_csv(output.csv_snooping_report)
        rows = json.loads(output.json_snooping_report.read_text())["layers"]
        conv = next(row for row in rows if row["layer_output"] == "conv1_out_dq")
        assert conv["qnn_op_type"] == "Conv2d"
        assert conv["fusion_pattern"] == "DQQFusion"
        assert conv["original_onnx_nodes"] == "conv1"
        # conv1_out_q is captured too, so the engine dequantizes it on CPU instead of exposing both from QNN.
        assert conv["cpu_fallback_reason"] == HOST_DEQUANTIZE_REASON
        assert "qnn_op_type" not in conv["comparators"]
        assert set(TRACE_COLUMNS) <= set(csv.columns)
        config.op_trace_file = None
        output = snooper.run(config)
        assert not set(TRACE_COLUMNS) & set(pd.read_csv(output.csv_snooping_report).columns)


def test_parser_trace_options(tmp_path, qdq_chain_model, qdq_input_sample, write_input_list):
    inputs = write_input_list(qdq_input_sample, tmp_path)
    args = ["--qdq_model", str(qdq_chain_model), "--input_list", str(inputs)]
    parser = OrtQnnSnooperParser()
    assert not parser.parse(args).enable_framework_op_trace
    parsed = parser.parse([*args, "--enable_framework_op_trace", "--dump_qnn_ep_input_graph"])
    assert parsed.enable_framework_op_trace
    assert parsed.dump_qnn_ep_input_graph
    with pytest.raises(Exception, match="requires --enable_framework_op_trace"):
        parser.parse([*args, "--dump_qnn_ep_input_graph"])


@pytest.mark.parametrize("algorithm", list(Algorithm))
def test_snooper_rejects_cached_context(
    tmp_path,
    cached_context_model,
    qdq_chain_model,
    fp32_chain_model,
    qdq_input_sample,
    write_input_list,
    algorithm,
):
    (tmp_path / "qnn_op_trace.json").write_text(json.dumps(document([mapping("qnn_relu", [source("relu1")])])))
    with pytest.raises(ValueError, match="EPContext models are not supported"):
        OrtQnnSnooperInputConfig(
            qdq_model=cached_context_model,
            algorithm=algorithm,
            reference_model=fp32_chain_model if algorithm == Algorithm.ONESHOT else None,
            working_directory=tmp_path,
            input_processor=InputProcessor(write_input_list(qdq_input_sample, tmp_path / "input"), qdq_chain_model),
        )


def test_layerwise_retains_all_non_adjacent_qdq_output_branches(
    tmp_path,
    make_model,
    save_model,
    write_input_list,
    ort_cpu_target,
    logger,
):
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "s", "z"], ["xq"], name="xq"),
        helper.make_node("DequantizeLinear", ["xq", "s", "z"], ["xdq"], name="xdq"),
        helper.make_node("Split", ["xdq"], ["a", "b"], name="split", axis=1, num_outputs=2),
        helper.make_node("QuantizeLinear", ["b", "s", "z"], ["bq"], name="qb"),
        helper.make_node("QuantizeLinear", ["a", "s", "z"], ["aq"], name="qa"),
        helper.make_node("DequantizeLinear", ["bq", "s", "z"], ["bdq"], name="db"),
        helper.make_node("DequantizeLinear", ["aq", "s", "z"], ["adq"], name="da"),
        helper.make_node("Add", ["adq", "bdq"], ["sum"], name="add"),
    ]
    model = make_model(
        nodes,
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("sum", onnx.TensorProto.FLOAT, [1, 2])],
        [
            helper.make_tensor("s", onnx.TensorProto.FLOAT, [], [0.1]),
            helper.make_tensor("z", onnx.TensorProto.UINT8, [], [0]),
        ],
    )
    path = save_model(model, tmp_path / "split.onnx")
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps(document([mapping(sources=[source("split")])])))
    inputs = write_input_list({"x": np.array([[1, 2, 3, 4]], np.float32)}, tmp_path / "inputs")
    config = OrtQnnSnooperInputConfig(
        qdq_model=path,
        working_directory=tmp_path,
        algorithm=Algorithm.LAYERWISE,
        op_trace_file=trace,
        set_intermediate_layers=["split"],
        input_processor=InputProcessor(inputs, path),
    )
    with patch("ort_qnn_snooper.plot_graphs"):
        result = OrtQnnSnooper(logger, isolate_inference=False).run(config)
    rows = json.loads(result.json_snooping_report.read_text())["layers"]
    assert {row["layer_output"] for row in rows} == {"aq", "adq", "bq", "bdq"}
    assert all(row["layer_status"] == "SUCCESS" for row in rows)


def test_fusion_groups_hold_fused_nodes_and_internal_tensors(make_model):
    # The leading DQ is not a member: ORT gives the fusion its own copy of it.
    assert OpTraceIndex(fusion_trace(), fusion_chain(make_model)).fusion_groups() == [
        FusionGroup(FUSED_NODES, ("Rank6ToRank5Fusion",), ("reshape1_2", "transpose_2", "reshape2_2"), FUSION_INTERNAL)
    ]


def test_fusion_groups_merge_shared_nodes_and_drop_single_nodes(make_model):
    doc = document(
        [
            mapping("tail", [source("reshape2"), source("r2_q")], node_group_type="FusionC"),
            mapping("head", [source("reshape1"), source("r1_q")], node_group_type="FusionA"),
            mapping(
                "middle",
                [source("r1_q"), source("r1_dq/duplicated"), source("transpose"), source("Transpose_token_3")],
                node_group_type="FusionB",
            ),
            mapping("requant", [source("r2_dq"), source("relu")], node_group_type="DQQFusion"),
            mapping("unit", [source("t_q"), source("t_dq")], node_group_type="OrtNodeUnit"),
        ]
    )
    assert OpTraceIndex(doc, fusion_chain(make_model)).fusion_groups() == [
        FusionGroup(
            ("reshape1", "r1_q", "r1_dq", "transpose"),
            ("FusionA", "FusionB"),
            ("head", "middle"),
            frozenset({"r1", "r1_q", "r1_dq"}),
        ),
        FusionGroup(("reshape2", "r2_q"), ("FusionC",), ("tail",), frozenset({"r2"})),
    ]


def test_kept_fusions_select_fusion_outputs(make_model, logger):
    proto = fusion_chain(make_model)
    snooper = OrtQnnSnooper(logger)
    snooper._trace_requested = True
    snooper._fusion_groups = OpTraceIndex(fusion_trace(), proto).fusion_groups()

    def select(layers, **kwargs):
        return snooper._get_valid_intermediates(layers, proto, is_qdq_model=True, **kwargs)

    assert select(["transpose"], keep_fusions=True) == ["r2_q", "r2_dq"]
    assert select(["transpose"]) == ["t_q", "t_dq"]
    assert select(["transpose"], qdq_ops_index=FP32_REFERENCE_QDQ_OPS_INDEX, keep_fusions=True) == ["r2_dq"]
    assert select([], keep_fusions=True) == ["input_q", "input_dq", "r2_q", "r2_dq", "relu_q", "output"]
    # The fusions belong to the QDQ model; an FP32 reference keeps its own selection.
    assert snooper._get_valid_intermediates(["transpose"], proto, keep_fusions=True) == ["t"]


def test_layers_inside_fusions_warn(make_model, logger, caplog):
    proto = fusion_chain(make_model)
    snooper = OrtQnnSnooper(logger)
    snooper._fusion_groups = OpTraceIndex(fusion_trace(), proto).fusion_groups()
    # reshape2 produces the fusion outputs, so it is compared as requested.
    snooper._warn_fused_layers(["reshape2", "relu"], proto, QDQ_OPS_INDEX_DEFAULT)
    assert "inside QNN HTP fusion" not in caplog.text
    snooper._warn_fused_layers(["transpose", "reshape2", "Reshape"], proto, QDQ_OPS_INDEX_DEFAULT)
    assert (
        "reshape1, transpose are inside QNN HTP fusion Rank6ToRank5Fusion (reshape1, transpose, reshape2). "
        "Comparing the fusion outputs instead, so that QNN EP keeps the fusion: r2_q, r2_dq"
    ) in caplog.text


@pytest.mark.parametrize(
    ("layers", "compared"),
    [
        (None, {"input_q", "input_dq", "r2_q", "r2_dq", "relu_q", "output"}),
        (["transpose"], {"r2_q", "r2_dq", "output"}),
    ],
)
def test_oneshot_compares_fusion_outputs(
    tmp_path, make_model, save_model, write_input_list, ort_cpu_target, logger, caplog, layers, compared
):
    path = save_model(fusion_chain(make_model), tmp_path / "fusion.onnx")
    groups = OpTraceIndex(fusion_trace(), onnx.load(path)).fusion_groups()
    config = fusion_config(path, tmp_path, write_input_list, set_intermediate_layers=layers)
    with (
        patch.object(OrtQnnSnooper, "_probe_fusion_groups", return_value=groups),
        patch("ort_qnn_snooper.plot_graphs"),
    ):
        result = OrtQnnSnooper(logger, isolate_inference=False).run(config)
    rows = json.loads(result.json_snooping_report.read_text())["layers"]
    assert {row["layer_output"] for row in rows} == compared
    assert ("transpose is inside QNN HTP fusion Rank6ToRank5Fusion" in caplog.text) == bool(layers)


@pytest.mark.parametrize("layers", [None, ["transpose"]])
def test_layerwise_snoops_fusion_as_one_subgraph(
    tmp_path, make_model, save_model, write_input_list, ort_cpu_target, logger, layers
):
    path = save_model(fusion_chain(make_model), tmp_path / "fusion.onnx")
    groups = OpTraceIndex(fusion_trace(), onnx.load(path)).fusion_groups()
    config = fusion_config(
        path, tmp_path, write_input_list, algorithm=Algorithm.LAYERWISE, set_intermediate_layers=layers
    )
    with (
        patch.object(OrtQnnSnooper, "_probe_fusion_groups", return_value=groups),
        patch("ort_qnn_snooper.plot_graphs"),
    ):
        result = OrtQnnSnooper(logger, isolate_inference=False).run(config)
    rows = json.loads(result.json_snooping_report.read_text())["layers"]
    # The fusion's last layer names its rows; a filter on any of its layers selects it.
    expected = {("reshape2", "r2_q"), ("reshape2", "r2_dq")}
    if layers is None:
        expected |= {("relu", "relu_q"), ("relu", "output")}
    assert {(row["layer_name"], row["layer_output"]) for row in rows} == expected
    assert all(row["layer_status"] == "SUCCESS" for row in rows)
    subgraphs = tmp_path / "subgraph_data"
    assert {path.name for path in subgraphs.iterdir()} == {name for name, _ in expected}
    fused = onnx.load(subgraphs / "reshape2" / "extracted_model.onnx").graph
    # The fusion's float input is quantized again, so the fusion starts at a DQ as in the model.
    requantized = ["input_dq_requant_QuantizeLinear", "input_dq_requant_DequantizeLinear"]
    assert [node.name for node in fused.node] == [*requantized, *FUSED_NODES, "r2_dq"]
    assert [value.name for value in fused.input] == ["input_dq"]


def test_fusion_probe_compiles_unmodified_model(tmp_path, make_model, save_model, logger, caplog):
    caplog.set_level(logging.INFO)
    proto = fusion_chain(make_model)
    path = save_model(proto, tmp_path / "fusion.onnx")
    trace = tmp_path / "qnn_op_trace.json"
    trace.write_text(json.dumps(fusion_trace()))
    result = OrtQnnInferenceEngineOutputConfig(op_trace_file=trace)
    with patch.object(OrtQnnSnooper, "_run_inference", return_value=result) as run:
        groups = OrtQnnSnooper(logger, isolate_inference=False)._probe_fusion_groups(path, proto, tmp_path)
    assert groups == OpTraceIndex(fusion_trace(), proto).fusion_groups()
    config = run.call_args.args[0]
    assert config.compile_only
    assert config.enable_framework_op_trace
    assert not config.outputs_list
    assert config.working_directory == tmp_path / "fusion_probe"
    assert "QNN HTP fuses Rank6ToRank5Fusion: reshape1, transpose, reshape2" in caplog.text


def test_fusion_probe_without_trace_warns(tmp_path, make_model, save_model, ort_cpu_target, logger, caplog):
    proto = fusion_chain(make_model)
    path = save_model(proto, tmp_path / "fusion.onnx")
    assert OrtQnnSnooper(logger, isolate_inference=False)._probe_fusion_groups(path, proto, tmp_path) == []
    assert "QNN EP did not produce a framework op trace of its fusions" in caplog.text
    assert len(list((tmp_path / "fusion_probe" / "op_trace").glob("run_*"))) == 1


@pytest.mark.parametrize("algorithm", list(Algorithm))
def test_fusion_probe_needs_trace_flags(
    tmp_path, qdq_chain_model, qdq_input_sample, write_input_list, logger, ort_cpu_target, algorithm
):
    config = OrtQnnSnooperInputConfig(
        qdq_model=qdq_chain_model,
        working_directory=tmp_path,
        algorithm=algorithm,
        set_intermediate_layers=["conv1"],
        input_processor=InputProcessor(write_input_list(qdq_input_sample, tmp_path / "input"), qdq_chain_model),
    )
    with patch.object(OrtQnnSnooper, "_probe_fusion_groups") as probe, patch("ort_qnn_snooper.plot_graphs"):
        OrtQnnSnooper(logger, isolate_inference=False).run(config)
    probe.assert_not_called()
    assert not (tmp_path / "fusion_probe").exists()


@pytest.mark.qnn
@pytest.mark.parametrize("algorithm", list(Algorithm))
def test_real_qnn_snooper_consumes_generated_trace(
    tmp_path,
    qdq_chain_model,
    qdq_input_sample,
    write_input_list,
    logger,
    algorithm,
):
    from conftest import requires_qnn_ep  # noqa: PLC0415

    if not requires_qnn_ep():
        pytest.skip("QNN EP is unavailable")
    config = OrtQnnSnooperInputConfig(
        qdq_model=qdq_chain_model,
        algorithm=algorithm,
        working_directory=tmp_path,
        enable_framework_op_trace=True,
        set_intermediate_layers=["conv1"],
        input_processor=InputProcessor(write_input_list(qdq_input_sample, tmp_path / "input"), qdq_chain_model),
    )
    with patch("ort_qnn_snooper.plot_graphs"):
        output = OrtQnnSnooper(logger, name=algorithm).run(config)
    rows = json.loads(output.json_snooping_report.read_text())["layers"]
    conv = next(row for row in rows if row["layer_output"] == "conv1_out_dq")
    # A layerwise subgraph's input is quantized again and its weight is static, so QNN HTP compiles the Conv.
    assert "Conv2d" in conv["qnn_op_type"]
    assert "ORT retried" not in conv["cpu_fallback_reason"]
    assert conv["fusion_pattern"]
    assert "conv1_q" in conv["original_onnx_nodes"]
    assert "Relu" not in conv["qnn_op_type"]
    if algorithm == Algorithm.LAYERWISE:
        assert all(row["layer_status"] == "SUCCESS" for row in rows)
        assert list((tmp_path / "subgraph_data" / "conv1" / "op_trace").glob("run_*/qnn_op_trace.json"))
    else:
        assert list((tmp_path / "op_trace").glob("run_*/qnn_op_trace.json"))


@pytest.mark.parametrize("use_cpu", [pytest.param(False, marks=pytest.mark.qnn), True])
def test_real_qnn_trace_generation(tmp_path, qdq_chain_model, qdq_input_sample, logger, use_cpu, monkeypatch, caplog):
    from conftest import requires_qnn_ep  # noqa: PLC0415

    if use_cpu:
        monkeypatch.setitem(sys.modules, "onnxruntime_qnn", None)
    elif not requires_qnn_ep():
        pytest.skip("QNN EP is unavailable")
    config = OrtQnnInferenceEngineInputConfig(
        qdq_model=qdq_chain_model,
        input_sample=qdq_input_sample,
        working_directory=tmp_path,
        enable_framework_op_trace=True,
        dump_qnn_ep_input_graph=True,
        use_cpu=use_cpu,
    )
    result = OrtQnnInferenceEngine(logger).run(config)
    assert result.output_data and np.isfinite(result.output_data["output"]).all()
    if use_cpu:
        assert result.op_trace_file is None
        assert result.cpu_fallback_reason == ""
        assert not (tmp_path / "op_trace").exists()
        assert "did not produce a framework op trace" not in caplog.text
        return
    assert result.op_trace_file is not None, "Installed QNN EP did not emit the requested framework op trace"
    index = OpTraceIndex.load(result.op_trace_file, onnx.load(qdq_chain_model))
    assert index.ops
    assert list(result.op_trace_file.parent.glob("*_qnn_ep_input_graph.json"))
