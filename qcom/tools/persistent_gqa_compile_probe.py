# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Generate and compile a small GQA graph; no inference or KV initialization."""

import argparse
import json
import os
from pathlib import Path

import onnx
from onnx import TensorProto, helper

import onnxruntime as ort


def make_model(persistent, keep_kv_io=False):
    query = helper.make_tensor_value_info("query", TensorProto.FLOAT16, [1, 1, 512])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 1, 256])
    caches = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT16, [1, 4, 128, 32])
        for name in ("past_key", "past_value", "present_key", "present_value")
    ]
    omit_kv_io = persistent and not keep_kv_io
    inputs = [query] if omit_kv_io else [query, *caches[:2]]
    outputs = [output] if omit_kv_io else [output, *caches[2:]]
    node = helper.make_node(
        "GroupQueryAttention",
        [
            "query",
            "",
            "",
            "" if omit_kv_io else "past_key",
            "" if omit_kv_io else "past_value",
            "seqlens_k",
            "total_sequence_length",
        ],
        ["output", "present_key", "present_value"],
        name="gqa",
        domain="com.microsoft",
        num_heads=8,
        kv_num_heads=4,
        do_rotary=0,
        local_window_size=-1,
    )
    graph = helper.make_graph(
        [node],
        "gqa_compile_probe",
        inputs,
        outputs,
        initializer=[
            helper.make_tensor("seqlens_k", TensorProto.INT32, [1], [0]),
            helper.make_tensor("total_sequence_length", TensorProto.INT32, [1], [128]),
        ],
        value_info=caches[2:] if omit_kv_io else [],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("com.microsoft", 1)],
        ir_version=9,
    )
    onnx.checker.check_model(model)
    assert len(model.graph.input) == (1 if omit_kv_io else 3)
    assert len(model.graph.output) == (1 if omit_kv_io else 3)
    assert len(model.graph.node[0].output) == 3
    if omit_kv_io:
        assert list(model.graph.node[0].input)[3:5] == ["", ""]
        assert {v.name for v in model.graph.value_info} == {"present_key", "present_value"}
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep-library", type=Path)
    parser.add_argument("--backend", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("build") / "persistent_gqa_probe")
    parser.add_argument("--mode", choices=["baseline", "persistent", "default"], default="persistent")
    parser.add_argument("--htp-arch", default="73")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--cross-device-prepare", action="store_true")
    parser.add_argument("--remove-kv-io", action="store_true", help="Use the earlier omitted-KV compile diagnostic")
    args = parser.parse_args()
    if args.self_check:
        for persistent in (False, True):
            make_model(persistent)
        make_model(True, keep_kv_io=True)
        print("Model self-check passed: baseline, retained KV persistent, and omitted KV diagnostic")
        return
    if args.ep_library is None or args.backend is None:
        parser.error("--ep-library and --backend are required for compilation")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump_dir = args.output_dir / f"{args.mode}_graph"
    dump_dir.mkdir(parents=True, exist_ok=True)
    affinity_path = args.output_dir / "op_affinity.json"
    affinity_path.write_text(json.dumps({"op_type": {"GroupQueryAttention": "HTP"}}), encoding="utf-8")
    model_path = args.output_dir / f"{args.mode}.onnx"
    compiled_path = args.output_dir / f"{args.mode}_ctx.onnx"
    keep_kv_io = args.mode == "persistent" and not args.remove_kv_io
    onnx.save(make_model(args.mode != "baseline", keep_kv_io=keep_kv_io), model_path)
    dll_dirs = (
        [os.add_dll_directory(str(p.resolve())) for p in {args.ep_library.parent, args.backend.parent}]
        if os.name == "nt"
        else []
    )
    ort.register_execution_provider_library("QNNExecutionProvider", str(args.ep_library.resolve()))
    devices = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
    assert devices, "QNN EP registration produced no devices"
    options = ort.SessionOptions()
    options.log_severity_level = 0
    options.add_session_config_entry("ep.context_enable", "1")
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_provider_for_devices(
        devices,
        {
            "backend_path": str(args.backend.resolve()),
            "htp_arch": args.htp_arch,
            "htp_num_cores": "1",
            "op_affinity": str(affinity_path.resolve()),
            "enable_htp_prepare_only": "1",
            "enable_htp_cross_device_prepare": "1" if args.cross_device_prepare else "0",
            "enable_htp_persistent_buffer": "1" if args.mode == "persistent" else "0",
            "dump_json_qnn_graph": "1",
            "json_qnn_graph_dir": str(dump_dir.resolve()),
        },
    )
    compiler = ort.ModelCompiler(options, str(model_path), embed_compiled_data_into_model=False)
    compiler.compile_to_file(str(compiled_path))
    compiled = onnx.load(compiled_path, load_external_data=False)
    assert any(n.op_type == "EPContext" for n in compiled.graph.node), "No compiled QNN graph"
    has_kv_io = args.mode == "baseline" or keep_kv_io
    assert len(compiled.graph.input) == (3 if has_kv_io else 1)
    assert len(compiled.graph.output) == (3 if has_kv_io else 1)
    tensors = {
        name: tensor
        for path in dump_dir.glob("*.json")
        for name, tensor in json.loads(path.read_text(encoding="utf-8")).get("graph", {}).get("tensors", {}).items()
    }
    assert tensors, "No QNN tensor dump to verify graph construction"
    persistent_tensors = {name: tensor for name, tensor in tensors.items() if tensor["type"] == 14}
    assert len(persistent_tensors) == (2 if args.mode == "persistent" else 0)
    for tensor in persistent_tensors.values():
        assert tensor["dims"] == [1, 4, 128, 32]
        assert tensor["data_type"] == 0x0216  # QNN_DATATYPE_FLOAT_16
    result = {
        "mode": args.mode,
        "ort_version": ort.__version__,
        "ep_library": str(args.ep_library.resolve()),
        "backend": str(args.backend.resolve()),
        "compiled_model": str(compiled_path.resolve()),
        "compile": "PASS",
        "cross_device_prepare": args.cross_device_prepare,
        "persistent_tensors": list(persistent_tensors),
        "inference": "NOT_RUN",
        "kv_io_retained": has_kv_io,
    }
    (args.output_dir / f"{args.mode}_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    for dll_dir in dll_dirs:
        dll_dir.close()


if __name__ == "__main__":
    main()
