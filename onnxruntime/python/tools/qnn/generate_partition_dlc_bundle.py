# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Fill a per-partition DLC debug bundle with inputs and CPU-EP goldens.

Usage:
    python generate_partition_dlc_bundle.py \\
        --bundle-dir /tmp/bundle \\
        --model model.onnx \\
        --inputs name1=path1.raw [name2=path2.raw ...]

Reads <bundle-dir>/manifest.json (produced at compile time by the QNN EP when
qnn.dump_partition_dlc_bundle=1), marks every boundary tensor as a graph output
of a copy of the ONNX model, runs once on the CPU EP with the user-supplied
inputs, then consolidates each partition into <bundle-dir>/<partition>/ holding
its <partition>.dlc, inputs/<name>.raw and goldens/<name>.raw. Files are raw
little-endian tensor data (no header), which is the format qnn-net-run consumes
via --input_list.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import onnx

import onnxruntime as ort
from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed

# ORT-core QDQ propagation renames an edge <base>_pre_q / _q_to_dq / _dq_to_q, with an
# optional _token_<N> uniquifier on collision (qdq_propagation.cc, graph.cc).
_QDQ_RENAME = re.compile(r"(_pre_q|_q_to_dq|_dq_to_q)(_token_\d+)?$")

ORT_TYPE_TO_NUMPY = {
    "float": np.float32,
    "double": np.float64,
    "float16": np.float16,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "bool": np.bool_,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle-dir", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--inputs", required=True, nargs="+", help="name=path.raw pairs; one per model input")
    p.add_argument(
        "--free-dim",
        nargs="*",
        default=[],
        metavar="NAME=VALUE",
        help="pin a dynamic dim_param (e.g. max_seq_len=1) before running",
    )
    return p.parse_args()


def collect_boundary_tensors(manifest):
    names = set()
    for p in manifest["partitions"]:
        for t in p["inputs"]:
            names.add(t["name"])
        for t in p["outputs"]:
            names.add(t["name"])
    return names


def add_boundary_outputs(model_path: Path, boundary_names, free_dims=None):
    # Load structure-only and save beside the source so external-data refs resolve; inlining a
    # >2GB model would hit protobuf's cap.
    model = onnx.load(str(model_path), load_external_data=False)
    for name, value in (free_dims or {}).items():
        make_dim_param_fixed(model.graph, name, value)
    existing_outputs = {o.name for o in model.graph.output}
    existing_value_info = {vi.name: vi for vi in model.graph.value_info}
    existing_inputs = {i.name for i in model.graph.input}
    node_outputs = {o for n in model.graph.node for o in n.output}
    sourceable = node_outputs | existing_value_info.keys() | existing_inputs

    # Recover <base> from QDQ-propagation renames; accept only if it's a real ONNX tensor.
    def onnx_name(name):
        if name in existing_outputs or name in sourceable:
            return name
        stripped = _QDQ_RENAME.sub("", name)
        return stripped if (stripped != name and stripped in sourceable) else None

    alias, skipped = {}, []
    for name in boundary_names:
        src = onnx_name(name)
        if src is None:
            skipped.append(name)
            continue
        alias[name] = src
        if src not in existing_outputs and src not in existing_inputs:
            model.graph.output.append(existing_value_info.get(src) or onnx.helper.make_empty_tensor_value_info(src))
    if skipped:
        print(f"skipped {len(skipped)} boundary tensors with no ONNX source: e.g. {skipped[0]}", file=sys.stderr)
    tmp_path = model_path.parent / f"_boundary_{model_path.stem}.onnx"
    onnx.save(model, str(tmp_path))
    return tmp_path, alias


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def main():
    args = parse_args()
    bundle_dir = args.bundle_dir
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        sys.exit(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    user_inputs = {}
    for spec in args.inputs:
        if "=" not in spec:
            sys.exit(f"--inputs expects name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        user_inputs[name] = Path(path)

    boundary_names = collect_boundary_tensors(manifest)
    free_dims = {}
    for spec in args.free_dim:
        name, _, value = spec.partition("=")
        free_dims[name] = int(value)
    modified_model, alias = add_boundary_outputs(args.model, boundary_names, free_dims)
    sess = ort.InferenceSession(str(modified_model), providers=["CPUExecutionProvider"])

    feeds = {}
    for inp in sess.get_inputs():
        if inp.name not in user_inputs:
            sys.exit(f"missing --inputs entry for model input {inp.name!r}")
        np_dtype = ORT_TYPE_TO_NUMPY.get(inp.type.replace("tensor(", "").replace(")", ""))
        if np_dtype is None:
            sys.exit(f"unsupported input dtype {inp.type!r} for {inp.name!r}")
        arr = np.fromfile(user_inputs[inp.name], dtype=np_dtype)
        if any(not (isinstance(d, int) and d > 0) for d in inp.shape):
            sys.exit(
                f"input {inp.name!r} has dynamic shape {inp.shape}; "
                f"freeze the model (e.g. via onnxruntime.tools.make_dynamic_shape_fixed) before running this helper."
            )
        feeds[inp.name] = arr.reshape(inp.shape)

    output_names = [o.name for o in sess.get_outputs()]
    outputs = sess.run(output_names, feeds)
    name_to_value = dict(zip(output_names, outputs, strict=False))
    for name, arr in feeds.items():
        name_to_value.setdefault(name, arr)

    quantized_dtypes = {"int8", "uint8", "int16", "uint16", "int4", "uint4", "int2", "uint2"}
    quantized_mismatch = False
    for p in manifest["partitions"]:
        pdir = bundle_dir / p["name"]
        (pdir / "inputs").mkdir(parents=True, exist_ok=True)
        (pdir / "goldens").mkdir(parents=True, exist_ok=True)
        # Consolidate the compile-time DLC into this partition's folder (idempotent on rerun).
        dlc_src = bundle_dir / p["dlc_path"]
        dlc_dst = pdir / Path(p["dlc_path"]).name
        if dlc_src.resolve() != dlc_dst.resolve() and dlc_src.exists():
            dlc_src.replace(dlc_dst)
        p["dlc_path"] = f"{p['name']}/{dlc_dst.name}"
        for kind, key in [("inputs", "inputs"), ("outputs", "goldens")]:
            for t in p[kind]:
                v = name_to_value.get(alias.get(t["name"], t["name"]))
                if v is None:
                    print(f"warn: boundary {kind[:-1]} {t['name']!r} not produced", file=sys.stderr)
                    continue
                fname = sanitize_filename(t["name"]) + ".raw"
                t["raw_file"] = f"{p['name']}/{key}/{fname}"
                np.ascontiguousarray(v).tofile(pdir / key / fname)
                if t.get("dtype", "").removesuffix("_t") in quantized_dtypes and np.issubdtype(v.dtype, np.floating):
                    quantized_mismatch = True

    # Drop the now-empty partitions/ dir the compiler left behind.
    partitions_dir = bundle_dir / "partitions"
    if partitions_dir.is_dir() and not any(partitions_dir.iterdir()):
        partitions_dir.rmdir()

    modified_model.unlink(missing_ok=True)
    manifest["goldens_source"] = "cpu"
    if quantized_mismatch:
        manifest["goldens_domain_mismatch"] = True
        print(
            "warn: manifest declares quantized boundary tensors but goldens were captured "
            "float-domain from CPU EP; quantize them before comparing against on-device DLC I/O.",
            file=sys.stderr,
        )
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Bundle ready: {bundle_dir}")
    print(f"To share: tar czf {bundle_dir.name}.tar.gz -C {bundle_dir.parent or '.'} {bundle_dir.name}")


if __name__ == "__main__":
    main()
