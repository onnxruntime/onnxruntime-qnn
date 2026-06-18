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
inputs, then writes <bundle-dir>/runtime/<partition>/inputs/<name>.raw and
goldens/<name>.raw for every partition. Files are raw little-endian tensor
data (no header), which is the format qnn-net-run consumes via --input_list.
"""

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx

import onnxruntime as ort

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
    return p.parse_args()


def collect_boundary_tensors(manifest):
    names = set()
    for p in manifest["partitions"]:
        for t in p["inputs"]:
            names.add(t["name"])
        for t in p["outputs"]:
            names.add(t["name"])
    return names


def add_boundary_outputs(model_path: Path, boundary_names) -> Path:
    model = onnx.load(str(model_path))
    existing_outputs = {o.name for o in model.graph.output}
    existing_value_info = {vi.name: vi for vi in model.graph.value_info}
    existing_inputs = {i.name for i in model.graph.input}
    for name in boundary_names:
        if name in existing_outputs or name in existing_inputs:
            continue
        if name in existing_value_info:
            model.graph.output.append(existing_value_info[name])
        else:
            vi = onnx.helper.make_empty_tensor_value_info(name)
            model.graph.output.append(vi)
    tmp_path = Path(tempfile.mkstemp(suffix=".onnx")[1])
    onnx.save(model, str(tmp_path))
    return tmp_path


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
    modified_model = add_boundary_outputs(args.model, boundary_names)
    sess = ort.InferenceSession(str(modified_model), providers=["CPUExecutionProvider"])

    feeds = {}
    for inp in sess.get_inputs():
        if inp.name not in user_inputs:
            sys.exit(f"missing --inputs entry for model input {inp.name!r}")
        np_dtype = ORT_TYPE_TO_NUMPY.get(inp.type.replace("tensor(", "").replace(")", ""))
        if np_dtype is None:
            sys.exit(f"unsupported input dtype {inp.type!r} for {inp.name!r}")
        arr = np.fromfile(user_inputs[inp.name], dtype=np_dtype)
        dynamic_dims = [d for d in inp.shape if not (isinstance(d, int) and d > 0)]
        if dynamic_dims:
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

    runtime_dir = bundle_dir / "runtime"
    quantized_dtypes = {"int8", "uint8", "int16", "uint16", "int4", "uint4", "int2", "uint2"}
    quantized_mismatch = False
    for p in manifest["partitions"]:
        pdir = runtime_dir / p["name"]
        (pdir / "inputs").mkdir(parents=True, exist_ok=True)
        (pdir / "goldens").mkdir(parents=True, exist_ok=True)
        for kind, key in [("inputs", "inputs"), ("outputs", "goldens")]:
            for t in p[kind]:
                v = name_to_value.get(t["name"])
                if v is None:
                    print(f"warn: boundary {kind[:-1]} {t['name']!r} not produced", file=sys.stderr)
                    continue
                fname = sanitize_filename(t["name"]) + ".raw"
                t["raw_file"] = f"{key}/{fname}"
                np.ascontiguousarray(v).tofile(pdir / key / fname)
                if t.get("dtype", "").rstrip("_t") in quantized_dtypes and np.issubdtype(v.dtype, np.floating):
                    quantized_mismatch = True

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
    assert sanitize_filename("/encoder/layer.0/Add_output_0") == "_encoder_layer.0_Add_output_0"
    assert sanitize_filename("simple_name") == "simple_name"
    assert sanitize_filename("a:b\\c") == "a_b_c"
    main()
