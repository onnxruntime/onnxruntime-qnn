# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
#
# Generic LoRA inference runner for QNN EP.
# Works with any EPContext ONNX model and any set of adapter .bin files.
#
# Usage:
#   # Run without any adapter (base model):
#   python test_lora.py --onnx model_ctx.onnx
#
#   # Run with a pre-written lora config file (<graph_name>;<bin_path>):
#   python test_lora.py --onnx model_ctx.onnx --lora-config ort_lora_configs/elementary.txt
#
#   # Sweep all lora config .txt files in a directory:
#   python test_lora.py --onnx model_ctx.onnx --sweep-configs ort_lora_configs/
#
#   # Run with a single adapter .bin (graph name auto-detected or via --graph-name):
#   python test_lora.py --onnx model_ctx.onnx --adapter path/to/adapter.bin
#
#   # Sweep all .bin files found in a directory:
#   python test_lora.py --onnx model_ctx.onnx --sweep-adapters path/to/adapters/
#
#   # Feed real inputs so adapters differ. --input-dir auto-detects either layout:
#   #   (a) ONNX test-data:  a dir of input_*.pb  (or a model dir with test_data_set_*/)
#   #   (b) QNN net-run:     a dir with input_list.txt + .raw files
#   python test_lora.py --onnx model_ctx.onnx --sweep-configs cfgs/ --input-dir test_data_set_0/
#
#   # Hot Switch mode (no binary section; select adapter via lora_alpha one-hot sweep):
#   python test_lora.py --onnx model_ctx.onnx --hot-switch
#
#   # Grouped LoRA cross-group switching (ordered encodings+weights sections per adapter):
#   python test_lora.py --onnx model_ctx.onnx --grouped-switch ort_lora_configs_grouped/
#
# --lora-alpha usage:
#   LoRAv3 models built with a vector lora_alpha expose it as a model input. It scales
#   adapter contribution: 0 = no effect (invisible), 1 = full effect. Use --lora-alpha:
#
#   (1) Binary-section modes (--lora-config / --adapter / --sweep-*):
#       The adapter weights are loaded via the binary section. lora_alpha determines
#       how strongly the adapter affects the output. Without --input-dir, use
#       --lora-alpha to ensure the adapter is non-zero:
#       python test_lora.py --onnx model_ctx.onnx --lora-config cfg.txt --lora-alpha 1.0 0.0
#       (--input-dir may supply a lora_alpha default; --lora-alpha overrides it.)
#
#   (2) Hot Switch mode:
#       lora_alpha IS the adapter selector — the binary section is not used. One position
#       is set to 1.0 per adapter slot. --hot-switch sweeps this automatically; use
#       --lora-alpha only to test a specific mix, e.g. --hot-switch --lora-alpha 0.8 0.2
#       would be ignored (--hot-switch overrides it via its own sweep).
#
#   (3) Models without a lora_alpha input (LoRAv2 or LoRAv3 built without the flag):
#       --lora-alpha is ignored with a warning.
#
#   Dtype note: the QNN graph natively stores lora_alpha as uint16 (quantized,
#   with a baked-in scale — typically covering the range [0, 1]). The wrapper
#   ONNX determines what dtype your app must feed:
#     - *_qnn_ctx_fp32_io.onnx (recommended): session input is float32; feed
#       real values 0.0-1.0 and the Q node inside quantizes automatically.
#     - *_qnn_ctx.onnx (quantized IO): session input is uint16 directly; feed
#       pre-quantized codes 0-65535. The script casts your --lora-alpha values
#       literally to the model's dtype and warns; --hot-switch uses iinfo.max
#       as a best-effort "on" code.
#
# Prerequisites:
#   pip install onnxruntime_qnn onnxruntime numpy onnx
#
# Lora config file format (one line):
#   <graph_name>;<absolute_or_relative_path_to_adapter.bin>
#
# The graph name is auto-detected from the EPContext node in the ONNX model when
# --graph-name is not provided (only needed for --adapter / --sweep-adapters modes).

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime_qnn as qnn_ep
from onnx import numpy_helper

import onnxruntime as ort

EP_REGISTRATION_NAME = "QNNExecutionProvider"

ONNX_DTYPE_MAP = {
    "float16": np.float16,
    "float": np.float32,
    "double": np.float64,
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


def ort_type_to_numpy(ort_type_str: str) -> np.dtype:
    """Convert ORT type string like 'tensor(float)' to numpy dtype.
    Raises for types with no plain numpy equivalent (e.g. bfloat16, complex*)
    so the mismatch surfaces here rather than as an obscure ORT type error.
    """
    inner = ort_type_str.split("(")[1].rstrip(")")
    if inner not in ONNX_DTYPE_MAP:
        raise ValueError(f"Unsupported ORT input dtype: {ort_type_str!r}. Supported: {sorted(ONNX_DTYPE_MAP)}")
    return ONNX_DTYPE_MAP[inner]


def detect_graph_name(onnx_path: str) -> str | None:
    """Extract graph name from the EPContext node's 'name' attribute in the ONNX model."""
    try:
        model = onnx.load(onnx_path)
        for node in model.graph.node:
            if node.op_type == "EPContext":
                return node.name if node.name else None
    except Exception:
        pass
    return None


def make_lora_config(graph_name: str, bin_path: str) -> str:
    """Write a temp lora config file in the format expected by ParseLoraConfig.
    Format: <graph_name>;<absolute_bin_path>
    Returns the path to the temp file (caller must delete it).
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, prefix="lora_cfg_") as tmp:
        tmp.write(f"{graph_name};{os.path.abspath(bin_path)}\n")
        return tmp.name


def parse_lora_config(config_path: str) -> tuple[str, str]:
    """Parse a user-provided lora config file.
    Expected format (one line): <graph_name>;<bin_path>
    Returns (graph_name, bin_path). The bin path is taken as written; if it is
    relative it is resolved against the config file's own directory.
    """
    with open(config_path) as f:
        line = f.readline().strip()
    assert ";" in line, (
        f"Invalid lora config format in {config_path}. Expected '<graph_name>;<bin_path>', got: {line!r}"
    )
    graph_name, bin_path = line.split(";", 1)
    graph_name = graph_name.strip()
    bin_path = bin_path.strip()
    if not os.path.isabs(bin_path):
        bin_path = os.path.join(os.path.dirname(os.path.abspath(config_path)), bin_path)
    return graph_name, bin_path


def build_zero_inputs(session: ort.InferenceSession) -> dict:
    """Build a zero-filled input feed from the session's declared shapes/dtypes."""
    feed = {}
    for inp in session.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        dtype = ort_type_to_numpy(inp.type)
        feed[inp.name] = np.zeros(shape, dtype=dtype)
    return feed


def apply_lora_alpha(feed: dict, lora_alpha: list | None) -> None:
    """Overwrite feed['lora_alpha'] with user-supplied values (CLI override).

    Called AFTER any --input-dir load so the CLI value wins over file-supplied
    alpha. No-op if the model has no lora_alpha input or lora_alpha is None.

    Dtype handling: the value is cast to the session's declared dtype for the
    lora_alpha input. For float dtypes (*_qnn_ctx_fp32_io.onnx wrapper) this is
    natural — the Q node inside quantizes 0.0-1.0 for you. For integer dtypes
    (*_qnn_ctx.onnx wrapper, e.g. uint16 with baked-in scale), the value is
    treated as a raw quantized code; a warning is printed because we cannot
    quantize user floats without the encoding scale (not exposed via ORT).
    """
    if lora_alpha is None:
        return
    if "lora_alpha" not in feed:
        print("  WARNING: --lora-alpha provided but model has no 'lora_alpha' input; ignoring.")
        return
    target = feed["lora_alpha"]
    expected_shape = target.shape
    expected_count = int(np.prod(expected_shape))
    assert len(lora_alpha) == expected_count, (
        f"--lora-alpha expects {expected_count} value(s) for shape {expected_shape}, got {len(lora_alpha)}."
    )
    if np.issubdtype(target.dtype, np.integer):
        print(
            f"  WARNING: model expects lora_alpha as {target.dtype} (quantized IO wrapper). "
            f"Values will be cast literally to {target.dtype}; use a *_fp32_io.onnx wrapper "
            f"for real-valued alpha with automatic quantization."
        )
    feed["lora_alpha"] = np.array(lora_alpha, dtype=target.dtype).reshape(expected_shape)
    print(f"  lora_alpha = {lora_alpha}  shape={expected_shape}  dtype={target.dtype}")


def parse_input_list(input_dir: str) -> dict:
    """Parse an 'input_list.txt' (QNN net-run format) into {input_name: abs_raw_path}.

    Format is one line of space-separated '<input_name>:=<relpath>' tokens, where
    relpath is relative to input_dir. Returns {} if no input_list.txt is present.
    """
    d = Path(input_dir)
    list_file = d / "input_list.txt"
    if not list_file.is_file():
        return {}
    mapping = {}
    for token in list_file.read_text().strip().split():
        if ":=" not in token:
            continue
        name, rel = token.split(":=", 1)
        mapping[name.strip()] = str((d / rel.strip()).resolve())
    return mapping


def _load_raw_inputs(session: ort.InferenceSession, feed: dict, input_dir: str) -> dict:
    """Overwrite zero-filled feed entries with real .raw tensors from input_dir.

    Each raw file is a flat binary buffer. An entry is loaded ONLY if its byte
    count exactly matches the session input's expected size; otherwise the
    zero-filled value is kept and a warning is printed. This avoids guessing
    shapes/dtypes -- the session's declared dtype/shape (already reflected in
    `feed`) is authoritative.
    """
    mapping = parse_input_list(input_dir)
    assert mapping, (
        f"--input-dir has no recognized layout: {input_dir} "
        f"contains no input_list.txt or it contains no valid ':=' tokens."
    )

    loaded, skipped = 0, 0
    for name, target in feed.items():
        raw = mapping.get(name)
        if raw is None or not os.path.isfile(raw):
            print(f"  WARNING: no raw input for '{name}'; keeping zeros.")
            skipped += 1
            continue
        actual_bytes = os.path.getsize(raw)
        if actual_bytes != target.nbytes:
            print(
                f"  WARNING: '{name}' size mismatch (raw {actual_bytes} B vs "
                f"expected {target.nbytes} B for {target.dtype}{target.shape}); keeping zeros."
            )
            skipped += 1
            continue
        feed[name] = np.fromfile(raw, dtype=target.dtype).reshape(target.shape)
        loaded += 1

    tail = f"; {skipped} kept as zeros." if skipped else "."
    print(f"  Loaded {loaded} real input(s) from {input_dir}" + tail)
    return feed


def _load_pb_inputs(session: ort.InferenceSession, feed: dict, data_dir: str) -> dict:
    """Overwrite feed entries from an ONNX test_data_set layout of input_*.pb files.

    Each input_<N>.pb is a serialized onnx.TensorProto (self-describing: dtype and
    shape are embedded). Mapping to session inputs:
      - by TensorProto.name when it matches a session input name;
      - otherwise positionally, input_<N>.pb -> the model's Nth input.
    data_dir may be a single test_data_set_* dir, or a model dir containing one or
    more test_data_set_* subdirs (the lowest-numbered is used).
    """
    d = Path(data_dir)
    if not any(d.glob("input_*.pb")):
        sets = sorted(d.glob("test_data_set_*"), key=lambda p: int(p.name.rsplit("_", 1)[1]))
        assert sets, f"No 'input_*.pb' or 'test_data_set_*' found in {d}"
        d = sets[0]
        print(f"  Using test data set: {d.name}")

    pb_files = sorted(d.glob("input_*.pb"), key=lambda p: int(p.stem.split("_")[1]))
    assert pb_files, f"No 'input_*.pb' files in {d}"

    input_names = [inp.name for inp in session.get_inputs()]
    loaded = 0
    for idx, pb in enumerate(pb_files):
        tensor = onnx.load_tensor(str(pb))
        arr = numpy_helper.to_array(tensor)
        if tensor.name in feed:
            key = tensor.name
        elif idx < len(input_names):
            key = input_names[idx]
        else:
            print(f"  WARNING: {pb.name} has no matching model input; skipping.")
            continue
        if arr.shape != feed[key].shape or arr.dtype != feed[key].dtype:
            print(
                f"  NOTE: {pb.name} -> '{key}' is {arr.dtype}{arr.shape} "
                f"(model expects {feed[key].dtype}{feed[key].shape}); using file's values."
            )
        feed[key] = arr
        loaded += 1

    print(f"  Loaded {loaded} .pb input(s) from {d}.")
    return feed


def load_real_inputs(session: ort.InferenceSession, feed: dict, input_dir: str) -> dict:
    """Load real inputs into `feed`, auto-detecting the directory layout:
    - ONNX test_data_set / input_*.pb  (self-describing TensorProto)
    - QNN net-run input_list.txt + .raw (byte-matched against declared shapes)
    """
    d = Path(input_dir)
    assert d.is_dir(), f"--input-dir not a directory: {input_dir}"
    if any(d.glob("input_*.pb")) or any(d.glob("test_data_set_*")):
        return _load_pb_inputs(session, feed, input_dir)
    return _load_raw_inputs(session, feed, input_dir)


def create_session(onnx_path: str, log_level: int = 3) -> ort.InferenceSession:
    ep_lib_path = qnn_ep.get_library_path()
    ort.register_execution_provider_library(EP_REGISTRATION_NAME, ep_lib_path)

    ep_name = qnn_ep.get_ep_names()[0]
    all_devices = ort.get_ep_devices()
    selected = [d for d in all_devices if d.ep_name == ep_name]
    assert selected, f"No EP devices found for '{ep_name}'"

    so = ort.SessionOptions()
    so.log_severity_level = log_level
    so.add_provider_for_devices(selected, {})

    session = ort.InferenceSession(onnx_path, sess_options=so)
    # Disable ORT's Python-layer EP fallback. On any QNN EP failure (e.g. the
    # expected step-1 error in grouped-switch) InferenceSession.run() would
    # otherwise catch the exception, call set_providers([CPU]), and retry --
    # which strips QNN, leaves only CPU, and destroys the session because CPU
    # cannot run EPContext. We want QNN failures to propagate cleanly instead.
    #
    # Note: we do NOT set the C++ "session.disable_cpu_ep_fallback" option here
    # because the *_qnn_ctx_fp32_io.onnx wrapper has QuantizeLinear /
    # DequantizeLinear nodes around the EPContext node that need CPU EP as a
    # partition fallback at session init. That option would block CPU from
    # being available at all, causing session init to fail for fp32_io models.
    try:
        session.disable_fallback()  # public method on recent ORT versions
    except AttributeError:
        session._enable_fallback = False  # underlying attribute
    return session


def run_inference(
    session: ort.InferenceSession,
    feed: dict,
    graph_name: str | None = None,
    adapter_bin: str | None = None,
    lora_config_path: str | None = None,
    label: str | None = None,
) -> list:
    """Run one inference pass with a pre-built input feed.

    The active adapter (if any) is applied via the qnn.lora_config run-option,
    specified in one of two ways:
      - lora_config_path: an existing config file (<graph>;<bin>)
      - graph_name + adapter_bin: a temp config is generated for this run
    Omit both to run with no binary section (base model, or Hot Switch where the
    adapter is selected by the lora_alpha values already in `feed`).
    `label` overrides the banner line. Returns the list of output arrays.
    """
    run_opts = ort.RunOptions()

    cfg = lora_config_path
    tmp_cfg = None
    if cfg is None and adapter_bin is not None:
        assert graph_name, "graph_name required when using adapter_bin"
        cfg = tmp_cfg = make_lora_config(graph_name, adapter_bin)

    if cfg is not None:
        run_opts.add_run_config_entry("qnn.lora_config", os.path.abspath(cfg))
        print(f"\n[{label or 'Adapter'}] {Path(adapter_bin or cfg).name}")
        print(f"  cfg = {cfg}" + ("  (temp)" if tmp_cfg else ""))
    else:
        print(f"\n[{label or 'No adapter'}]  Running base model")

    try:
        outputs = session.run(None, feed, run_options=run_opts)
    finally:
        if tmp_cfg and os.path.exists(tmp_cfg):
            os.unlink(tmp_cfg)

    print(f"  outputs: {len(outputs)} tensor(s)")
    for i, out in enumerate(outputs):
        name = session.get_outputs()[i].name
        print(f"    [{i}] {name}: shape={out.shape} dtype={out.dtype} mean={out.mean():.4f} max={out.max():.4f}")

    return outputs


def run_adapters(session: ort.InferenceSession, feed: dict, items: list, is_sweep: bool) -> None:
    """Run a list of adapter items; each item is (bin_to_verify, run_inference_kwargs).

    In sweep mode a baseline no-adapter run precedes the list and items whose bin
    is missing are skipped; otherwise a missing bin is a hard error.
    """
    if is_sweep:
        run_inference(session, feed)  # baseline: no adapter
    for verify_bin, kwargs in items:
        if verify_bin and not Path(verify_bin).exists():
            if is_sweep:
                print(f"\n[SKIP] adapter bin not found: {verify_bin}")
                continue
            raise AssertionError(f"Adapter bin not found: {verify_bin}")
        run_inference(session, feed, **kwargs)


def grouped_switch(
    session: ort.InferenceSession,
    feed: dict,
    encodings_cfg: str,
    weights_cfg: str,
    adapter_label: str,
) -> list:
    """Grouped-LoRA cross-group switch: apply the encodings section then the
    weights section, in that order.

    QNN's grouped mode requires two ordered binary sections per adapter
    (encodings before weights). ORT's qnn.lora_config applies exactly ONE
    section per session.run(), so a cross-group switch takes TWO runs:
      step 1: qnn.lora_config = <adapter>_encodings.txt  (encodings applied
              via contextApplyBinarySection in OnRunStart; the immediate
              graph.execute of this run typically fails with QNN error 6000
              because the graph state is inconsistent — new encodings but
              adapter weights not yet updated. The section WAS applied.)
      step 2: qnn.lora_config = <adapter>_weights.txt    (weights applied;
              graph state is now consistent — this run's output is the real
              result.)
    Returns the outputs of step 2. CPU-EP fallback is disabled at the session
    level (see create_session), so step 1's expected error propagates cleanly
    and we swallow it to continue to step 2.
    """
    print(f"\n[Grouped switch: {adapter_label}]  ordered two-section apply")
    print("  step 1/2: applying encodings section (adapter not yet active)")
    try:
        run_inference(session, feed, lora_config_path=encodings_cfg, label=f"{adapter_label} (encodings section)")
    except Exception as e:
        # Expected: contextApplyBinarySection succeeded before graph.execute
        # raised. The section is applied; proceed to step 2.
        print(
            f"  NOTE: encodings-only graph.execute raised "
            f"{type(e).__name__} — section applied before error; continuing to step 2."
        )

    print("  step 2/2: applying weights section (adapter now active)")
    return run_inference(session, feed, lora_config_path=weights_cfg, label=f"{adapter_label} (weights section)")


def grouped_switch_sweep(session: ort.InferenceSession, feed: dict, grouped_dir: str) -> None:
    """Sweep every adapter in a grouped-configs directory.

    Pairs each '<adapter>_encodings.txt' with its '<adapter>_weights.txt' and
    performs the ordered two-section apply per adapter (see grouped_switch).
    """
    d = Path(grouped_dir)
    assert d.is_dir(), f"--grouped-switch not a directory: {d}"
    enc_cfgs = sorted(d.glob("*_encodings.txt"))
    assert enc_cfgs, f"No '*_encodings.txt' configs found in {d}"

    print(f"\nGrouped cross-group switching: {len(enc_cfgs)} adapter(s) from {d}")
    run_inference(session, feed)  # baseline: no adapter
    for enc in enc_cfgs:
        adapter = enc.name[: -len("_encodings.txt")]
        weights = d / f"{adapter}_weights.txt"
        if not weights.exists():
            print(f"\n[SKIP] {adapter}: no matching weights config ({weights.name})")
            continue
        missing = [
            b for b in (parse_lora_config(str(enc))[1], parse_lora_config(str(weights))[1]) if not Path(b).exists()
        ]
        if missing:
            print(f"\n[SKIP] {adapter}: missing bin(s): {missing}")
            continue
        grouped_switch(session, feed, str(enc), str(weights), adapter)


def hot_switch_sweep(session: ort.InferenceSession, feed: dict) -> None:
    """Hot Switch mode: no binary section is applied. All adapters are already
    resident in the concurrency; the active adapter is selected purely by the
    lora_alpha vector at execution time. Sweeps a one-hot vector across each
    position of the model's lora_alpha input.

    Dtype handling matches the session's declared lora_alpha dtype:
      - float (*_qnn_ctx_fp32_io.onnx): "on" = 1.0, "off" = 0.0 (Q node quantizes).
      - integer (*_qnn_ctx.onnx quantized IO): "on" = iinfo(dtype).max, "off" = 0.
        This is a best-effort proxy for "fully active" without access to the
        real encoding scale; for exact control use *_fp32_io.onnx.
    """
    assert "lora_alpha" in feed, (
        "Hot Switch requires a 'lora_alpha' model input, but none was found. "
        "This model was not built for hot switching."
    )
    target = feed["lora_alpha"]
    alpha_shape = target.shape
    n = int(np.prod(alpha_shape))
    if np.issubdtype(target.dtype, np.integer):
        on_val = np.iinfo(target.dtype).max
        print(
            f"  NOTE: lora_alpha is {target.dtype} (quantized IO wrapper); "
            f"using iinfo({target.dtype}).max={on_val} as the 'on' code. "
            f"For exact real-valued alpha, use a *_fp32_io.onnx wrapper."
        )
    else:
        on_val = 1
    print(f"\nHot Switch sweep: lora_alpha shape={alpha_shape} dtype={target.dtype} → {n} adapter position(s)")

    for i in range(n):
        one_hot = np.zeros(n, dtype=target.dtype)
        one_hot[i] = on_val
        feed["lora_alpha"] = one_hot.reshape(alpha_shape)
        run_inference(session, feed, label=f"Hot Switch: adapter[{i}] alpha={one_hot.tolist()}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser (adapter modes + all flags)."""
    parser = argparse.ArgumentParser(description="Generic LoRA inference runner for QNN EP (Windows ARM64 / Linux x86)")
    parser.add_argument(
        "--onnx",
        required=True,
        help="Path to EPContext wrapper ONNX model (*_qnn_ctx.onnx or *_fp32_io.onnx)",
    )

    # --- Adapter input modes (mutually exclusive) ---
    adapter_group = parser.add_mutually_exclusive_group()
    adapter_group.add_argument(
        "--lora-config",
        help="Path to a pre-written lora config file (format: <graph_name>;<bin_path>).",
    )
    adapter_group.add_argument(
        "--sweep-configs",
        help="Directory of lora config .txt files — run all of them in sequence.",
    )
    adapter_group.add_argument(
        "--adapter",
        help="Path to a single adapter .bin file. Requires graph name (auto-detected or --graph-name).",
    )
    adapter_group.add_argument(
        "--sweep-adapters",
        help="Directory of adapter .bin files — run all of them in sequence. "
        "Requires graph name (auto-detected or --graph-name).",
    )
    adapter_group.add_argument(
        "--hot-switch",
        action="store_true",
        help="Hot Switch mode: no binary section applied. Sweeps a one-hot "
        "lora_alpha vector across each adapter position. For models built "
        "with all adapters resident in a single concurrency.",
    )
    adapter_group.add_argument(
        "--grouped-switch",
        metavar="DIR",
        help="Grouped-LoRA cross-group switching. Directory of paired "
        "'<adapter>_encodings.txt' / '<adapter>_weights.txt' configs "
        "(e.g. ort_lora_configs_grouped/). Each adapter is switched by "
        "applying its encodings section then its weights section across two "
        "session.run() calls.",
    )

    parser.add_argument(
        "--graph-name",
        help="QNN graph name inside the context binary. "
        "Auto-detected from the EPContext node if not provided. "
        "Only needed for --adapter / --sweep-adapters modes.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        nargs="+",
        default=None,
        metavar="VAL",
        help="LoRA alpha vector values (e.g. --lora-alpha 1.0 0.0). Only used if the model has a 'lora_alpha' input.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Set ORT log level to VERBOSE (default: WARNING)",
    )
    parser.add_argument(
        "--input-dir",
        metavar="DIR",
        help="Directory of real inputs, auto-detected: either ONNX test-data "
        "(input_*.pb, self-describing; or a model dir with test_data_set_*/), "
        "or QNN net-run (input_list.txt + .raw). Loaded once and reused across "
        "the sweep. Without this, inputs are zero-filled and all adapters "
        "produce identical outputs.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    onnx_path = os.path.abspath(args.onnx)
    assert Path(onnx_path).exists(), f"ONNX model not found: {onnx_path}"

    graph_name = args.graph_name
    if graph_name is None and (args.adapter or args.sweep_adapters):
        graph_name = detect_graph_name(onnx_path)
        if graph_name:
            print(f"Auto-detected graph name: '{graph_name}'")
        else:
            print("WARNING: Could not auto-detect graph name from ONNX. Use --graph-name to specify it explicitly.")

    log_level = 0 if args.verbose else 3
    print(f"Loading session: {onnx_path}")
    session = create_session(onnx_path, log_level)
    print("Session ready.")

    print("\nModel inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: shape={inp.shape} dtype={inp.type}")
    print("Model outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: shape={out.shape} dtype={out.type}")

    if args.hot_switch and args.lora_alpha is not None:
        print(
            "  WARNING: --lora-alpha is ignored under --hot-switch "
            "(the hot-switch sweep overwrites lora_alpha on every iteration)."
        )
    feed = build_zero_inputs(session)
    if args.input_dir:
        feed = load_real_inputs(session, feed, args.input_dir)
    apply_lora_alpha(feed, args.lora_alpha)

    try:
        if args.lora_config or args.sweep_configs:
            if args.sweep_configs:
                d = Path(args.sweep_configs)
                assert d.is_dir(), f"--sweep-configs not a directory: {d}"
                cfgs = sorted(d.glob("*.txt"))
                assert cfgs, f"No .txt config files found in {d}"
            else:
                cfgs = [Path(args.lora_config)]
                assert cfgs[0].exists(), f"Lora config not found: {cfgs[0]}"

            items = []
            for cfg in cfgs:
                _, parsed_bin = parse_lora_config(str(cfg))
                items.append((parsed_bin, {"lora_config_path": str(cfg), "label": cfg.stem}))
            run_adapters(session, feed, items, is_sweep=bool(args.sweep_configs))

        elif args.adapter or args.sweep_adapters:
            assert graph_name, (
                "graph name is required for --adapter / --sweep-adapters. "
                "Use --graph-name or ensure the ONNX EPContext node has a name."
            )
            if args.sweep_adapters:
                d = Path(args.sweep_adapters)
                assert d.is_dir(), f"--sweep-adapters not a directory: {d}"
                # Exclude the main context binary (*.serialized.bin) — not an adapter section.
                bins = sorted(b for b in d.glob("*.bin") if not b.name.endswith(".serialized.bin"))
                assert bins, f"No adapter .bin files found in {d}"
            else:
                bins = [Path(args.adapter)]
                assert bins[0].exists(), f"Adapter bin not found: {bins[0]}"

            items = [(str(b), {"graph_name": graph_name, "adapter_bin": str(b), "label": b.stem}) for b in bins]
            run_adapters(session, feed, items, is_sweep=bool(args.sweep_adapters))

        elif args.hot_switch:
            hot_switch_sweep(session, feed)

        elif args.grouped_switch:
            grouped_switch_sweep(session, feed, args.grouped_switch)

        else:
            run_inference(session, feed)

        print("\nDone.")
    finally:
        del session
        ort.unregister_execution_provider_library(EP_REGISTRATION_NAME)


if __name__ == "__main__":
    main()
