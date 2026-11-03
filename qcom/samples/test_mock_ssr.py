# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
"""Generate a QNN context model, then test mock SSR recovery and repeated inference.

Basic usage from the repository root on Windows ARM64:
  python qcom/samples/test_mock_ssr.py

The default Abs model and its input are selected automatically. QnnMockSSR.dll is a test-only DLL built by the
Windows non-static QNN EP unit-test configuration; its post-build step copies it beside onnxruntime_provider_test.
If QnnMockSSR.dll, QnnHtp.dll, and QnnSystem.dll are not in the current directory or the plugin DLL directory, add
that directory with --dll-dir C:/path/to/onnxruntime_provider_test.

For an ordinary ONNX model, the script tests
- unrecoverable SSR during AOT generation,
- successful recovery after loading an external AOT context,
- unrecoverable SSR after loading an embedded AOT context,
- and unrecoverable SSR in JIT mode.
Supplying an existing external EPContext model runs only the cached-context recovery case.
"""

import argparse
import contextlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx

import onnxruntime as ort

try:
    import onnxruntime_qnn
except ModuleNotFoundError as error:
    if error.name != "onnxruntime_qnn":
        raise
    onnxruntime_qnn = None

DETECTED = "NPU crashed. SSR detected during QNN graph execute."
RECOVERED = "SSR recovery succeeded."
RECOVERY_NOT_SUPPORTED = "Recovery not supported (no context binary on disk"
REGISTRATION = "QNNExecutionProvider"
REQUIRED_QNN_DLLS = ("QnnMockSSR.dll", "QnnHtp.dll", "QnnSystem.dll")
QNN_CONTEXT_SOURCES = ("qnn", "qnnexecutionprovider", "qairtexport")
CONTEXT_GENERATION_WORKERS = ("generate", "embed_generate", "aot_generation_ssr")
UNRECOVERABLE_SSR_WORKERS = ("aot_generation_ssr", "embed_ssr", "jit_ssr")
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
DEFAULT_MODEL = Path("cmake/external/onnx/onnx/backend/test/data/node/test_abs/model.onnx").resolve()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Input ONNX model; defaults to Abs model.")
    parser.add_argument(
        "--inputs",
        type=Path,
        help="ONNX input_*.pb directory or single TensorProto .pb; required for custom models.",
    )
    parser.add_argument(
        "--ep-library", type=Path, help="QNN plugin DLL; defaults to onnxruntime_qnn.get_library_path()."
    )
    parser.add_argument(
        "--dll-dir", action="append", default=[], type=Path, help="Additional DLL directory; repeatable."
    )
    parser.add_argument("--ep-options", type=Path, help="JSON object of additional QNN provider options.")
    parser.add_argument("--runs", type=int, default=10, help="Runs on the same mock session, including recovery (>=2).")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--timeout", type=float, default=300, help="Timeout in seconds for each worker process.")
    parser.add_argument(
        "--allow-cpu-fallback", action="store_true", help="Allow CPU partitions; recovery logs remain required."
    )
    parser.add_argument("--output-dir", type=Path, help="New directory for logs, outputs and report.json.")
    parser.add_argument(
        "--worker",
        choices=[
            "generate",
            "baseline",
            "mock",
            "aot_generation_ssr",
            "embed_generate",
            "embed_ssr",
            "jit_ssr",
        ],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--context-model", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.runs < 2:
        parser.error("--runs must be >=2 to check subsequent inference on the recovered session")
    for name in ("rtol", "atol", "timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name == "timeout" and value == 0):
            parser.error(f"--{name} must be finite and {'positive' if name == 'timeout' else 'nonnegative'}")
    return args


def load_inputs(path, input_names):
    def read_tensor(file, expected_name):
        tensor = onnx.TensorProto()
        try:
            tensor.ParseFromString(file.read_bytes())
            if tensor.name and tensor.name != expected_name:
                raise ValueError(f"Tensor name {tensor.name!r} does not match model input {expected_name!r}")
            return onnx.numpy_helper.to_array(tensor, base_dir=str(file.parent))
        except Exception as error:
            raise ValueError(f"Cannot load TensorProto input {file}: {error}") from error

    if path.is_dir():
        files = {}
        for file in path.glob("input_*.pb"):
            match = re.fullmatch(r"input_(\d+)\.pb", file.name, re.IGNORECASE)
            if not match or int(match[1]) in files:
                raise ValueError(f"Invalid or duplicate input index: {file}")
            files[int(match[1])] = file
        if set(files) != set(range(len(input_names))):
            raise ValueError(f"Expected input_0.pb through input_{len(input_names) - 1}.pb in {path}")
        return {name: read_tensor(files[index], name) for index, name in enumerate(input_names)}
    if path.suffix.lower() == ".pb":
        if len(input_names) != 1:
            raise ValueError("For a multi-input model, --inputs must be a directory containing input_*.pb")
        return {input_names[0]: read_tensor(path, input_names[0])}
    raise ValueError("--inputs must be an ONNX test-data directory or TensorProto .pb file")


def make_demo_inputs(input_names):
    if input_names != ["x"]:
        raise ValueError("The default Abs demo requires exactly one input named 'x'")
    return {"x": np.random.default_rng(0).standard_normal((3, 4, 5)).astype(np.float32)}


def find_missing_qnn_dlls(directories):
    """Return required QNN test DLLs not found in any loader search directory."""
    return [name for name in REQUIRED_QNN_DLLS if not any((directory / name).is_file() for directory in directories)]


def check_qnn_dlls(directories):
    missing = find_missing_qnn_dlls(directories)
    if missing:
        locations = ", ".join(str(directory) for directory in directories)
        raise FileNotFoundError(
            f"Missing required QNN test DLL(s): {', '.join(missing)}. Search directories: {locations}"
        )


def get_string_attribute(attrs, name, default=""):
    value = attrs.get(name, default)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise ValueError(f"EPContext attribute {name!r} must be a string")


def is_qnn_context_node(node, attrs):
    return (
        node.op_type == "EPContext"
        and get_string_attribute(attrs, "source").lower() in QNN_CONTEXT_SOURCES
        and get_string_attribute(attrs, "ep_context_type", "bin").lower() == "bin"
    )


def validate_context_model(model_path):
    model = onnx.load(str(model_path), load_external_data=False)
    contexts = []
    for node in model.graph.node:
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        if not is_qnn_context_node(node, attrs):
            continue
        # Only the main EPContext node carries the binary in a shared context.
        if attrs.get("main_context", 1) != 1:
            continue
        if attrs.get("embed_mode", 1) != 0:
            raise ValueError("SSR recovery requires embed_mode=0; this model has an embedded QNN context")
        binary_name = get_string_attribute(attrs, "ep_cache_context")
        if not binary_name:
            raise ValueError("Main QNN EPContext node has no external ep_cache_context path")
        if Path(binary_name).is_absolute() or ".." in binary_name:
            raise ValueError("External ep_cache_context must be a relative path without '..'")
        binary = (model_path.parent / binary_name).resolve()
        if not binary.is_file() or binary.stat().st_size == 0:
            raise ValueError(f"Context binary is missing or empty: {binary}")
        contexts.append(str(binary))
    if not contexts:
        raise ValueError("No main QNN EPContext node found; supply an already generated external context model")
    return sorted(set(contexts))


def compare_outputs(outputs, references, names, rtol, atol):
    if not outputs or len(outputs) != len(references) or len(outputs) != len(names):
        raise ValueError("Output count differs from baseline")
    for name, actual, expected in zip(names, outputs, references, strict=True):
        if not isinstance(actual, np.ndarray) or not isinstance(expected, np.ndarray):
            raise ValueError(f"Only tensor outputs are supported: {name}")
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(f"Output shape/dtype differs from baseline: {name}")
        if actual.dtype.kind in "fc":
            if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                raise ValueError(f"Non-finite output: {name}")
            np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=f"Output {name}")
            continue
        np.testing.assert_array_equal(actual, expected, err_msg=f"Output {name}")


def output_tensor_path(output_dir, phase, index):
    return output_dir / f"{phase}_output_{index}.pb"


def save_outputs(output_dir, phase, names, outputs):
    for index, (name, output) in enumerate(zip(names, outputs, strict=True)):
        tensor = onnx.numpy_helper.from_array(output, name=name)
        output_tensor_path(output_dir, phase, index).write_bytes(tensor.SerializeToString())


def load_outputs(output_dir, phase, names):
    outputs = []
    for index, name in enumerate(names):
        path = output_tensor_path(output_dir, phase, index)
        tensor = onnx.TensorProto()
        try:
            tensor.ParseFromString(path.read_bytes())
            if tensor.name != name:
                raise ValueError(f"Tensor name {tensor.name!r} does not match output {name!r}")
            outputs.append(onnx.numpy_helper.to_array(tensor, base_dir=str(path.parent)))
        except Exception as error:
            raise ValueError(f"Cannot load TensorProto output {path}: {error}") from error
    return outputs


def run_worker(args):
    is_demo = args.model.resolve() == DEFAULT_MODEL
    # Keep DLL directories alive through session destruction and plugin unregistration.
    with contextlib.ExitStack() as stack:
        directories = [Path.cwd(), args.ep_library.parent, *args.dll_dir]
        for directory in dict.fromkeys(directories):
            stack.enter_context(os.add_dll_directory(str(directory)))

        if not hasattr(ort, "register_execution_provider_library"):
            raise RuntimeError("Install an ORT version with the plugin EP Python API")
        ort.set_default_logger_severity(2)  # Include warning-level recovery-success evidence.
        options = {}
        if args.ep_options:
            options = json.loads(args.ep_options.read_text(encoding="utf-8"))
            if not isinstance(options, dict) or any(not isinstance(v, str) for v in options.values()):
                raise ValueError("--ep-options must contain a JSON object with string values")
        backend = "QnnMockSSR.dll" if args.worker in ("mock", *UNRECOVERABLE_SSR_WORKERS) else "QnnHtp.dll"
        options.pop("backend_type", None)
        options["backend_path"] = backend
        session = None
        ort.register_execution_provider_library(REGISTRATION, str(args.ep_library))
        try:
            devices = [
                d
                for d in ort.get_ep_devices()
                if d.ep_name == REGISTRATION and d.device.type == ort.OrtHardwareDeviceType.NPU
            ]
            if not devices:
                raise RuntimeError("QNN plugin did not expose an NPU device")
            so = ort.SessionOptions()
            so.log_severity_level = 2
            so.add_session_config_entry("ep.context_file_path", str(args.context_model))
            if args.worker in CONTEXT_GENERATION_WORKERS:
                so.add_session_config_entry("ep.context_enable", "1")
                so.add_session_config_entry("ep.context_embed_mode", "1" if args.worker == "embed_generate" else "0")
            if not args.allow_cpu_fallback:
                so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            # SSR recovery is validated on one QNN NPU device per worker/session.
            device = devices[0]
            so.add_provider_for_devices([device], options)
            session_model = (
                args.model if args.worker in (*CONTEXT_GENERATION_WORKERS, "jit_ssr") else args.context_model
            )
            session = ort.InferenceSession(str(session_model), sess_options=so)
            input_names = [item.name for item in session.get_inputs()]
            feeds = (
                make_demo_inputs(input_names)
                if is_demo and args.inputs is None
                else load_inputs(args.inputs, input_names)
            )
            names = [item.name for item in session.get_outputs()]
            references = None
            if args.worker == "mock":
                references = load_outputs(args.output_dir, "baseline", names)
            if args.worker in UNRECOVERABLE_SSR_WORKERS:
                try:
                    session.run(names, feeds)
                except Exception as error:
                    error_text = str(error)
                    if RECOVERY_NOT_SUPPORTED not in error_text:
                        raise RuntimeError(f"Expected unrecoverable SSR error, got: {error_text}") from error
                    result = {
                        "phase": args.worker,
                        "outcome": "expected_unrecoverable_ssr",
                        "onnxruntime_version": ort.__version__,
                        "backend_path": backend,
                        "provider_options": options,
                        "error": error_text,
                    }
                    (args.output_dir / f"{args.worker}_result.json").write_text(
                        json.dumps(result, indent=2), encoding="utf-8"
                    )
                    print(f"{args.worker}: expected unrecoverable SSR", flush=True)
                    return
                raise RuntimeError("Expected unrecoverable SSR error, but inference succeeded")
            times = []
            runs = args.runs if args.worker == "mock" else 1
            for index in range(runs):
                start = time.perf_counter()
                outputs = session.run(names, feeds)
                times.append((time.perf_counter() - start) * 1000)
                compare_outputs(outputs, references if references is not None else outputs, names, args.rtol, args.atol)
                if index == 0 and args.worker != "generate":
                    save_outputs(args.output_dir, args.worker, names, outputs)
                print(f"{args.worker} Run {index + 1}/{runs}: PASS ({times[-1]:.3f} ms)", flush=True)
            result = {
                "phase": args.worker,
                "runs": runs,
                "run_ms": times,
                "output_names": names,
                "onnxruntime_version": ort.__version__,
                "backend_path": backend,
                "provider_options": options,
            }
            if args.worker != "generate":
                result["output_files"] = [
                    output_tensor_path(args.output_dir, args.worker, index).name for index in range(len(names))
                ]
            (args.output_dir / f"{args.worker}_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        finally:
            session = None
            ort.unregister_execution_provider_library(REGISTRATION)


def check_recovery_log(log):
    log = normalize_log(log)
    detected = log.count(DETECTED)
    recovered = log.count(RECOVERED)
    if detected != 1 or recovered != 1 or log.index(DETECTED) > log.index(RECOVERED):
        raise RuntimeError(
            f"Expected one SSR followed by one successful recovery; detected={detected}, recovered={recovered}"
        )
    return {"ssr_detected": detected, "recovery_succeeded": recovered}


def check_unrecoverable_ssr_log(log):
    log = normalize_log(log)
    detected = log.count(DETECTED)
    recovered = log.count(RECOVERED)
    if detected != 1 or recovered != 0 or RECOVERY_NOT_SUPPORTED not in log:
        raise RuntimeError(f"Expected one unrecoverable SSR; detected={detected}, recovery_succeeded={recovered}")
    return {"ssr_detected": detected, "recovery_succeeded": recovered, "outcome": "expected_unrecoverable_ssr"}


def normalize_log(log):
    """Normalize Windows subprocess output before matching ORT log messages."""
    return ANSI_ESCAPE.sub("", log.replace("\x00", ""))


def run_controller(args):
    if os.name != "nt" or platform.machine().lower() not in ("arm64", "aarch64"):
        raise RuntimeError("Run this script with native ARM64 Python on Windows with a supported QNN NPU")
    for name in ("model", "inputs", "ep_options", "ep_library"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.ep_library is None:
        if onnxruntime_qnn is None:
            raise RuntimeError("Install onnxruntime_qnn or provide --ep-library")
        args.ep_library = Path(onnxruntime_qnn.get_library_path()).resolve()
    is_demo = args.model == DEFAULT_MODEL
    if args.inputs is None and not is_demo:
        raise ValueError("Provide --inputs with a .pb file or test-data directory for a custom model")
    for path in (
        args.model,
        args.ep_library,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.inputs is not None and not args.inputs.exists():
        raise FileNotFoundError(args.inputs)
    args.dll_dir = [path.resolve() for path in args.dll_dir]
    for directory in args.dll_dir:
        if not directory.is_dir():
            raise NotADirectoryError(directory)
    dll_directories = list(dict.fromkeys([Path.cwd().resolve(), args.ep_library.parent, *args.dll_dir]))
    check_qnn_dlls(dll_directories)
    output_dir = args.output_dir or Path(f"mock_ssr_{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%fZ}")
    args.output_dir = output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "FAIL",
        "test": "mock SSR application E2E (no physical DSP reset)",
        "model": str(args.model),
        "ep_library": str(args.ep_library),
        "rtol": args.rtol,
        "atol": args.atol,
        "allow_cpu_fallback": args.allow_cpu_fallback,
    }
    try:
        report["inputs"] = str(args.inputs) if args.inputs else "generated default Abs input x[3,4,5] float32"
        original = onnx.load(str(args.model), load_external_data=False)
        has_context = any(
            is_qnn_context_node(node, {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute})
            for node in original.graph.node
        )
        args.context_model = args.model if has_context else args.output_dir / f"{args.model.stem}_ctx.onnx"
        report["context_model"] = str(args.context_model)
        if has_context:
            report["context_binaries"] = validate_context_model(args.context_model)
        common = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--model",
            str(args.model),
            "--ep-library",
            str(args.ep_library),
            "--output-dir",
            str(args.output_dir),
            "--runs",
            str(args.runs),
            "--rtol",
            str(args.rtol),
            "--atol",
            str(args.atol),
        ]
        if args.inputs:
            common.extend(["--inputs", str(args.inputs)])
        for directory in args.dll_dir:
            common.extend(["--dll-dir", str(directory)])
        if args.ep_options:
            common.extend(["--ep-options", str(args.ep_options)])
        if args.allow_cpu_fallback:
            common.append("--allow-cpu-fallback")
        if has_context:
            phase_context_models = {"baseline": args.context_model, "mock": args.context_model}
        else:
            embed_context_model = args.output_dir / f"{args.model.stem}_embed_ctx.onnx"
            aot_generation_context_model = args.output_dir / f"{args.model.stem}_generation_ssr_ctx.onnx"
            phase_context_models = {
                "aot_generation_ssr": aot_generation_context_model,
                "generate": args.context_model,
                "baseline": args.context_model,
                "mock": args.context_model,
                "embed_generate": embed_context_model,
                "embed_ssr": embed_context_model,
                "jit_ssr": args.context_model,
            }
            report["embedded_context_model"] = str(embed_context_model)
        for phase, context_model in phase_context_models.items():
            log_path = args.output_dir / f"{phase}.log"
            print(f"Running {phase}; log: {log_path}", flush=True)
            with log_path.open("w", encoding="utf-8") as log_file:
                completed = subprocess.run(
                    [*common, "--context-model", str(context_model), "--worker", phase],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=False,
                )
            if completed.returncode:
                raise RuntimeError(f"{phase} failed (exit {completed.returncode}); inspect {log_path}")
            report[phase] = json.loads((args.output_dir / f"{phase}_result.json").read_text(encoding="utf-8"))
            if phase == "generate":
                report["context_binaries"] = validate_context_model(args.context_model)
        report["evidence"] = {
            "mock": check_recovery_log((args.output_dir / "mock.log").read_text(encoding="utf-8", errors="replace"))
        }
        for phase in UNRECOVERABLE_SSR_WORKERS:
            if phase in phase_context_models:
                report["evidence"][phase] = check_unrecoverable_ssr_log(
                    (args.output_dir / f"{phase}.log").read_text(encoding="utf-8", errors="replace")
                )
        report["status"] = "PASS"
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS: mock SSR recovery and unrecoverable SSR cases passed. Report: {args.output_dir / 'report.json'}")


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.worker:
            run_worker(args)
        else:
            run_controller(args)
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
