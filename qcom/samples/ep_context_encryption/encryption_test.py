#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
#
# Drives the prepare_app -> run_app round trip and reports PASS/FAIL.
#
# Runs prepare_app (compile + encrypt, dumps answer_prepare.raw from the
# plaintext model) then run_app (decrypt + run, dumps answer_run.raw from the
# decrypted context model), then compares the two answer files itself.
#
# Same-machine usage (both exes installed locally):
#   python encryption_test.py roundtrip <prepare_app.exe> <run_app.exe> <input_model.onnx>
#                              <input.raw> [--xor-key 5a] [--tol 1e-3]
#                              [--workdir DIR] [--htp-arch ARCH]
#
# Cross-device usage (prepare_app and run_app run on different machines):
#   python encryption_test.py prepare <prepare_app.exe> <input_model.onnx> <input.raw>
#                              [--xor-key 5a] [--workdir DIR] [--htp-arch ARCH]
#   python encryption_test.py run <run_app.exe> <ctx_model> <cipher_bin>
#                              <input.raw> [--xor-key 5a] [--workdir DIR]
#   python encryption_test.py compare <answer_prepare.raw> <answer_run.raw> [--tol 1e-3]

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

DEFAULT_ABS_TOL = 1e-3
DEFAULT_XOR_KEY = "5a"


def run_app(args, label):
    print(f"[encryption_test] running: {' '.join(str(a) for a in args)}")
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        print(f"[encryption_test] FAIL: {label} exited with code {result.returncode}", file=sys.stderr)
        return False
    return True


def compare_raw(path_a, path_b, tol):
    a = np.fromfile(path_a, dtype=np.float32)
    b = np.fromfile(path_b, dtype=np.float32)

    if a.size != b.size:
        print(
            f"[encryption_test] FAIL: size mismatch: {path_a} has {a.size} value(s), {path_b} has {b.size} value(s).",
            file=sys.stderr,
        )
        return False

    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.all():
        bad = np.flatnonzero(~finite)[:5]
        for i in bad:
            print(f"[encryption_test] FAIL: non-finite value at index {i}: a={a[i]}, b={b[i]}", file=sys.stderr)
        return False

    abs_diff = np.abs(a - b)
    max_abs_diff = float(abs_diff.max()) if abs_diff.size else 0.0
    bad_mask = abs_diff > tol
    bad_count = int(bad_mask.sum())

    if bad_count > 0:
        bad_idx = np.flatnonzero(bad_mask)[:5]
        for i in bad_idx:
            print(
                f"[encryption_test] FAIL: mismatch at index {i}: a={a[i]}, b={b[i]} (|diff|={abs_diff[i]} > {tol})",
                file=sys.stderr,
            )
        print(
            f"[encryption_test] FAIL: {bad_count} of {a.size} value(s) outside tolerance {tol} "
            f"(max |diff| = {max_abs_diff})",
            file=sys.stderr,
        )
        return False

    print(f"[encryption_test] PASS: {a.size} value(s) within {tol} (max |diff| = {max_abs_diff})")
    return True


def do_prepare(args):
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    args.ctx_model = str(workdir / "enc_ctx.onnx")
    args.cipher_bin = str(workdir / "enc_cipher.bin")
    args.answer_prepare = str(workdir / "answer_prepare.raw")

    prepare_cmd = [
        args.prepare_app,
        args.input_model,
        args.ctx_model,
        args.cipher_bin,
        args.xor_key,
        args.input_raw,
        args.answer_prepare,
    ]
    if args.htp_arch:
        prepare_cmd.append(args.htp_arch)
    if not run_app(prepare_cmd, "prepare_app"):
        return 1

    print("[encryption_test] prepare done. Copy these to the run device's workdir:")
    print(f"  {args.ctx_model}")
    print(f"  {args.cipher_bin}")
    print(f"  {args.input_raw} (as input.raw, if not already there)")
    print(f"  {args.answer_prepare} (needed later for compare)")
    return 0


def do_run(args):
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    args.answer_run = str(workdir / "answer_run.raw")
    run_cmd = [args.run_app, args.ctx_model, args.cipher_bin, args.xor_key, args.input_raw, args.answer_run]
    if not run_app(run_cmd, "run_app"):
        return 1

    print("[encryption_test] run done. Copy this back to compare against answer_prepare.raw:")
    print(f"  {args.answer_run}")
    return 0


def do_compare(args):
    return 0 if compare_raw(args.answer_prepare, args.answer_run, args.tol) else 1


def do_roundtrip(args):
    if do_prepare(args) != 0:
        return 1
    if do_run(args) != 0:
        return 1
    return do_compare(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="phase", required=True)

    p_roundtrip = sub.add_parser("roundtrip", help="run prepare_app then run_app on the same machine")
    p_roundtrip.add_argument("prepare_app", help="path to prepare_app(.exe)")
    p_roundtrip.add_argument("run_app", help="path to run_app(.exe)")
    p_roundtrip.add_argument("input_model", help="ONNX model prepare_app compiles (e.g. a QDQ model)")
    p_roundtrip.add_argument("input_raw", help="float32 .raw input fed to both the plaintext and decrypted model")
    p_roundtrip.add_argument("--xor-key", default=DEFAULT_XOR_KEY, help="1-byte XOR key in hex (default 5a)")
    p_roundtrip.add_argument("--tol", type=float, default=DEFAULT_ABS_TOL, help="max allowed abs diff")
    p_roundtrip.add_argument("--workdir", default=".", help="directory for intermediate/output files")
    p_roundtrip.add_argument("--htp-arch", default=None, help="optional target HTP arch passed to prepare_app")
    p_roundtrip.set_defaults(func=do_roundtrip)

    p_prepare = sub.add_parser("prepare", help="run only prepare_app (e.g. on the x86 compile host)")
    p_prepare.add_argument("prepare_app", help="path to prepare_app(.exe)")
    p_prepare.add_argument("input_model", help="ONNX model prepare_app compiles (e.g. a QDQ model)")
    p_prepare.add_argument("input_raw", help="float32 .raw input fed to the plaintext model")
    p_prepare.add_argument("--xor-key", default=DEFAULT_XOR_KEY, help="1-byte XOR key in hex (default 5a)")
    p_prepare.add_argument("--workdir", default=".", help="directory for intermediate/output files")
    p_prepare.add_argument("--htp-arch", default=None, help="optional target HTP arch passed to prepare_app")
    p_prepare.set_defaults(func=do_prepare)

    p_run = sub.add_parser("run", help="run only run_app (e.g. on the ARM64 device)")
    p_run.add_argument("run_app", help="path to run_app(.exe)")
    p_run.add_argument("ctx_model", help="enc_ctx.onnx produced by the prepare phase (copied over)")
    p_run.add_argument("cipher_bin", help="enc_cipher.bin produced by the prepare phase (copied over)")
    p_run.add_argument("input_raw", help="float32 .raw input fed to the decrypted model")
    p_run.add_argument("--xor-key", default=DEFAULT_XOR_KEY, help="1-byte XOR key in hex (default 5a)")
    p_run.add_argument("--workdir", default=".", help="directory for the run-phase output file")
    p_run.set_defaults(func=do_run)

    p_compare = sub.add_parser("compare", help="compare answer_prepare.raw against answer_run.raw")
    p_compare.add_argument("answer_prepare", help="answer_prepare.raw from the prepare phase")
    p_compare.add_argument("answer_run", help="answer_run.raw from the run phase")
    p_compare.add_argument("--tol", type=float, default=DEFAULT_ABS_TOL, help="max allowed abs diff")
    p_compare.set_defaults(func=do_compare)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
