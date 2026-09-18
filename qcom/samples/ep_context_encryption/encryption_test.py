#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
#
# Drives the full prepare_app -> run_app round trip and reports PASS/FAIL.
#
# Runs prepare_app (compile + encrypt, dumps answer_prepare.raw from the
# plaintext model) then run_app (decrypt + run, dumps answer_run.raw from the
# decrypted context model), then compares the two answer files itself.
#
# Usage:
#   python encryption_test.py <prepare_app.exe> <run_app.exe> <input_model.onnx>
#                              <input.raw> [--xor-key 5a] [--tol 1e-3]
#                              [--workdir DIR] [--htp-arch ARCH]

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepare_app", help="path to prepare_app(.exe)")
    parser.add_argument("run_app", help="path to run_app(.exe)")
    parser.add_argument("input_model", help="ONNX model prepare_app compiles (e.g. a QDQ model)")
    parser.add_argument("input_raw", help="float32 .raw input fed to both the plaintext and decrypted model")
    parser.add_argument("--xor-key", default=DEFAULT_XOR_KEY, help="1-byte XOR key in hex (default 5a)")
    parser.add_argument("--tol", type=float, default=DEFAULT_ABS_TOL, help="max allowed abs diff")
    parser.add_argument("--workdir", default=".", help="directory for intermediate/output files")
    parser.add_argument("--htp-arch", default=None, help="optional target HTP arch passed to prepare_app")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    ctx_model = workdir / "enc_ctx.onnx"
    cipher_bin = workdir / "enc_cipher.bin"
    answer_prepare = workdir / "answer_prepare.raw"
    answer_run = workdir / "answer_run.raw"

    prepare_cmd = [
        args.prepare_app,
        args.input_model,
        str(ctx_model),
        str(cipher_bin),
        args.xor_key,
        args.input_raw,
        str(answer_prepare),
    ]
    if args.htp_arch:
        prepare_cmd.append(args.htp_arch)
    if not run_app(prepare_cmd, "prepare_app"):
        return 1

    run_cmd = [args.run_app, str(ctx_model), str(cipher_bin), args.xor_key, args.input_raw, str(answer_run)]
    if not run_app(run_cmd, "run_app"):
        return 1

    if not compare_raw(answer_prepare, answer_run, args.tol):
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
