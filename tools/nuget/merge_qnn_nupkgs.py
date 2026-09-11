# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT


"""Merge per-architecture Qualcomm.ML.OnnxRuntime.QNN NuGet packages into one.

The QNN NuGet build produces a separate ``.nupkg`` per architecture: an ARM64X
package whose native binaries live under ``runtimes/win-arm64/native`` and an
x64 package whose binaries live under ``runtimes/win-x64/native``. Because those
folders do not collide, the two packages can be merged into a single package
purely at the zip level -- no rebuild required.

This script discovers the two per-architecture packages in an input directory
and merges the x64 ``runtimes`` tree into the ARM64X package (used as the base,
so all of its shared metadata/targets/props/managed-DLL content is carried
through unmodified), writing one merged ``.nupkg``.

Usage:
    python merge_qnn_nupkgs.py --input_dir <folder-with-nupkgs> --output_dir <out>
"""

import argparse
import re
import sys
import zipfile
from pathlib import Path

PACKAGE_ID = "Qualcomm.ML.OnnxRuntime.QNN"

# RID folder each per-arch package is expected to own.
ARM64_RUNTIME_PREFIX = "runtimes/win-arm64/"
X64_RUNTIME_PREFIX = "runtimes/win-x64/"


def parse_arguments():
    parser = argparse.ArgumentParser(description=f"Merge per-architecture {PACKAGE_ID} nupkgs into a single package.")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing the per-architecture .nupkg files to merge.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory the merged .nupkg is written to.",
    )
    return parser.parse_args()


def _normalize(name):
    # nupkg entries always use forward slashes; normalize so prefix checks are reliable.
    return name.replace("\\", "/")


def _runtime_rids(zip_path):
    """Return the set of ``runtimes/win-<rid>/`` prefixes present in a package."""
    rids = set()
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            norm = _normalize(name)
            if norm.startswith(ARM64_RUNTIME_PREFIX):
                rids.add(ARM64_RUNTIME_PREFIX)
            elif norm.startswith(X64_RUNTIME_PREFIX):
                rids.add(X64_RUNTIME_PREFIX)
    return rids


def discover_packages(input_dir):
    """Locate exactly one ARM64X package and one x64 package in ``input_dir``.

    The two per-arch packages are distinguished by which ``runtimes`` folder each
    carries rather than by filename (they typically share the same version, and
    callers may stage each in its own subfolder to avoid same-name collisions).
    The search is recursive so packages staged in per-arch subfolders are found.
    """
    candidates = sorted(Path(input_dir).rglob("*.nupkg"))
    if not candidates:
        raise SystemExit(f"error: no '*.nupkg' files found under '{input_dir}'.")

    arm64_pkg = None
    x64_pkg = None
    for pkg in candidates:
        rids = _runtime_rids(pkg)
        has_arm64 = ARM64_RUNTIME_PREFIX in rids
        has_x64 = X64_RUNTIME_PREFIX in rids
        if has_arm64 and has_x64:
            raise SystemExit(
                f"error: '{pkg.name}' already contains both win-arm64 and win-x64 runtimes; "
                "expected two disjoint single-arch packages, not an already-merged one."
            )
        if has_arm64:
            if arm64_pkg is not None:
                raise SystemExit(
                    f"error: found more than one package with win-arm64 runtimes ('{arm64_pkg.name}' and '{pkg.name}')."
                )
            arm64_pkg = pkg
        elif has_x64:
            if x64_pkg is not None:
                raise SystemExit(
                    f"error: found more than one package with win-x64 runtimes ('{x64_pkg.name}' and '{pkg.name}')."
                )
            x64_pkg = pkg

    if arm64_pkg is None or x64_pkg is None:
        raise SystemExit(
            "error: could not find both a win-arm64 package and a win-x64 package in "
            f"'{input_dir}'. Found: "
            f"arm64={arm64_pkg.name if arm64_pkg else None}, "
            f"x64={x64_pkg.name if x64_pkg else None}."
        )
    return arm64_pkg, x64_pkg


def merge(arm64_pkg, x64_pkg, output_dir):
    """Merge the x64 runtimes tree into the arm64x base package.

    Shared (non-runtimes) files are taken unconditionally from the ARM64X base
    package -- no byte-identical check is performed. In practice these files
    (nuspec, [Content_Types].xml, _rels/.rels, core-properties, the managed
    helper DLL) are all regenerated per-build/per-pack and legitimately differ
    between the two per-arch packages even when built from the same commit, so
    comparing them produced repeated false-positive merge failures.
    """
    with zipfile.ZipFile(arm64_pkg) as base_zip, zipfile.ZipFile(x64_pkg) as x64_zip:
        x64_names = {_normalize(n): n for n in x64_zip.namelist()}

        # Sanity: the base must not already carry x64 runtimes, and the x64 package
        # must actually contain them.
        x64_runtime_entries = [n for n in x64_names if n.startswith(X64_RUNTIME_PREFIX)]
        if not x64_runtime_entries:
            raise SystemExit(f"error: '{x64_pkg.name}' contains no '{X64_RUNTIME_PREFIX}' entries.")

        version = _derive_version(base_zip, arm64_pkg)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{PACKAGE_ID}.{version}.nupkg"

        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as out_zip:
            # 1. Everything from the base (arm64x) package verbatim.
            for info in base_zip.infolist():
                out_zip.writestr(info, base_zip.read(info.filename))
            # 2. Only the x64 runtimes tree from the x64 package.
            for info in x64_zip.infolist():
                norm = _normalize(info.filename)
                if norm.startswith(X64_RUNTIME_PREFIX):
                    out_zip.writestr(info, x64_zip.read(info.filename))

    return out_path


def _derive_version(base_zip, base_pkg_path):
    """Read the package version from the base package's .nuspec, falling back to the filename."""
    nuspec_name = next((n for n in base_zip.namelist() if _normalize(n).endswith(".nuspec")), None)
    if nuspec_name is not None:
        text = base_zip.read(nuspec_name).decode("utf-8", errors="replace")
        match = re.search(r"<version>\s*(.*?)\s*</version>", text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

    # Fallback: strip the "Qualcomm.ML.OnnxRuntime.QNN." prefix and ".nupkg" suffix.
    stem = Path(base_pkg_path).name
    prefix = f"{PACKAGE_ID}."
    if stem.startswith(prefix) and stem.endswith(".nupkg"):
        return stem[len(prefix) : -len(".nupkg")]
    raise SystemExit(f"error: could not derive package version from '{stem}'.")


def _version_from_pkg(pkg_path):
    with zipfile.ZipFile(pkg_path) as zf:
        return _derive_version(zf, pkg_path)


# Dev-build versions embed a per-job build timestamp (see OnnxRuntime.CSharp.proj's
# CurrentDate/CurrentTime) ahead of the commit hash, e.g. "2.5.0-dev-20260713-0920-f400981".
# That timestamp legitimately differs between the two per-arch CI jobs even when both are
# built from the same commit, so it's stripped out before comparing -- what actually matters
# is that both packages came from the same commit, not that they were packed at the same minute.
_DEV_VERSION_RE = re.compile(r"^(?P<base>.+-dev)-\d{8}-\d{4}-(?P<commit>[0-9a-fA-F]+)$")


def _version_identity(version):
    """Return a comparable form of `version` with any dev-build timestamp stripped out."""
    match = _DEV_VERSION_RE.match(version)
    if match:
        return f"{match.group('base')}-{match.group('commit')}"
    return version


def _check_versions_match(arm64_pkg, x64_pkg):
    arm64_version = _version_from_pkg(arm64_pkg)
    x64_version = _version_from_pkg(x64_pkg)
    if _version_identity(arm64_version) != _version_identity(x64_version):
        raise SystemExit(
            f"error: ARM64X package version '{arm64_version}' does not match "
            f"x64 package version '{x64_version}'. Refusing to merge mismatched packages."
        )


def main():
    args = parse_arguments()

    if not Path(args.input_dir).is_dir():
        raise SystemExit(f"error: --input_dir '{args.input_dir}' is not a directory.")

    arm64_pkg, x64_pkg = discover_packages(args.input_dir)
    print(f"ARM64x base package: {arm64_pkg.name}")
    print(f"x64 package:         {x64_pkg.name}")

    _check_versions_match(arm64_pkg, x64_pkg)

    out_path = merge(arm64_pkg, x64_pkg, args.output_dir)
    print(f"Merged package written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
