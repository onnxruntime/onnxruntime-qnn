#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Parse an Application Verifier ``Leak`` layer XML log and report leaks.

App Verifier writes one ``<avrf:logEntry LayerName="Leak">`` per leaked heap
allocation when a DLL is unloaded (``FreeLibrary`` brings its refcount to 0).
The log looks like::

    <avrf:logfile xmlns:avrf="Application Verifier">
      <avrf:logSession PID="..." Version="2">
        <avrf:logEntry LayerName="Leak" StopCode="0x900" Severity="Error">
          <avrf:message>A heap allocation was leaked.</avrf:message>
          <avrf:parameter1>ADDR - Address of the leaked allocation. ...</avrf:parameter1>
          <avrf:parameter3>ADDR - Address of the owner dll name. ...</avrf:parameter3>
          <avrf:parameter4>BASE - Base of the owner dll. ...</avrf:parameter4>
          <avrf:stackTrace>
            <avrf:trace>module!symbol+off (src\\path @ line)</avrf:trace>
            ...
          </avrf:stackTrace>
        </avrf:logEntry>
      </avrf:logSession>
    </avrf:logfile>

Usage in CI: register App Verifier with the QAIRT backend DLLs excluded, run
``onnxruntime_provider_test``, ``appverif -export log ... -with to=result.xml``,
then run this parser on ``result.xml``. Any leak entry fails the job.

The owner DLL of a leak cannot be identified from ``parameter4`` (the base
address is randomised by ASLR and not comparable across runs). Instead we read
the call stack: a leak swept through ``EpLibraryPlugin::Unload`` /
``UnregisterExecutionProviderLibrary`` was found when the EP DLL unloaded, i.e.
it belongs to ``onnxruntime_providers_qnn.dll`` -- the target of this check.

Exit code: 0 if no leaks, 1 if any leak entry is present (or the file is
missing / malformed).
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

__all__ = [
    "LeakEntry",
    "StackFrame",
    "parse_leaks",
    "format_report",
]

# The avrf namespace URI is the literal string "Application Verifier" (with a
# space), so ElementTree tags are "{Application Verifier}name".
_NS = "Application Verifier"


def _tag(name: str) -> str:
    return f"{{{_NS}}}{name}"


# Stack-frame markers that indicate the leak was reported while the QNN EP DLL
# was being unloaded -- i.e. the allocation is owned by
# onnxruntime_providers_qnn.dll.
_EP_UNLOAD_MARKERS = (
    "EpLibraryPlugin::Unload",
    "UnregisterExecutionProviderLibrary",
)


class StackFrame:
    """One ``<avrf:trace>`` line, e.g. ``module!symbol+off (src\\path @ 123)``."""

    def __init__(self, raw: str) -> None:
        self.raw = raw.strip()
        self.module = ""
        self.symbol = ""
        self.source = ""  # "path @ line" when present, else ""
        self._parse()

    def _parse(self) -> None:
        text = self.raw
        # Split off the trailing "(source @ line)" if present.
        if text.endswith(")") and "(" in text:
            head, _, tail = text.rpartition("(")
            src = tail[:-1].strip()  # drop ")"
            # "( @ 0)" means no real source info.
            if src and src != "@ 0":
                self.source = src
            text = head.strip()
        # head is "module!symbol+off"
        if "!" in text:
            self.module, _, self.symbol = text.partition("!")
        else:
            self.symbol = text

    @property
    def has_source(self) -> bool:
        return bool(self.source)


class LeakEntry:
    """A single leaked heap allocation reported by App Verifier."""

    def __init__(
        self,
        leaked_address: str,
        owner_dll_base: str,
        stack: list[StackFrame],
        time: str = "",
    ) -> None:
        self.leaked_address = leaked_address
        self.owner_dll_base = owner_dll_base
        self.stack = stack
        self.time = time

    @property
    def is_ep_dll_leak(self) -> bool:
        """True if the call stack shows the leak surfaced at EP DLL unload."""
        return any(
            marker in frame.raw
            for frame in self.stack
            for marker in _EP_UNLOAD_MARKERS
        )

    @property
    def origin_frame(self) -> StackFrame | None:
        """The most relevant frame: the test that triggered the allocation.

        Prefer a frame in the QNN test sources (e.g.
        ``onnxruntime_provider_test!...\\qnn\\gemm_test.cc @ 80``), skipping the
        ORT-core unload machinery and MSVC/STL internal frames that carry a
        source path but aren't informative.
        """
        # 1. A frame in the QNN provider tests is the real trigger.
        for frame in self.stack:
            src = frame.source.replace("/", "\\").lower()
            if "test\\providers\\qnn" in src:
                return frame
        # 2. Otherwise the first test-binary frame with source that isn't STL/CRT.
        for frame in self.stack:
            if not frame.has_source or "onnxruntime_provider_test!" not in frame.raw:
                continue
            src = frame.source.replace("/", "\\").lower()
            if "\\vc\\tools\\msvc" in src or "\\crt\\" in src or "\\include\\" in src:
                continue
            return frame
        # 3. Fall back to the first frame carrying any source location.
        for frame in self.stack:
            if frame.has_source:
                return frame
        return None


def _leading_hex(text: str) -> str:
    """Return the leading hex token of a parameter string ('24cd... - ...')."""
    token = text.strip().split(" ", 1)[0].split("-", 1)[0].strip()
    return token


def parse_leaks(xml_text: str) -> list[LeakEntry]:
    """Parse App Verifier XML text into a list of Leak entries."""
    # Be lenient: an empty/whitespace file means "no leaks".
    if not xml_text.strip():
        return []
    root = ET.fromstring(xml_text)
    leaks: list[LeakEntry] = []
    for entry in root.iter(_tag("logEntry")):
        if entry.get("LayerName") != "Leak":
            continue
        p1 = entry.findtext(_tag("parameter1"), default="")
        p4 = entry.findtext(_tag("parameter4"), default="")
        stack_el = entry.find(_tag("stackTrace"))
        frames: list[StackFrame] = []
        if stack_el is not None:
            for trace in stack_el.findall(_tag("trace")):
                if trace.text:
                    frames.append(StackFrame(trace.text))
        leaks.append(
            LeakEntry(
                leaked_address=_leading_hex(p1),
                owner_dll_base=_leading_hex(p4),
                stack=frames,
                time=entry.get("Time", ""),
            )
        )
    return leaks


def format_report(leaks: list[LeakEntry], source: str, verbose: bool) -> str:
    """Render a human-readable report for CI logs."""
    lines: list[str] = []
    if not leaks:
        lines.append(f"PASS: no leaks detected in {source}")
        return "\n".join(lines)

    ep_leaks = sum(leak.is_ep_dll_leak for leak in leaks)
    lines.append(
        f"FAIL: {len(leaks)} leak(s) detected in {source} "
        f"({ep_leaks} attributed to onnxruntime_providers_qnn.dll)"
    )
    for i, leak in enumerate(leaks, 1):
        owner = "EP DLL (onnxruntime_providers_qnn.dll)" if leak.is_ep_dll_leak else "other"
        lines.append("")
        lines.append(f"  [{i}] leaked allocation @ 0x{leak.leaked_address}  owner: {owner}")
        origin = leak.origin_frame
        if origin is not None:
            lines.append(f"      origin: {origin.symbol}  ({origin.source})")
        if verbose:
            lines.append("      stack trace:")
            for frame in leak.stack:
                lines.append(f"        {frame.raw}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parse an App Verifier Leak XML log; exit 1 if any leak is found."
    )
    parser.add_argument("xml_path", type=Path, help="Path to the exported App Verifier XML log.")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print the full call stack for each leak.",
    )
    args = parser.parse_args(argv)

    if not args.xml_path.is_file():
        print(f"FAIL: App Verifier log not found: {args.xml_path}", file=sys.stderr)
        return 1

    xml_text = args.xml_path.read_text(encoding="utf-8", errors="replace")
    try:
        leaks = parse_leaks(xml_text)
    except ET.ParseError as exc:
        print(f"FAIL: could not parse {args.xml_path}: {exc}", file=sys.stderr)
        return 1

    print(format_report(leaks, str(args.xml_path), args.verbose))
    return 1 if leaks else 0


if __name__ == "__main__":
    sys.exit(main())
