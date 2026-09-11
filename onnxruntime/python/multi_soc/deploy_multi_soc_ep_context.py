#!/usr/bin/env python3
# -*- mode: python -*-
# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""List and remove HTP_CACHE_RECORD entries inside a FCB / EP-context file.

A trimmed-down sibling of ``qairt-dlc-deploy`` that operates only on HTP cache
records:

  * ``--list``          : list each HTP cache record with its DSP arch, SoC
                          model and VTCM size (read via ``libDlModelToolsPy``).
  * ``--remove_record`` : remove records by name and drop the matching SoC
                          entries from the accompanying ONNX model's QNN EP
                          compatibility metadata, keeping the two in sync.

The metadata lives in ``metadata_props`` under key
``ep_compatibility_info.QNNExecutionProvider``. For a Flexible Context Binary the
QNN EP writes the V2 layout, a single ':'-separated string::

    v2:<BackendId>:<SDK>:<BackendApi>:<HtpArch1>,...:<SoCModel1>,...:<VtcmMb1>,...:<IsHtpUsrDrv>

Fields 4-6 (HtpArch, SoCModel, VtcmMb) are parallel comma-separated lists: entry
*i* of each is one SoC. A removed record is matched to the entry at the same
index *i* it holds in the sorted HTP cache record name list, and that index is
dropped from all three lists.
"""

import argparse
import os
import re
import sys
import tempfile
import traceback

import onnx

# libDlModelToolsPy312 is a QAIRT SDK pybind extension copied flat into this
# package at wheel-build time; make its directory importable regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Only the Python 3.12 build is bundled into the wheel, so it is the sole module
# imported here; all classes/enums live at its top level.
try:
    import libDlModelToolsPy312 as modeltools
except ImportError as import_error:
    # This is the QAIRT SDK's libDlModelToolsPy extension. It fails to import in
    # two common situations; spell both out so the user knows how to fix it
    # instead of getting a bare ModuleNotFoundError traceback.
    running_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    sys.exit(
        f"Failed to import 'libDlModelToolsPy312' ({import_error}).\n"
        f"This tool requires Python 3.12 (currently running {running_version}); the extension is\n"
        "bundled only in the onnxruntime-qnn Python 3.12 wheel. To fix:\n"
        "  - Run under Python 3.12 from an installed onnxruntime-qnn wheel, e.g.\n"
        "    'python -m onnxruntime_qnn.deploy_multi_soc_ep_context ...'.\n"
        "  - When running from a source checkout instead, put the QAIRT SDK's\n"
        "    'lib/python/qti/aisw/dlc_utils/<platform>' directory on PYTHONPATH."
    )


# Key under which the QNN EP stores per-SoC compatibility info in metadata_props.
_EP_COMPAT_KEY = "ep_compatibility_info.QNNExecutionProvider"

# Positions of the parallel (DspArch, SoCModel, VTCMSize) lists in the
# ':'-separated value, and the expected field count.
_LIST_FIELD_INDICES = (4, 5, 6)
_NUM_FIELDS = 8

# V2-format field layout: the value is only editable when field 0 is "v2".
_VERSION_FIELD_INDEX = 0
_EXPECTED_VERSION = "v2"

# Attribute names on the EPContext node, mirroring onnx_ctx_model_helper.h.
_EPCONTEXT_OP = "EPContext"
_EMBED_MODE_ATTR = "embed_mode"
_EP_CACHE_CONTEXT_ATTR = "ep_cache_context"


class DeployError(Exception):
    """Raised for an expected, user-facing failure. ``main`` catches it and prints
    the message (no traceback), then exits non-zero. Distinct from an incidental
    ``Exception`` (e.g. from onnx/modeltools or a bug), which keeps its traceback."""


def _safe(fn, *args, default="N/A"):
    """Call a reader accessor, returning ``default`` on error or empty value so
    one bad record can't abort the listing."""
    try:
        value = fn(*args)
        return value if value not in (None, "") else default
    except Exception:
        return default


def get_htp_cache_record_names(reader):
    """Return the HTP_CACHE_RECORD names sorted by embedded number (natural
    order, e.g. ``cache_2`` before ``cache_10``) rather than lexicographically."""
    # Split into non-digit / digit runs; compare digit runs as ints.
    natural_sort_key = lambda name: [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name)]  # noqa: E731
    names = reader.get_record_names_by_record_type(modeltools.DlcRecordType.HTP_CACHE_RECORD, False)
    return sorted(names, key=natural_sort_key)


def list_records(reader):
    """Print a table of HTP cache records with DSP arch, SoC model and VTCM size."""
    record_names = get_htp_cache_record_names(reader)

    if not record_names:
        print("No HTP_CACHE_RECORD entries found in the FCB.")
        return

    rows = []
    for name in record_names:
        dsp_arch = _safe(reader.get_cache_dsp_arch, name)
        soc_model = _safe(reader.get_cache_soc_model, name)

        # VTCM size is per-graph; report the first graph's value.
        vtcm = "N/A"
        num_graphs = _safe(reader.get_cache_num_of_graphs, name, default=0)
        if num_graphs:
            vtcm = _safe(reader.get_htp_cache_graph_vtcm_size_by_index, name, 0)

        rows.append(
            (
                name,
                f"v{dsp_arch}" if dsp_arch != "N/A" else "N/A",
                str(soc_model),
                str(vtcm),
            )
        )

    headers = ("Record Name", "DSP Arch", "SoC Model", "VTCM Size")
    widths = [max(len(headers[col]), *(len(row[col]) for row in rows)) for col in range(len(headers))]

    def fmt(cells):
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print(fmt(headers))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in rows:
        print(fmt(row))


def remove_records(updater, record_name):
    """Remove one HTP cache record from the updater's in-memory state. Does not
    persist; the caller saves once after all removals so a multi-record removal
    is all-or-nothing."""
    record_name = record_name.strip()
    if not record_name:
        return

    remove_result = updater.remove_record_and_get_names(record_name, modeltools.DlcRecordType.HTP_CACHE_RECORD)
    remove_success, removed_record_names = remove_result

    if not remove_success:
        raise DeployError(
            f"Removing record {record_name} unsuccessful. Please make sure the "
            "correct HTP_CACHE_RECORD name is passed and that it is not a "
            "mandatory record."
        )

    print(f"Removed record {record_name} from the FCB (pending save).")
    orphaned = [r for r in removed_record_names if r != record_name]
    if orphaned:
        print(f"Also removed {len(orphaned)} orphaned child record(s): {', '.join(orphaned)}")


def _apply_metadata_removal(model, indices_to_remove):
    """Drop ``indices_to_remove`` from the parallel DspArch/SoCModel/VTCMSize
    lists in ``model``'s QNN EP compatibility metadata, in place. The indices are
    positions in the sorted record name list, which align with the SoC entries.
    Caller loads and saves the model (shared by the non-embed and embed flows).
    """
    if not indices_to_remove:
        return

    entry = next((p for p in model.metadata_props if p.key == _EP_COMPAT_KEY), None)
    if entry is None:
        print(f"Metadata key '{_EP_COMPAT_KEY}' not found in the ONNX model.")
        return

    fields = entry.value.split(":")
    if len(fields) != _NUM_FIELDS:
        raise DeployError(
            f"Unexpected '{_EP_COMPAT_KEY}' value: expected {_NUM_FIELDS} "
            f"':'-separated fields, got {len(fields)}. ONNX model left unchanged."
        )

    # The list positions below are only valid for V2; refuse anything else.
    version = fields[_VERSION_FIELD_INDEX]
    if version != _EXPECTED_VERSION:
        raise DeployError(
            f"Unsupported '{_EP_COMPAT_KEY}' version '{version}': expected "
            f"'{_EXPECTED_VERSION}'. ONNX model left unchanged."
        )

    # Parse the parallel lists; they must be equal length.
    lists = [fields[i].split(",") for i in _LIST_FIELD_INDICES]
    lengths = {len(lst) for lst in lists}
    if len(lengths) != 1:
        raise DeployError(
            f"DspArch/SoCModel/VTCMSize lists have mismatched lengths "
            f"{sorted(lengths)}; refusing to edit '{_EP_COMPAT_KEY}'."
        )
    (entry_count,) = lengths

    drop = {i for i in indices_to_remove if 0 <= i < entry_count}
    if not drop:
        raise DeployError(
            f"None of the removed records map to an entry in '{_EP_COMPAT_KEY}' "
            f"(it lists {entry_count} SoC entr{'y' if entry_count == 1 else 'ies'}, "
            f"requested indices {sorted(indices_to_remove)}). ONNX model left "
            "unchanged."
        )

    keep = [i for i in range(entry_count) if i not in drop]
    for field_pos, lst in zip(_LIST_FIELD_INDICES, lists, strict=True):
        fields[field_pos] = ",".join(lst[i] for i in keep)

    entry.value = ":".join(fields)
    print(
        f"Updated '{_EP_COMPAT_KEY}': removed {len(drop)} SoC "
        f"entr{'y' if len(drop) == 1 else 'ies'} "
        f"(indices {', '.join(str(i) for i in sorted(drop))}), "
        f"{len(keep)} remaining."
    )


def _update_ep_cache_context_path(model, bin_path):
    """Repoint every non-embed EPContext node's ``ep_cache_context`` (which holds
    the FCB filename) at ``bin_path``'s basename, so the model references the
    freshly written ``--output_bin`` instead of dangling at the old name."""
    new_value = os.path.basename(bin_path)
    for node in model.graph.node:
        if node.op_type != _EPCONTEXT_OP:
            continue
        attrs = {a.name: a for a in node.attribute}
        embed_attr = attrs.get(_EMBED_MODE_ATTR)
        embed_mode = int(embed_attr.i) if embed_attr is not None else 1
        if embed_mode != 0:
            continue
        cache_attr = attrs.get(_EP_CACHE_CONTEXT_ATTR)
        if cache_attr is None:
            continue
        old_value = cache_attr.s.decode("utf-8", errors="replace") if cache_attr.s else ""
        cache_attr.s = new_value.encode("utf-8")
        if old_value != new_value:
            print(
                f"Updated '{_EP_CACHE_CONTEXT_ATTR}' on EPContext node "
                f"'{node.name or '<unnamed>'}': '{old_value}' -> '{new_value}'."
            )


def _resolve_removed_indices(updater, remove_record_arg):
    """Resolve a comma-separated ``--remove_record`` value to (names, indices),
    where each index is the record's position in the sorted record list — the
    contract linking it to its SoC entry, shared by both flows."""
    sorted_names = get_htp_cache_record_names(updater)
    name_to_index = {name: idx for idx, name in enumerate(sorted_names)}

    names_to_remove = []
    removed_indices = []
    for record in remove_record_arg.split(","):
        name = record.strip()
        if not name:
            continue
        if name not in name_to_index:
            raise DeployError(
                f"'{name}' is not an HTP cache record in the FCB; its ONNX metadata entry cannot be located by index."
            )
        names_to_remove.append(name)
        removed_indices.append(name_to_index[name])
    return names_to_remove, removed_indices


def _find_embedded_ctx_node(model):
    """Return ``(node, ep_cache_context_attr, embed_mode)`` for the EPContext node
    carrying the embedded QNN context binary. Exits when there is no EPContext
    node, when the model is non-embed (edit its standalone binary via
    ``--input_bin``), or when the ``ep_cache_context`` payload is missing/empty."""
    ctx_nodes = [n for n in model.graph.node if n.op_type == _EPCONTEXT_OP]
    if not ctx_nodes:
        raise DeployError("No EPContext node found in the ONNX model; it is not an EP-context model.")

    for node in ctx_nodes:
        attrs = {a.name: a for a in node.attribute}
        embed_attr = attrs.get(_EMBED_MODE_ATTR)
        embed_mode = int(embed_attr.i) if embed_attr is not None else 1
        if embed_mode == 0:
            raise DeployError(
                "The ONNX model is a non-embed-mode EP-context model. Edit its "
                "standalone '*_qnn.bin' binary directly via --input_bin instead."
            )

        cache_attr = attrs.get(_EP_CACHE_CONTEXT_ATTR)
        if cache_attr is None or not cache_attr.s:
            raise DeployError(
                f"EPContext node '{node.name}' has no '{_EP_CACHE_CONTEXT_ATTR}' payload; nothing to extract."
            )
        return node, cache_attr, embed_mode

    # No EPContext node carried an ep_cache_context payload.
    raise DeployError("No embedded QNN context binary found in any EPContext node.")


def run_embed_mode(args):
    """List/remove records inside an embed-mode EP-context ONNX model. The binary
    bytes in the EPContext node are staged to a temp file, edited via DlcUpdater,
    then read back and re-embedded into ``--output_onnx``."""
    if not args.input_onnx:
        raise DeployError(
            "Without --input_bin the tool runs in embed mode and reads the "
            "embedded binary from --input_onnx, which must be provided."
        )
    if args.inplace:
        if args.output_onnx:
            raise DeployError("--inplace cannot be combined with --output_onnx.")
        # Edit the model in place: re-embed the stripped binary into the input.
        args.output_onnx = args.input_onnx
    if args.remove_record and not args.output_onnx:
        raise DeployError(
            "--remove_record in embed mode requires --output_onnx for the edited "
            "ONNX model (or --inplace to overwrite --input_onnx)."
        )

    model = onnx.load(args.input_onnx)
    _, cache_attr, _ = _find_embedded_ctx_node(model)

    # Stage the embedded binary bytes on disk (DlcUpdater needs a file path): use
    # --temp_bin when given, else an auto temp file. Removed either way at the end.
    if args.temp_bin:
        tmp_path = args.temp_bin
        with open(tmp_path, "wb") as tmp_file:
            tmp_file.write(cache_attr.s)
    else:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".bin")
        with os.fdopen(tmp_fd, "wb") as tmp_file:
            tmp_file.write(cache_attr.s)
    try:
        updater = modeltools.DlcUpdater(tmp_path)
        if not updater.is_valid():
            raise DeployError("Could not open the embedded binary. The 'ep_cache_context' payload may be corrupt.")
        if not updater.initialize():
            raise DeployError("Could not initialize the updater on the embedded binary.")

        if args.remove_record:
            names_to_remove, removed_indices = _resolve_removed_indices(updater, args.remove_record)

            # Edit metadata in memory, remove records, then persist once — so a
            # multi-record removal is all-or-nothing and the re-embedded binary
            # stays consistent with the metadata.
            _apply_metadata_removal(model, removed_indices)
            for name in names_to_remove:
                remove_records(updater, name)

            if not updater.save():
                raise DeployError("Failed to save the stripped embedded FCB.")

            # Read the stripped FCB back and re-embed it.
            with open(tmp_path, "rb") as stripped:
                cache_attr.s = stripped.read()
            onnx.save(model, args.output_onnx)
            print(f"Wrote re-embedded EP-context model to {args.output_onnx}.")

        # List last so a combined --list reflects the post-removal state.
        if args.list:
            list_records(updater)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def run_non_embed_mode(args):
    """List/remove records inside a standalone (non-embed) binary. When removing,
    the stripped binary goes to ``--output_bin`` and the model's metadata to
    ``--output_onnx``, leaving both inputs untouched."""
    if args.remove_record:
        if not args.input_onnx:
            raise DeployError(
                "--remove_record requires --input_onnx so the model's QNN EP "
                "compatibility metadata can be kept in sync with the binary."
            )
        if args.inplace:
            if args.output_onnx or args.output_bin:
                raise DeployError("--inplace cannot be combined with --output_onnx / --output_bin.")
            # Edit both artifacts in place by pointing the outputs at the inputs.
            args.output_onnx = args.input_onnx
            args.output_bin = args.input_bin
        if not args.output_onnx:
            raise DeployError(
                "--remove_record requires --output_onnx for the updated ONNX model "
                "(or --inplace to overwrite the inputs)."
            )
        if not args.output_bin:
            raise DeployError(
                "--remove_record requires --output_bin for the stripped binary (or --inplace to overwrite the inputs)."
            )

    # DlcUpdater inherits the reader API, so one object covers list and remove.
    # When removing, give it --output_bin so the input --input_bin is untouched.
    if args.remove_record:
        updater = modeltools.DlcUpdater(args.input_bin, args.output_bin)
    else:
        updater = modeltools.DlcUpdater(args.input_bin)
    if not updater.is_valid():
        raise DeployError("Could not open the binary file. Please check the file name.")
    if not updater.initialize():
        raise DeployError("Could not initialize the updater.")

    if args.remove_record:
        names_to_remove, removed_indices = _resolve_removed_indices(updater, args.remove_record)

        # Validate/edit metadata in memory first (aborts before any write),
        # remove records, save the FCB, then save the ONNX — so both the
        # multi-record removal and the FCB/ONNX pair are all-or-nothing.
        model = onnx.load(args.input_onnx)
        _apply_metadata_removal(model, removed_indices)
        _update_ep_cache_context_path(model, args.output_bin)

        for name in names_to_remove:
            remove_records(updater, name)

        if not updater.save():
            raise DeployError(f"Failed to save the FCB to {args.output_bin}.")
        print(f"Successfully saved the stripped FCB to {args.output_bin}.")

        onnx.save(model, args.output_onnx)

    # List last so that, combined with --remove_record, the table reflects the
    # post-removal state for verification in one invocation.
    if args.list:
        list_records(updater)


def main():
    try:
        parser = argparse.ArgumentParser(
            description="List and remove HTP_CACHE_RECORD entries in a binary / EP-context file.",
            formatter_class=argparse.RawTextHelpFormatter,
            # Suppress the auto-added -h/--help (and its default group) so every
            # argument lands under a single custom "optional arguments" section.
            add_help=False,
        )
        optional = parser.add_argument_group("optional arguments")
        optional.add_argument(
            "-h",
            "--help",
            action="help",
            default=argparse.SUPPRESS,
            help="Show this help message and exit.",
        )
        # Inputs
        optional.add_argument(
            "--input_onnx",
            type=str,
            metavar="",
            required=False,
            help="Path to the input ONNX model. In non-embed mode (--input_bin)\n"
            "this is the model that accompanies the binary; it is required with\n"
            "--remove_record so the SoC entries matching the removed HTP cache\n"
            "records are dropped from the\n"
            "'ep_compatibility_info.QNNExecutionProvider' metadata value.\n"
            "In embed mode (no --input_bin) this is also the source of the\n"
            "embedded binary and must be provided.",
        )
        optional.add_argument(
            "--input_bin",
            metavar="",
            required=False,
            type=str,
            help="Path to a standalone binary / EP-context file (non-embed mode).\n"
            "Omit this for an embed-mode EP-context model, in which case the\n"
            "embedded binary is read from the ONNX model given by --input_onnx.",
        )
        # Actions
        optional.add_argument(
            "-l",
            "--list",
            action="store_true",
            required=False,
            help="List all HTP_CACHE_RECORD entries together with their DSP\n"
            "architecture, SoC model and (first graph) VTCM size.",
        )
        optional.add_argument(
            "-r",
            "--remove_record",
            type=str,
            metavar="",
            required=False,
            help="Remove HTP_CACHE_RECORD entries from the binary by name.\n"
            "The record type is fixed to HTP_CACHE_RECORD, so only the\n"
            "record name is required.\n"
            "To remove multiple records, separate the names with a comma:\n"
            "-r my_cache,my_cache_2\n"
            "Requires --input_onnx so the model's QNN EP compatibility\n"
            "metadata is kept in sync with the binary.",
        )
        # Outputs
        optional.add_argument(
            "--output_onnx",
            type=str,
            metavar="",
            required=False,
            help="Destination path for the updated ONNX model. Required with\n"
            "--remove_record in both modes. In embed mode the stripped binary is\n"
            "re-embedded into it; in non-embed mode it carries the updated\n"
            "compatibility metadata. The input --input_onnx is left untouched.",
        )
        optional.add_argument(
            "--output_bin",
            type=str,
            metavar="",
            required=False,
            help="Non-embed mode only: destination path for the stripped binary.\n"
            "Required with --remove_record; the input --input_bin is left\n"
            "untouched.",
        )
        optional.add_argument(
            "--temp_bin",
            type=str,
            metavar="",
            required=False,
            help="Embed mode only: path to stage the extracted binary while it is\n"
            "edited. When given, the file is written there; otherwise an\n"
            "auto-generated temporary file is used. The file is removed when\n"
            "the tool finishes either way.",
        )
        optional.add_argument(
            "--inplace",
            action="store_true",
            required=False,
            help="Update the input model (and, in non-embed mode, the input binary)\n"
            "in place instead of writing to new paths. Mutually exclusive with\n"
            "--output_onnx / --output_bin.",
        )

        if len(sys.argv) == 1:
            parser.print_help(sys.stderr)
            sys.exit(1)

        args = parser.parse_args()

        if not any([args.list, args.remove_record]):
            raise DeployError(
                "At least one action (list, remove_record) must be provided.\n"
                "Please refer to the help page for more detail."
            )

        if args.input_onnx and not os.path.isfile(args.input_onnx):
            raise DeployError(f"Could not find the ONNX model at {args.input_onnx}.")

        # No standalone binary -> embed mode (binary lives in the ONNX model).
        # Each mode validates its own arguments.
        if not args.input_bin:
            run_embed_mode(args)
        else:
            run_non_embed_mode(args)

    except SystemExit:
        raise
    except DeployError as deploy_error:
        # Expected, user-facing failure: print the message without a traceback.
        sys.exit(str(deploy_error))
    except Exception:
        traceback.print_exc()
        sys.exit(-2)


if __name__ == "__main__":
    main()
