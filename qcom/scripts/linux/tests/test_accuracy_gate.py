# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
#
# Fixture-based unit tests for the accuracy-routing gate (accuracy_gate.py).
#
# All inputs are synthetic (hand-built gtest JSON + accuracy lists) -- no test
# binary, no build, no QNN backend. This is the class-1 self-verification that
# proves the gate's decision logic is correct: it is the ONLY layer that can
# meaningfully verify the routing without either a self-fulfilling
# generate-then-compare (class 2) or a landed golden store (class 3).

import json

import accuracy_gate as gate


# ---------------------------------------------------------------------------
# Synthetic gtest JSON builders (schema locked against a real snapshot run)
# ---------------------------------------------------------------------------
def _tc_passed(name):
    return {"name": name, "status": "RUN", "result": "COMPLETED"}


def _tc_drift(name):
    # DRIFT keeps result COMPLETED but carries failures -- must not be
    # classified by result alone.
    return {
        "name": name,
        "status": "RUN",
        "result": "COMPLETED",
        "failures": [{"failure": f"[QNN_SNAPSHOT_DRIFT] name={name}"}],
    }


def _tc_skipped(name, msg="[QNN_GOLDEN_ABSENT] op=... name=..."):
    return {
        "name": name,
        "status": "RUN",
        "result": "SKIPPED",
        "skipped": [{"message": msg}],
    }


def _tc_notrun(name):
    return {"name": name, "status": "NOTRUN", "result": "SUPPRESSED"}


def _suite(suite_name, testcases):
    return {"name": suite_name, "testsuite": testcases}


def _json_report(suites):
    return {"testsuites": suites}


def _write_json(tmp_path, obj, fname="snap.json"):
    p = tmp_path / fname
    p.write_text(json.dumps(obj))
    return str(p)


# Canonical accuracy case lists.
PLAIN = "QnnUnit_Accuracy_ClipPlainTest"
QDQF = "QnnUnit_Accuracy_ClipQDQFloatTest"
SNAP_PLAIN = "QnnUnit_Snapshot_ClipPlainTest"
SNAP_QDQF = "QnnUnit_Snapshot_ClipQDQFloatTest"
SESS_QDQF = "QnnUnit_SessionSnapshot_ClipQDQFloatTest"


# ===========================================================================
# classify_testcase
# ===========================================================================
def test_classify_passed():
    assert gate.classify_testcase(_tc_passed("Case/A")) == gate.PASSED


def test_classify_drift_ignores_completed_result():
    # The critical case: result is COMPLETED but failures present -> DRIFT.
    assert gate.classify_testcase(_tc_drift("Case/A")) == gate.DRIFT


def test_classify_skipped():
    assert gate.classify_testcase(_tc_skipped("Case/A")) == gate.SKIPPED


def test_classify_notrun_is_skipped():
    assert gate.classify_testcase(_tc_notrun("Case/A")) == gate.SKIPPED


def test_classify_unknown_result_is_skipped():
    # No PASS proof -> safe default is run (== SKIPPED bucket).
    assert gate.classify_testcase({"name": "Case/A"}) == gate.SKIPPED


# ===========================================================================
# derive_accuracy_suite
# ===========================================================================
def test_derive_from_snapshot():
    assert gate.derive_accuracy_suite(SNAP_PLAIN) == PLAIN
    assert gate.derive_accuracy_suite(SNAP_QDQF) == QDQF


def test_derive_from_session_snapshot():
    assert gate.derive_accuracy_suite(SESS_QDQF) == QDQF


# ===========================================================================
# parse_accuracy_list
# ===========================================================================
def test_parse_accuracy_list():
    text = (
        "QnnUnit_Accuracy_ClipPlainTest.\n"
        "  Case/Clip_f32  # GetParam() = 64-byte object <...>\n"
        "  Case/Clip_int32\n"
        "QnnUnit_Accuracy_ClipQDQFloatTest.\n"
        "  Case/Clip_U8_Rank4\n"
    )
    cases = gate.parse_accuracy_list(text)
    assert cases == [
        (PLAIN, "Case/Clip_f32"),
        (PLAIN, "Case/Clip_int32"),
        (QDQF, "Case/Clip_U8_Rank4"),
    ]


# ===========================================================================
# compute_run_set -- the decision table
# ===========================================================================
def test_version_match_all_pass_runs_nothing():
    acc = [(PLAIN, "Case/Clip_f32"), (PLAIN, "Case/Clip_int32")]
    snap = {
        (SNAP_PLAIN, "Case/Clip_f32"): gate.PASSED,
        (SNAP_PLAIN, "Case/Clip_int32"): gate.PASSED,
    }
    run, skip = gate.compute_run_set(acc, snap, version_match=True)
    assert run == []
    assert {d.name for d in skip} == {"Case/Clip_f32", "Case/Clip_int32"}


def test_version_match_drift_runs_only_that_case():
    acc = [(PLAIN, "Case/Clip_f32"), (PLAIN, "Case/Clip_int32")]
    snap = {
        (SNAP_PLAIN, "Case/Clip_f32"): gate.DRIFT,
        (SNAP_PLAIN, "Case/Clip_int32"): gate.PASSED,
    }
    run, skip = gate.compute_run_set(acc, snap, version_match=True)
    assert [d.full_name for d in run] == [f"{PLAIN}.Case/Clip_f32"]
    assert run[0].reasons == {gate.R_DRIFT}
    assert [d.name for d in skip] == ["Case/Clip_int32"]


def test_version_match_absent_runs():
    acc = [(PLAIN, "Case/Clip_f32")]
    snap = {(SNAP_PLAIN, "Case/Clip_f32"): gate.SKIPPED}
    run, skip = gate.compute_run_set(acc, snap, version_match=True)
    assert [d.name for d in run] == ["Case/Clip_f32"]
    assert run[0].reasons == {gate.R_ABSENT}


def test_version_match_unmapped_runs():
    # Accuracy case with no paired snapshot -> union safety -> run.
    acc = [(PLAIN, "Case/Clip_OnlyAccuracy")]
    snap = {}
    run, skip = gate.compute_run_set(acc, snap, version_match=True)
    assert [d.name for d in run] == ["Case/Clip_OnlyAccuracy"]
    assert run[0].reasons == {gate.R_UNMAPPED}


def test_version_mismatch_runs_all():
    acc = [(PLAIN, "Case/Clip_f32"), (PLAIN, "Case/Clip_int32")]
    snap = {
        (SNAP_PLAIN, "Case/Clip_f32"): gate.PASSED,
        (SNAP_PLAIN, "Case/Clip_int32"): gate.PASSED,
    }
    run, skip = gate.compute_run_set(acc, snap, version_match=False)
    assert len(run) == 2
    assert skip == []
    assert all(gate.R_VERSION in d.reasons for d in run)


def test_version_mismatch_union_keeps_drift_reason():
    # No short-circuit: a drifted case under version-mismatch carries BOTH
    # reasons, so the drift is not masked.
    acc = [(PLAIN, "Case/Clip_f32")]
    snap = {(SNAP_PLAIN, "Case/Clip_f32"): gate.DRIFT}
    run, _ = gate.compute_run_set(acc, snap, version_match=False)
    assert run[0].reasons == {gate.R_VERSION, gate.R_DRIFT}


# ===========================================================================
# QDQFloat dual-source: Group B via SessionSnapshot, Group C via Snapshot
# ===========================================================================
def test_qdqfloat_dual_source_routes_independently():
    acc = [
        (QDQF, "Case/Clip_U8_Rank4"),  # Group C -> op-builder snapshot
        (QDQF, "Case/Clip_U8_DefaultMinMax_Rank4"),  # Group B -> session snapshot
    ]
    snap = {
        (SNAP_QDQF, "Case/Clip_U8_Rank4"): gate.PASSED,
        (SESS_QDQF, "Case/Clip_U8_DefaultMinMax_Rank4"): gate.DRIFT,
    }
    run, skip = gate.compute_run_set(acc, snap, version_match=True)
    assert [d.name for d in run] == ["Case/Clip_U8_DefaultMinMax_Rank4"]
    assert [d.name for d in skip] == ["Case/Clip_U8_Rank4"]


def test_same_accuracy_case_two_sources_non_pass_wins():
    # If a case somehow appears from both snapshot sources, any non-PASS forces
    # a run (PASS is proof only if the other source also passed).
    acc = [(QDQF, "Case/Clip_Dup")]
    snap = {
        (SNAP_QDQF, "Case/Clip_Dup"): gate.PASSED,
        (SESS_QDQF, "Case/Clip_Dup"): gate.DRIFT,
    }
    run, skip = gate.compute_run_set(acc, snap, version_match=True)
    assert [d.name for d in run] == ["Case/Clip_Dup"]
    assert run[0].reasons == {gate.R_DRIFT}


# ===========================================================================
# build_gtest_filter
# ===========================================================================
def test_filter_empty_returns_non_matching_sentinel():
    f = gate.build_gtest_filter([])
    assert f == gate._EMPTY_FILTER_SENTINEL
    # Sentinel must not be a real accuracy prefix that gtest expands to all.
    assert "ZZZ" in f


def test_filter_joins_full_names():
    run = [
        gate.CaseDecision(PLAIN, "Case/Clip_f32", True, {gate.R_DRIFT}),
        gate.CaseDecision(QDQF, "Case/Clip_U8_Rank4", True, {gate.R_ABSENT}),
    ]
    assert gate.build_gtest_filter(run) == (f"{PLAIN}.Case/Clip_f32:{QDQF}.Case/Clip_U8_Rank4")


# ===========================================================================
# manifest / version loading
# ===========================================================================
# Canonical current tool versions used by the end-to-end tests. A match requires
# BOTH qairt and ort to equal the manifest.
def _cur(qairt="2.35.0", ort="1.20.0"):
    return gate.ToolVersions(qairt=qairt, ort=ort)


def _write_manifest(tmp_path, qairt="2.35.0", ort="1.20.0"):
    obj = {}
    if qairt is not None:
        obj["qairt_version"] = qairt
    if ort is not None:
        obj["ort_version"] = ort
    (tmp_path / "manifest.json").write_text(json.dumps(obj))


def test_load_manifest_versions(tmp_path):
    _write_manifest(tmp_path, qairt="2.35.0", ort="1.20.0")
    v = gate.load_manifest_versions(str(tmp_path))
    assert v.qairt == "2.35.0"
    assert v.ort == "1.20.0"


def test_load_manifest_absent(tmp_path):
    assert gate.load_manifest_versions(str(tmp_path)) == gate.ToolVersions()
    assert gate.load_manifest_versions(None) == gate.ToolVersions()


def test_load_manifest_malformed(tmp_path):
    (tmp_path / "manifest.json").write_text("not json{")
    assert gate.load_manifest_versions(str(tmp_path)) == gate.ToolVersions()


def test_load_manifest_partial_ort_missing(tmp_path):
    # Manifest with only qairt -> ort is None -> cannot match -> full run.
    _write_manifest(tmp_path, qairt="2.35.0", ort=None)
    v = gate.load_manifest_versions(str(tmp_path))
    assert v.qairt == "2.35.0"
    assert v.ort is None
    assert v.matches(_cur()) is False


def test_versions_match_requires_both_axes():
    m = gate.ToolVersions(qairt="2.35.0", ort="1.20.0")
    assert _cur("2.35.0", "1.20.0").matches(m) is True
    assert _cur("2.35.0", "9.9.9").matches(m) is False  # ort drift
    assert _cur("9.9.9", "1.20.0").matches(m) is False  # qairt drift
    assert gate.ToolVersions(qairt=None, ort="1.20.0").matches(m) is False


def test_detect_version_env_override(monkeypatch):
    monkeypatch.setenv("QNN_UT_QAIRT_VERSION", "9.9.9")
    monkeypatch.setenv("QNN_UT_ORT_VERSION", "8.8.8")
    assert gate.detect_qairt_version() == "9.9.9"
    assert gate.detect_ort_version() == "8.8.8"
    cur = gate.detect_current_versions()
    assert cur.qairt == "9.9.9"
    assert cur.ort == "8.8.8"


# ===========================================================================
# run_gate end-to-end (synthetic files)
# ===========================================================================
def test_run_gate_no_manifest_full_run(tmp_path):
    snap_json = _write_json(
        tmp_path,
        _json_report([_suite(SNAP_PLAIN, [_tc_passed("Case/Clip_f32")])]),
    )
    acc = [(PLAIN, "Case/Clip_f32")]
    # golden_root without manifest.json -> version_match False -> full run.
    res = gate.run_gate(snap_json, str(tmp_path), acc, current=_cur())
    assert res.version_match is False
    assert [d.name for d in res.run_set] == ["Case/Clip_f32"]
    assert res.run_set[0].reasons == {gate.R_VERSION}


def test_run_gate_ort_mismatch_full_run(tmp_path):
    # qairt matches but ort differs -> still a full run (ort is load-bearing).
    _write_manifest(tmp_path, qairt="2.35.0", ort="1.20.0")
    snap_json = _write_json(
        tmp_path,
        _json_report([_suite(SNAP_PLAIN, [_tc_passed("Case/Clip_f32")])]),
    )
    acc = [(PLAIN, "Case/Clip_f32")]
    res = gate.run_gate(snap_json, str(tmp_path), acc, current=_cur(ort="9.9.9"))
    assert res.version_match is False
    assert res.run_set[0].reasons == {gate.R_VERSION}


def test_run_gate_version_match_skips_passed(tmp_path):
    _write_manifest(tmp_path, qairt="2.35.0", ort="1.20.0")
    snap_json = _write_json(
        tmp_path,
        _json_report([_suite(SNAP_PLAIN, [_tc_passed("Case/Clip_f32")])]),
    )
    acc = [(PLAIN, "Case/Clip_f32")]
    res = gate.run_gate(snap_json, str(tmp_path), acc, current=_cur())
    assert res.version_match is True
    assert res.run_set == []
    assert [d.name for d in res.skip_set] == ["Case/Clip_f32"]


def test_run_gate_version_match_drift_and_summary(tmp_path):
    _write_manifest(tmp_path, qairt="2.35.0", ort="1.20.0")
    snap_json = _write_json(
        tmp_path,
        _json_report(
            [
                _suite(
                    SNAP_PLAIN,
                    [_tc_drift("Case/Clip_f32"), _tc_passed("Case/Clip_int32")],
                )
            ]
        ),
    )
    acc = [(PLAIN, "Case/Clip_f32"), (PLAIN, "Case/Clip_int32")]
    res = gate.run_gate(snap_json, str(tmp_path), acc, current=_cur())
    assert [d.name for d in res.run_set] == ["Case/Clip_f32"]
    assert res.drift_cases == [(SNAP_PLAIN, "Case/Clip_f32")]

    summary = gate.render_summary(res)
    # DRIFT block lists the drifted case, and only it.
    assert "[DRIFT] 1 snapshot case(s)" in summary
    assert "Case/Clip_f32" in summary
    assert "MATCH" in summary
    # VERSION block surfaces both axes.
    assert "qairt:" in summary
    assert "ort:" in summary


def test_summary_drift_section_excludes_absent_and_version(tmp_path):
    # A case that is absent (skipped) or forced by version must NOT show up in
    # the DRIFT block -- drift is snapshot FAILED only.
    _write_manifest(tmp_path, qairt="old", ort="old")
    snap_json = _write_json(
        tmp_path,
        _json_report([_suite(SNAP_PLAIN, [_tc_skipped("Case/Clip_f32")])]),
    )
    acc = [(PLAIN, "Case/Clip_f32")]
    res = gate.run_gate(snap_json, str(tmp_path), acc, current=_cur("new", "new"))
    summary = gate.render_summary(res)
    assert "[DRIFT] none" in summary
    assert "MISMATCH" in summary
