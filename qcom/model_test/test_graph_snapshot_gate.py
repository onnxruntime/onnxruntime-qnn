import json
from pathlib import Path

from graph_snapshot_gate import check_snapshot_gate


def write_manifest(root: Path, **overrides: object) -> None:
    manifest = {
        "snapshot_schema_version": 1,
        "modelzoo_platform": "windows-arm64",
        "htp_arch": "v73",
        "ort_version": "1.2.3",
        "qairt_version": "2.50.40",
    }
    manifest.update(overrides)
    (root / "manifest.json").write_text(json.dumps(manifest))


def check(current: Path, golden: Path):
    return check_snapshot_gate(
        current,
        golden,
        Path("suite") / "model",
        "windows-arm64",
        "v73",
        "1.2.3",
        "2.50.40",
    )


def test_matching_snapshot_skips_real_execution(tmp_path: Path) -> None:
    current = tmp_path / "current"
    golden = tmp_path / "golden"
    current.mkdir()
    (golden / "suite" / "model").mkdir(parents=True)
    write_manifest(golden)
    payload = b'{"graph":{"nodes":{}}}\n'
    (current / "graph.json").write_bytes(payload)
    (golden / "suite" / "model" / "graph.json").write_bytes(payload)

    assert check(current, golden).skip_real_execution


def test_manifest_or_snapshot_difference_falls_back(tmp_path: Path) -> None:
    current = tmp_path / "current"
    golden = tmp_path / "golden"
    current.mkdir()
    (golden / "suite" / "model").mkdir(parents=True)
    write_manifest(golden, htp_arch="v81")
    (current / "graph.json").write_text("{}")
    (golden / "suite" / "model" / "graph.json").write_text("{}")
    assert not check(current, golden).skip_real_execution

    write_manifest(golden)
    (golden / "suite" / "model" / "graph.json").write_text('{"changed":true}')
    assert not check(current, golden).skip_real_execution
