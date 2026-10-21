import json
import subprocess
import sys
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "publish_modelzoo_snapshots.py"


def make_wheel(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("onnxruntime_qnn-1.2.3.dist-info/METADATA", "Metadata-Version: 2.1\nVersion: 1.2.3\n")
        archive.writestr("onnxruntime_qnn/build_and_package_info.py", "qnn_version = '2.50.40'\n")


def test_dry_run_writes_versioned_manifest_and_archive(tmp_path: Path) -> None:
    snapshots = tmp_path / "snapshots"
    graph = snapshots / "suite" / "model" / "graph.json"
    graph.parent.mkdir(parents=True)
    graph.write_text('{"qnn_json_graph_schema_version": 1}\n')
    wheel = tmp_path / "package.whl"
    make_wheel(wheel)
    output = tmp_path / "output"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--snapshot-dir",
            str(snapshots),
            "--wheel",
            str(wheel),
            "--modelzoo-platform",
            "windows-arm64",
            "--htp-arch",
            "v73",
            "--output-dir",
            str(output),
            "--git-sha",
            "0123456789abcdef",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout)["mode"] == "dry-run"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["htp_arch"] == "v73"
    assert manifest["modelzoo_platform"] == "windows-arm64"
    assert manifest["ort_version"] == "1.2.3"
    assert manifest["qairt_version"] == "2.50.40"
    with zipfile.ZipFile(output / "snapshots.zip") as archive:
        assert archive.namelist() == ["manifest.json", "suite/model/graph.json"]
