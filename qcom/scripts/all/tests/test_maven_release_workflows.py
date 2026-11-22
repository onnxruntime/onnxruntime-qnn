# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Check release modes and existing caller defaults without publishing."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[4] / ".github" / "workflows"
NODE = shutil.which("node")


def workflow(name):
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def run_node(script):
    if NODE is None:
        pytest.fail("Node.js is required to evaluate the GitHub Actions expressions")
    result = subprocess.run([NODE, "--input-type=module"], input=script, text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


@pytest.mark.parametrize("publish_mode", ["non_maven", "maven", "all"])
def test_release_paths(publish_mode):
    entry = workflow("qualcomm-internal-upload-artifactory-main.yml")
    upload = workflow("qualcomm-internal-upload-artifactory-from-github.yml")
    build = workflow("qualcomm-internal-build-artifacts.yml")
    assert upload["jobs"]["upload-android-maven-workflow"]["needs"] == ["build-artifacts"]
    dispatch_inputs = entry[True]["workflow_dispatch"]["inputs"]
    assert dispatch_inputs["publish_mode"]["type"] == "choice"
    assert dispatch_inputs["publish_mode"]["options"] == ["non_maven", "maven"]
    assert dispatch_inputs["publish_mode"]["default"] == "non_maven"
    assert upload[True]["workflow_call"]["inputs"]["publish_mode"]["default"] == "all"
    assert build[True]["workflow_call"]["inputs"]["publish_mode"]["default"] == "all"
    expressions = {
        "upload_mode": entry["jobs"]["publish-artifacts"]["with"]["publish_mode"],
        "build_mode": upload["jobs"]["build-artifacts"]["with"]["publish_mode"],
        "build_aar": upload["jobs"]["build-artifacts"]["with"]["build_android_aarch64_aar"],
        "publish_maven": upload["jobs"]["upload-android-maven-workflow"]["if"],
        "device_test": build["jobs"]["test-android-aarch64"]["if"],
        "publish_wheels": upload["jobs"]["upload-wheel-workflow"]["if"],
    }
    result = run_node(
        f"""
        const expressions = {json.dumps(expressions)};
        const inputs = {{publish_mode: {json.dumps(publish_mode)}, build_android_aarch64: false}};
        const evaluate = key => new Function('inputs', 'return ' + expressions[key].slice(3, -2))(inputs);
        inputs.publish_mode = evaluate('upload_mode');
        inputs.publish_mode = evaluate('build_mode');
        inputs.build_android_aarch64_aar = evaluate('build_aar');
        console.log(JSON.stringify(Object.fromEntries(Object.keys(expressions).map(key => [key, evaluate(key)]))));
        """
    )
    android = publish_mode != "non_maven"
    assert result == {
        "upload_mode": publish_mode,
        "build_mode": publish_mode,
        "build_aar": android,
        "publish_maven": android,
        "device_test": android,
        "publish_wheels": publish_mode != "maven",
    }
