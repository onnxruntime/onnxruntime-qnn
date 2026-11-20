# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Check the two release paths without publishing."""

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
        pytest.skip("Node.js is required to evaluate the GitHub Actions expressions")
    result = subprocess.run([NODE, "--input-type=module"], input=script, text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


@pytest.mark.parametrize(("publish_mode", "maven_only"), [("non_maven", False), ("maven", True)])
def test_release_paths(publish_mode, maven_only):
    entry = workflow("qualcomm-internal-upload-artifactory-main.yml")
    upload = workflow("qualcomm-internal-upload-artifactory-from-github.yml")
    build = workflow("qualcomm-internal-build-artifacts.yml")
    dispatch_inputs = entry[True]["workflow_dispatch"]["inputs"]
    assert "maven_only" not in dispatch_inputs
    assert dispatch_inputs["publish_mode"]["type"] == "choice"
    assert dispatch_inputs["publish_mode"]["options"] == ["non_maven", "maven"]
    assert dispatch_inputs["publish_mode"]["default"] == "non_maven"
    expressions = {
        "maven_only": entry["jobs"]["publish-artifacts"]["with"]["maven_only"],
        "skip_android": entry["jobs"]["publish-artifacts"]["with"]["skip_android_builds"],
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
        inputs.maven_only = evaluate('maven_only');
        inputs.skip_android_builds = evaluate('skip_android');
        inputs.build_android_aarch64_aar = evaluate('build_aar');
        console.log(JSON.stringify(Object.fromEntries(Object.keys(expressions).map(key => [key, evaluate(key)]))));
        """
    )
    assert result == {
        "maven_only": maven_only,
        "skip_android": not maven_only,
        "build_aar": maven_only,
        "publish_maven": maven_only,
        "device_test": maven_only,
        "publish_wheels": not maven_only,
    }
