# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Check release path selection and source-SHA validation without publishing."""

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
        pytest.skip("Node.js is required to execute the GitHub Actions expressions and script")
    result = subprocess.run([NODE, "--input-type=module"], input=script, text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


@pytest.mark.parametrize("maven_only", [False, True])
def test_release_paths(maven_only):
    entry = workflow("qualcomm-internal-upload-artifactory-main.yml")
    upload = workflow("qualcomm-internal-upload-artifactory-from-github.yml")
    build = workflow("qualcomm-internal-build-artifacts.yml")
    expressions = {
        "skip_android": entry["jobs"]["publish-artifacts"]["with"]["skip_android_builds"],
        "build_aar": upload["jobs"]["build-artifacts"]["with"]["build_android_aarch64_aar"],
        "publish_maven": upload["jobs"]["upload-android-maven-workflow"]["if"],
        "device_test": build["jobs"]["test-android-aarch64"]["if"],
        "publish_wheels": upload["jobs"]["upload-wheel-workflow"]["if"],
    }
    result = run_node(
        f"""
        const expressions = {json.dumps(expressions)};
        const inputs = {{maven_only: {json.dumps(maven_only)}, build_android_aarch64: false}};
        const evaluate = key => new Function('inputs', 'return ' + expressions[key].slice(3, -2))(inputs);
        inputs.skip_android_builds = evaluate('skip_android');
        inputs.build_android_aarch64_aar = evaluate('build_aar');
        console.log(JSON.stringify(Object.fromEntries(Object.keys(expressions).map(key => [key, evaluate(key)]))));
        """
    )
    assert result == {
        "skip_android": not maven_only,
        "build_aar": maven_only,
        "publish_maven": maven_only,
        "device_test": maven_only,
        "publish_wheels": not maven_only,
    }


@pytest.mark.parametrize(
    "sha,exists", [("", True), ("main", True), ("a" * 39, True), ("g" * 40, True), ("a" * 40, False), ("a" * 40, True)]
)
def test_source_sha_validation(sha, exists):
    entry = workflow("qualcomm-internal-upload-artifactory-main.yml")
    script = next(
        step["with"]["script"]
        for step in entry["jobs"]["check-admin-permission"]["steps"]
        if step["name"] == "Validate source SHA"
    )
    result = run_node(
        f"""
        process.env.SOURCE_SHA = {json.dumps(sha)};
        let requested = null;
        const github = {{rest: {{repos: {{getCommit: async args => {{
            requested = args.ref;
            if (!{json.dumps(exists)}) throw new Error('Commit not found');
            return {{data: {{sha: args.ref}}}};
        }}}}}}}};
        const execute = new (Object.getPrototypeOf(async function() {{}}).constructor)(
            'github', 'context', 'core', {json.dumps(script)});
        try {{
            await execute(github, {{repo: {{owner: 'owner', repo: 'repo'}}}}, {{info: () => {{}}}});
            console.log(JSON.stringify({{success: true, requested}}));
        }} catch {{
            console.log(JSON.stringify({{success: false, requested}}));
        }}
        """
    )
    assert result["success"] == (sha == "a" * 40 and exists)
    assert result["requested"] == (sha if sha == "a" * 40 else None)


def test_source_sha_reaches_all_release_workers():
    pending = ["qualcomm-internal-upload-artifactory-main.yml"]
    visited = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        for job in workflow(name)["jobs"].values():
            if job.get("uses", "").startswith("./.github/workflows/"):
                assert job["with"]["source_sha"] == "${{ inputs.source_sha }}", name
                pending.append(Path(job["uses"]).name)
            for step in job.get("steps", []):
                if step.get("uses", "").startswith("actions/checkout@"):
                    assert step["with"]["ref"] == "${{ inputs.source_sha }}", name
