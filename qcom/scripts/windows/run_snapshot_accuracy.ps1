# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

param(
    [Parameter(HelpMessage = "The build config")]
    [ValidateSet("Debug", "Release", "RelWithDebInfo")]
    [string]$Config = "Release",
    [Parameter(HelpMessage = "Path to onnx/models models")]
    $OnnxModelsRoot = $null
)

# Windows owns the baseline CTest setup; the shared runner owns the
# snapshot-to-accuracy lifecycle used by every platform and golden producer.
$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path -Path "$(Split-Path -Parent $MyInvocation.MyCommand.Definition)").Path
$RepoRoot = (Resolve-Path -Path (Join-Path $RootDir "..\..\.."))
$BaselineRunner = Join-Path $RootDir "run_tests.ps1"
$SharedRunner = Join-Path $RepoRoot "qcom\scripts\all\run_snapshot_accuracy.sh"

if (-not (Test-Path -LiteralPath $BaselineRunner)) {
    throw "$BaselineRunner not found"
}
if (-not (Test-Path -LiteralPath $SharedRunner)) {
    throw "$SharedRunner not found"
}
if ($env:QNN_UT_SNAPSHOT_GOLDEN_UPDATE -eq "1") {
    throw "QNN_UT_SNAPSHOT_GOLDEN_UPDATE=1 is not allowed in validation. Use the dedicated golden producer."
}

& $BaselineRunner -Config $Config -OnnxModelsRoot $OnnxModelsRoot
if (-not $?) {
    throw "Baseline Windows test runner failed."
}

$GitBash = Join-Path $env:ProgramW6432 "Git\bin\bash.exe"
if (-not (Test-Path -LiteralPath $GitBash)) {
    throw "Git Bash was not found under $GitBash"
}

function ConvertTo-GitBashPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $ConvertedPath = & $GitBash -c 'cygpath -u "$1"' _ $Path
    if ($LASTEXITCODE -ne 0 -or -not $ConvertedPath) {
        throw "Failed to convert path for Git Bash: $Path"
    }
    $ConvertedPath.Trim()
}

$RootDirBash = ConvertTo-GitBashPath -Path $RootDir
$SharedRunnerBash = ConvertTo-GitBashPath -Path $SharedRunner

# The shared runner invokes native Python and QNN test binaries from Git Bash.
$env:PYTHON_BIN = "python.exe"
$env:ORT_QNN_ENABLE_CPU_BACKEND = "1"

& $GitBash $SharedRunnerBash "--build-dir=$RootDirBash"
if ($LASTEXITCODE -ne 0) {
    throw "Shared snapshot/accuracy runner failed (exit $LASTEXITCODE)."
}
