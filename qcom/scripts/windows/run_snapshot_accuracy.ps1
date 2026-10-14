# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

param(
    [Parameter(HelpMessage = "The build config")]
    [ValidateSet("Debug", "Release", "RelWithDebInfo")]
    [string]$Config = "Release",
    [Parameter(HelpMessage = "Path to onnx/models models")]
    $OnnxModelsRoot = $null
)

# Internal-symbol validation runs normal CTest first, then gives the snapshot
# and accuracy tiers their required two-pass routing semantics. The golden store
# is intentionally consumed only through QNN_UT_SNAPSHOT_GOLDEN_DIR; a separate
# producer flow provisions it.
$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path -Path "$(Split-Path -Parent $MyInvocation.MyCommand.Definition)").Path
$RepoRoot = (Resolve-Path -Path (Join-Path $RootDir "..\..\.."))
$BaselineRunner = Join-Path $RootDir "run_tests.ps1"
$GateScript = Join-Path $RepoRoot "qcom\scripts\linux\accuracy_gate.py"

if (-not (Test-Path $BaselineRunner)) {
    throw "$BaselineRunner not found"
}
if (-not (Test-Path $GateScript)) {
    throw "$GateScript not found"
}
if ($env:QNN_UT_SNAPSHOT_GOLDEN_UPDATE -eq "1") {
    throw "QNN_UT_SNAPSHOT_GOLDEN_UPDATE=1 is not allowed in validation. Use the dedicated golden producer."
}

& $BaselineRunner -Config $Config -OnnxModelsRoot $OnnxModelsRoot
if (-not $?) {
    throw "Baseline Windows test runner failed."
}

$ProviderTestExe = Join-Path $RootDir "onnxruntime_provider_test.exe"
if (-not (Test-Path $ProviderTestExe)) {
    $ProviderTestExe = Join-Path (Join-Path $RootDir $Config) "onnxruntime_provider_test.exe"
}
if (-not (Test-Path $ProviderTestExe)) {
    throw "onnxruntime_provider_test.exe not found under $RootDir"
}

# Windows x86 uses the QNN CPU attach point for the HTP emulator. The baseline
# runner also sets this, but keep the direct invocations self-contained.
$env:ORT_QNN_ENABLE_CPU_BACKEND = "1"

$ResultsDir = Join-Path $RootDir "snapshot_accuracy_results"
$SnapshotJson = Join-Path $ResultsDir "snapshot_results.json"
$AccuracyJson = Join-Path $ResultsDir "accuracy_results.json"
$AccuracyList = Join-Path $ResultsDir "accuracy_tests.txt"
$AccuracyFilterFile = Join-Path $ResultsDir "accuracy_filter.txt"
$GateSummary = Join-Path $ResultsDir "accuracy_gate_summary.txt"

New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
Remove-Item -LiteralPath $SnapshotJson, $AccuracyJson, $AccuracyList, $AccuracyFilterFile, $GateSummary `
    -Force -ErrorAction SilentlyContinue

$GoldenRoot = $env:QNN_UT_SNAPSHOT_GOLDEN_DIR
if ($GoldenRoot) {
    Write-Host "Snapshot golden store: $GoldenRoot"
}
else {
    Write-Warning "QNN_UT_SNAPSHOT_GOLDEN_DIR is unset; snapshots will route all accuracy tests."
}

Write-Host "--=-=-=- Running QNN snapshot tests -=--=-=-"
& $ProviderTestExe "--gtest_filter=QnnSnapshot_*" "--gtest_output=json:$SnapshotJson"
$SnapshotExit = $LASTEXITCODE
if ($SnapshotExit -ne 0) {
    Write-Warning "Snapshot tests exited $SnapshotExit. The accuracy gate will treat affected or unreported cases as unverified."
}

$AccuracyTests = & $ProviderTestExe "--gtest_list_tests" "--gtest_filter=QnnAcc_*"
$AccuracyListExit = $LASTEXITCODE
$AccuracyTests | Set-Content -LiteralPath $AccuracyList -Encoding ascii
if ($AccuracyListExit -ne 0) {
    throw "Failed to list QnnAcc_* tests (exit $AccuracyListExit)."
}

Write-Host "--=-=-=- Routing QNN accuracy tests from snapshot results -=--=-=-"
& python $GateScript `
    "--snapshot-json=$SnapshotJson" `
    "--golden-root=$GoldenRoot" `
    "--accuracy-list-file=$AccuracyList" `
    "--emit-filter-file=$AccuracyFilterFile" `
    "--emit-summary-file=$GateSummary"
if ($LASTEXITCODE -ne 0) {
    throw "accuracy_gate.py failed."
}

$AccuracyFilter = (Get-Content -LiteralPath $AccuracyFilterFile -Raw).Trim()
if (-not $AccuracyFilter) {
    throw "accuracy_gate.py did not emit an accuracy filter."
}

Write-Host "--=-=-=- Running routed QNN accuracy tests -=--=-=-"
Write-Host "Accuracy filter: $AccuracyFilter"
& $ProviderTestExe "--gtest_filter=$AccuracyFilter" "--gtest_output=json:$AccuracyJson"
if ($LASTEXITCODE -ne 0) {
    throw "Routed QnnAcc_* tests failed. See $AccuracyJson"
}
