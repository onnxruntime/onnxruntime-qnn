# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

param(
    [Parameter(Mandatory = $true)][string]$ArtifactRoot,
    [Parameter(Mandatory = $true)][string]$SourceRunId
)

$ErrorActionPreference = 'Stop'
$testFile = Join-Path $ArtifactRoot 'CTestTestfile.cmake'
$lines = Get-Content -LiteralPath $testFile
$firstBlockEnd = [array]::IndexOf($lines, 'endif()')
if ($firstBlockEnd -lt 0 -or ($lines -join "`n") -notmatch 'onnxruntime_provider_test_3') {
    throw 'Expected the original four-shard, multi-config provider CTest metadata.'
}

$selection = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'single_process_diagnostic_selection.json') -Raw | ConvertFrom-Json
if ($selection.source_run_id -ne $SourceRunId -or
    $selection.expected_active_cases -ne $selection.original_active_cases - @($selection.excluded_for_diagnosis).Count) {
    throw 'Diagnostic selection does not match the source CI inventory.'
}
$testFilter = $selection.filter
$singleTest = ($lines[0..$firstBlockEnd] -join "`n").Replace(
    'onnxruntime_provider_test_0', 'onnxruntime_provider_test').Replace(
    '--gtest_filter=-QnnAcc_*', '--gtest_filter=' + $testFilter).Replace(
    '.shard0.results.xml', '.results.xml')
$singleTest = $singleTest.Replace(' ENVIRONMENT "GTEST_TOTAL_SHARDS=4;GTEST_SHARD_INDEX=0" RUN_SERIAL "TRUE"', '')
if ($singleTest -match 'GTEST_.*SHARD|onnxruntime_provider_test_[0-3]' -or
    $singleTest -notmatch [regex]::Escape('--gtest_filter=' + $testFilter)) {
    throw 'Single-process CTest conversion failed validation.'
}
Set-Content -LiteralPath $testFile -Value $singleTest -Encoding ascii

$binaryRoot = Join-Path $ArtifactRoot 'Release'
$hashes = @('onnxruntime_provider_test.exe', 'onnxruntime_providers_qnn.dll', 'onnxruntime.dll') | ForEach-Object {
    $hash = Get-FileHash -LiteralPath (Join-Path $binaryRoot $_) -Algorithm SHA256
    [ordered]@{ file = $_; sha256 = $hash.Hash }
}
$manifest = [ordered]@{
    source_run_id = $SourceRunId
    processes = 1
    filter = $testFilter
    expected_active_cases = $selection.expected_active_cases
    excluded_for_diagnosis = @($selection.excluded_for_diagnosis)
    binaries = @($hashes)
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $binaryRoot 'single-process-diagnostic.json') -Encoding utf8
Write-Host "Diagnostic only: one registered CTest process; $(@($selection.excluded_for_diagnosis).Count) HTP cases excluded, including observed single-process failures; original binaries reused."
