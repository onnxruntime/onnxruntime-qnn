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

$singleTest = ($lines[0..$firstBlockEnd] -join "`n").Replace(
    'onnxruntime_provider_test_0', 'onnxruntime_provider_test').Replace(
    '--gtest_filter=-QnnAcc_*', '--gtest_filter=-QnnAcc_*:QnnHTPBackendTests.ReduceL2*').Replace(
    '.shard0.results.xml', '.results.xml')
$singleTest = $singleTest.Replace(' ENVIRONMENT "GTEST_TOTAL_SHARDS=4;GTEST_SHARD_INDEX=0" RUN_SERIAL "TRUE"', '')
if ($singleTest -match 'GTEST_.*SHARD|onnxruntime_provider_test_[0-3]' -or
    $singleTest -notmatch [regex]::Escape('--gtest_filter=-QnnAcc_*:QnnHTPBackendTests.ReduceL2*')) {
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
    filter = '-QnnAcc_*:QnnHTPBackendTests.ReduceL2*'
    expected_active_cases = 3331
    excluded_for_diagnosis = @(
        'QnnHTPBackendTests.ReduceL2S8Opset13',
        'QnnHTPBackendTests.ReduceL2U8Opset13_NoKeepDims',
        'QnnHTPBackendTests.ReduceL2U16Opset18',
        'QnnHTPBackendTests.ReduceL2U8Opset18'
    )
    binaries = @($hashes)
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $binaryRoot 'reduce-l2-diagnostic.json') -Encoding utf8
Write-Host 'Diagnostic only: one registered CTest process; four HTP ReduceL2 cases excluded; original binaries reused.'
