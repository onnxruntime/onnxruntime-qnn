# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

param(
    [Parameter(Mandatory=$true)]
    [string]$ZipDirectory,
    [Parameter(Mandatory=$true)]
    [string]$TestPackageZip,
    [Parameter(Mandatory=$true)]
    [string]$ModelPath,
    [Parameter(Mandatory=$true)]
    [string]$ZipArch,
    [Parameter(Mandatory=$true)]
    [string]$TestBinArch,
    [Parameter(Mandatory=$false)]
    [string]$BackendDll = "QnnHtp.dll"
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding           = [System.Text.Encoding]::UTF8
$env:NO_COLOR             = "1"

# --- Extract the release zip ---
$releaseZip = Get-ChildItem -Path $ZipDirectory -Filter "*${ZipArch}.zip" | Select-Object -First 1
if (-not $releaseZip) {
    Write-Host ""
    Write-Host "ERROR: No ${ZipArch} zip found in $ZipDirectory" -ForegroundColor Red
    exit 1
}
$releaseExtractDir = Join-Path $ZipDirectory "$($releaseZip.BaseName)_extracted"
Write-Host ""
Write-Host "Extracting release zip: $($releaseZip.Name)" -ForegroundColor Yellow
Expand-Archive -Path $releaseZip.FullName -DestinationPath $releaseExtractDir -Force
Get-ChildItem -Path $releaseExtractDir -Recurse | Unblock-File

# --- Extract test_package.zip ---
$testPkgsDir = Join-Path (Split-Path $TestPackageZip) "test_package"
Write-Host "Extracting test_package.zip" -ForegroundColor Yellow
Expand-Archive -Path $TestPackageZip -DestinationPath $testPkgsDir -Force

# --- Locate the test binaries folder in test_package ---
$testBinDir = Get-ChildItem -Path $testPkgsDir -Directory |
    Where-Object { $_.Name -eq $TestBinArch } |
    Select-Object -First 1

if (-not $testBinDir) {
    Write-Host "ERROR: No ${TestBinArch} folder found in $testPkgsDir" -ForegroundColor Red
    Write-Host "Contents of ${testPkgsDir}:"
    Get-ChildItem -Path $testPkgsDir | ForEach-Object { Write-Host "  $($_.Name)" }
    exit 1
}

# --- Find onnxruntime_perf_test.exe in the test binaries ---
$perfTest = Get-ChildItem -Path $testBinDir.FullName -Recurse -Filter "onnxruntime_perf_test.exe" |
    Select-Object -First 1

if (-not $perfTest) {
    Write-Host "ERROR: onnxruntime_perf_test.exe not found in $($testBinDir.FullName)" -ForegroundColor Red
    exit 1
}

# --- Copy onnxruntime_perf_test.exe to the extracted release folder ---
Copy-Item -Path $perfTest.FullName -Destination $releaseExtractDir -Force
$copiedExe = Join-Path $releaseExtractDir "onnxruntime_perf_test.exe"
Unblock-File -Path $copiedExe
Write-Host "Copied onnxruntime_perf_test.exe to release test directory" -ForegroundColor Cyan

# --- Copy onnxruntime.dll and onnxruntime_providers_shared.dll from test_package ---
foreach ($dll in @("onnxruntime.dll", "onnxruntime_providers_shared.dll")) {
    $src = Join-Path $testBinDir.FullName $dll
    Copy-Item -Path $src -Destination $releaseExtractDir -Force
    Unblock-File -Path (Join-Path $releaseExtractDir $dll)
    Write-Host "Copied $dll to release test directory" -ForegroundColor Cyan
}

# --- Run the perf test smoke test ---
Write-Host "Running smoke test" -ForegroundColor Cyan
Write-Host ""

Push-Location $releaseExtractDir
try {
    & ".\onnxruntime_perf_test.exe" `
        -I `
        --plugin_ep_libs "QNNExecutionProvider|onnxruntime_providers_qnn.dll" `
        --plugin_eps QNNExecutionProvider `
        -m times `
        -r 1 `
        -p burst `
        -i "backend_path|${BackendDll}" `
        "$ModelPath"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Smoke test FAILED with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
} finally {
    Pop-Location
}
Write-Host "Smoke test PASSED" -ForegroundColor Green
