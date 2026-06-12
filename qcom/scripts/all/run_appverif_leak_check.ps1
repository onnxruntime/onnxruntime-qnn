# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

<#
.SYNOPSIS
    Run App Verifier (Leak layer) memory-leak detection on onnxruntime_provider_test.

.DESCRIPTION
    Implements the flow from the Confluence tutorial
    "Enable App-Verifier on QA WoS device in Admin mode":

        Setup (enable Leak, exclude QAIRT backend DLLs)
          -> Run onnxruntime_provider_test
          -> Export App Verifier XML log
          -> Teardown (always, even on failure)
          -> Parse the XML and gate on the result

    Requires an elevated (administrator) PowerShell: 'appverif -enable' writes
    the Image File Execution Options registry key.

    The script's exit code is the parser's exit code, so it can gate a CI job:
        0  -> no leak (PASS / green)
        1  -> leak detected, or log missing/parse error (FAIL / red)

.PARAMETER BuildDir
    Directory containing onnxruntime_provider_test.exe and its PDBs,
    e.g. ...\build\windows-arm64\RelWithDebInfo.

.PARAMETER ParserScript
    Path to appverif_leak_to_report.py. Defaults to the .py next to this script.

.PARAMETER GTestFilter
    Value passed to --gtest_filter. Defaults to "*".

.PARAMETER ExcludeDlls
    QAIRT backend DLLs excluded from leak detection (vendor-side leaks).

.PARAMETER LogPath
    Where to write the exported App Verifier XML. Defaults to <BuildDir>\appverif_result.xml.

.EXAMPLE
    .\run_appverif_leak_check.ps1 -BuildDir C:\artifacts\RelWithDebInfo
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$BuildDir,

    [string]$ParserScript = (Join-Path $PSScriptRoot "appverif_leak_to_report.py"),

    [string]$GTestFilter = "*",

    # OpenCL.dll: GPU-backend leaks are owned by the Adreno OpenCL runtime
    # (verified via WinDBG -- the verifier-stop owner base maps to OpenCL.dll,
    # not QnnGpu.dll), so excluding QnnGpu alone does not suppress them.
    [string[]]$ExcludeDlls = @("QnnCpu.dll", "QnnHtp.dll", "QnnGpu.dll", "QnnIr.dll", "QnnSaver.dll", "OpenCL.dll"),

    [string]$LogPath,

    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

$Target = "onnxruntime_provider_test.exe"

function Test-IsAdmin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($id)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# --- Pre-flight checks -------------------------------------------------------
if (-not (Test-IsAdmin)) {
    throw "This script must run in an elevated (administrator) PowerShell; 'appverif -enable' requires admin."
}

if (-not (Get-Command appverif -ErrorAction SilentlyContinue)) {
    throw "appverif not found on PATH. Install Application Verifier (Windows SDK) first."
}

$exePath = Join-Path $BuildDir $Target
if (-not (Test-Path $exePath)) {
    throw "$Target not found in $BuildDir"
}

if (-not (Test-Path $ParserScript)) {
    throw "Parser script not found: $ParserScript"
}

if (-not $LogPath) {
    $LogPath = Join-Path $BuildDir "appverif_result.xml"
}

$excludeArg = ($ExcludeDlls -join ",")

Write-Host "=== App Verifier leak check ==="
Write-Host "  Target      : $Target"
Write-Host "  BuildDir    : $BuildDir"
Write-Host "  ExcludeDlls : $excludeArg"
Write-Host "  GTestFilter : $GTestFilter"
Write-Host "  LogPath     : $LogPath"
Write-Host "  Parser      : $ParserScript"
Write-Host ""

Push-Location $BuildDir
try {
    # --- Setup ---------------------------------------------------------------
    # Clear any stale registration first, then enable the Leak layer.
    Write-Host "[1/4] Enabling App Verifier (Leak)..."
    & appverif -disable * -for $Target | Out-Null
    & appverif -enable Leak -for $Target -with "Leak.ExcludeDlls=$excludeArg"
    if ($LASTEXITCODE -ne 0) {
        throw "appverif -enable failed with exit code $LASTEXITCODE"
    }

    # --- Run -----------------------------------------------------------------
    # NOTE: On a detected leak, App Verifier writes the XML and then terminates
    # the process with 0xC0000409 (-1073740767). That is EXPECTED -- we must not
    # let it abort the script, so the test exit code is intentionally ignored
    # here; the verdict comes from parsing the exported log below.
    Write-Host "[2/4] Running $Target --gtest_filter=`"$GTestFilter`" ..."
    & ".\$Target" --gtest_filter="$GTestFilter"
    $testExit = $LASTEXITCODE
    Write-Host "      test process exit code: $testExit"

    # --- Export --------------------------------------------------------------
    Write-Host "[3/4] Exporting App Verifier log to $LogPath ..."
    & appverif -export log -for $Target -with "to=$LogPath"
}
finally {
    # --- Teardown (ALWAYS) ---------------------------------------------------
    Write-Host "[teardown] Disabling App Verifier for $Target ..."
    & appverif -disable * -for $Target | Out-Null
}

# --- Parse & gate ------------------------------------------------------------
Write-Host "[4/4] Parsing $LogPath ..."
if (-not (Test-Path $LogPath)) {
    Write-Error "App Verifier log was not produced: $LogPath"
    exit 1
}

& $Python $ParserScript $LogPath
$parseExit = $LASTEXITCODE

Write-Host ""
if ($parseExit -eq 0) {
    Write-Host "RESULT: PASS (no leak detected)"
} else {
    Write-Host "RESULT: FAIL (leak detected or log error)"
}
exit $parseExit
