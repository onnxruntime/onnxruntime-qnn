# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

param(
    [Parameter(Mandatory=$true)]
    [string]$NuGetDirectory,
    [Parameter(Mandatory=$true)]
    [string]$ExpectedVersion,
    [Parameter(Mandatory=$true)]
    [string[]]$RuntimeIdentifiers,
    [Parameter(Mandatory=$true)]
    [string]$ModelPath,
    [Parameter(Mandatory=$true)]
    [string]$BackendDll
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding           = [System.Text.Encoding]::UTF8
$env:NO_COLOR             = "1"

# =============================================================================
# Step 1: Verify dotnet CLI is installed
# =============================================================================

Write-Host ""
Write-Host "Step 1: Checking required tools..." -ForegroundColor Cyan

# --- dotnet CLI ---
$dotnetCmd = Get-Command dotnet -ErrorAction SilentlyContinue
if (-not $dotnetCmd) {
    Write-Host "  ERROR: dotnet CLI not found in PATH." -ForegroundColor Red
    Write-Host "         Install the .NET SDK from https://dotnet.microsoft.com/download and ensure it is on PATH." -ForegroundColor Red
    Write-Host ""
    Write-Host "Step 1 FAILED: dotnet CLI is missing. Cannot proceed." -ForegroundColor Red
    exit 1
}

$dotnetVersion = (& dotnet --version 2>&1).ToString().Trim()
Write-Host "  dotnet : $($dotnetCmd.Source)" -ForegroundColor Green
Write-Host "           version $dotnetVersion" -ForegroundColor Green

Write-Host ""
Write-Host "Step 1 PASSED" -ForegroundColor Green

# =============================================================================
# Step 2: Create a temporary .NET console project
# =============================================================================

Write-Host ""
Write-Host "Step 2: Creating temporary .NET console project..." -ForegroundColor Cyan

$projectDir = Join-Path ([System.IO.Path]::GetTempPath()) "QnnEpNuGetTest_$(Get-Random)"
Write-Host "  Project directory: $projectDir"

try {
    & dotnet new console -n QnnEpNuGetTest -o $projectDir -f net8.0 --no-restore
    Write-Host ""
    Write-Host "Step 2 PASSED" -ForegroundColor Green

    # =============================================================================
    # Step 3: Add NuGet sources
    # =============================================================================

    Write-Host ""
    Write-Host "Step 3: Adding NuGet sources..." -ForegroundColor Cyan

    Push-Location $projectDir

    # Add local .nupkg folder as a source
    & dotnet nuget add source $NuGetDirectory --name QnnLocalSource
    Write-Host "  Added local source : $NuGetDirectory" -ForegroundColor Green

    # Add nuget.org as a source
    & dotnet nuget add source "https://api.nuget.org/v3/index.json" --name nuget
    Write-Host "  Added nuget.org source" -ForegroundColor Green

    Write-Host ""
    Write-Host "Step 3 PASSED" -ForegroundColor Green

    # =============================================================================
    # Step 4: Install packages
    # =============================================================================

    Write-Host ""
    Write-Host "Step 4: Installing packages..." -ForegroundColor Cyan

    & dotnet add package System.Numerics.Tensors --version 9.0.0
    Write-Host "  Installed System.Numerics.Tensors 9.0.0" -ForegroundColor Green

    # Pin Microsoft.ML.OnnxRuntime to the version used to compile onnxruntime-qnn.
    # Parse the ort_core tag from cmake/deps.txt so the pin stays in sync automatically.
    $depsFile = Join-Path $PSScriptRoot "../../../cmake/deps.txt"
    $ortLine  = Get-Content $depsFile | Where-Object { $_ -match '^ort_core;' }
    if ($ortLine -match '/tags/v([^/]+)\.zip') {
        $ortVersion = $Matches[1]
    } else {
        Write-Host "  ERROR: Could not parse ort_core version from cmake/deps.txt" -ForegroundColor Red
        exit 1
    }
    & dotnet add package Microsoft.ML.OnnxRuntime --version $ortVersion
    Write-Host "  Installed Microsoft.ML.OnnxRuntime $ortVersion" -ForegroundColor Green

    & dotnet add package Qualcomm.ML.OnnxRuntime.QNN `
        --source $NuGetDirectory `
        --version $ExpectedVersion
    Write-Host "  Installed Qualcomm.ML.OnnxRuntime.QNN $ExpectedVersion" -ForegroundColor Green

    Write-Host ""
    Write-Host "Step 4 PASSED" -ForegroundColor Green

    # =============================================================================
    # Step 5: Build for each Runtime Identifier and verify native DLL output
    # =============================================================================

    Write-Host ""
    Write-Host "Step 5: Building and verifying for Runtime Identifiers: $($RuntimeIdentifiers -join ', ')..." -ForegroundColor Cyan

    $buildFailures = 0

    foreach ($rid in $RuntimeIdentifiers) {
        Write-Host ""
        Write-Host "  --- RID: $rid ---" -ForegroundColor Yellow

        # Restore for this specific RID using -p:RuntimeIdentifier to bypass SDK RID graph
        # validation (allows non-standard RIDs like win-arm64ec that the CLI -r flag rejects)
        & dotnet restore -p:RuntimeIdentifier=$rid
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  BUILD FAIL: dotnet restore exited with code $LASTEXITCODE for RID $rid" -ForegroundColor Red
            $buildFailures++
            continue
        }

        # Build targeting this RID
        & dotnet build -p:RuntimeIdentifier=$rid --no-restore
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  BUILD FAIL: dotnet build exited with code $LASTEXITCODE for RID $rid" -ForegroundColor Red
            $buildFailures++
            continue
        }

        # Verify onnxruntime_providers_qnn.dll landed in the build output
        $outputDir = Join-Path $projectDir "bin/Debug/net8.0/$rid"
        $dll = Get-ChildItem -Path $outputDir -Filter "onnxruntime_providers_qnn.dll" -Recurse -ErrorAction SilentlyContinue |
            Select-Object -First 1

        if (-not $dll) {
            Write-Host "  DLL FAIL: onnxruntime_providers_qnn.dll not found in $outputDir" -ForegroundColor Red
            $buildFailures++
            continue
        }
        Write-Host "  BUILD PASS: onnxruntime_providers_qnn.dll found at $($dll.FullName)" -ForegroundColor Green

        # --- Step 6: Write a minimal C# smoke test and run inference ---
        $programCs = Join-Path $projectDir "Program.cs"
        Set-Content -Path $programCs -Encoding UTF8 -Value @"
using Microsoft.ML.OnnxRuntime;

var modelPath  = args[0];
var backendDll = args[1];

var options = new SessionOptions();
options.AppendExecutionProvider("QNN", new Dictionary<string, string>
{
    ["backend_path"] = backendDll
});

using var session = new InferenceSession(modelPath, options);
Console.WriteLine(`$"QNN EP smoke test PASSED — session created with backend: {backendDll}");
"@

        & dotnet run -p:RuntimeIdentifier=$rid --no-build -- "$ModelPath" "$BackendDll"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  SMOKE FAIL: inference exited with code $LASTEXITCODE for RID $rid" -ForegroundColor Red
            $buildFailures++
        } else {
            Write-Host "  SMOKE PASS: QNN EP loaded and session created for RID $rid" -ForegroundColor Green
        }
    }

    if ($buildFailures -gt 0) {
        Write-Host ""
        Write-Host "Step 5 FAILED: $buildFailures RID(s) did not produce the expected output." -ForegroundColor Red
        exit 1
    }

    Write-Host ""
    Write-Host "Step 5 PASSED" -ForegroundColor Green

} finally {
    Pop-Location -ErrorAction SilentlyContinue
    if (Test-Path $projectDir) {
        Remove-Item -Path $projectDir -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host ""
        Write-Host "Cleaned up temporary project directory." -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host "All steps PASSED" -ForegroundColor Green
