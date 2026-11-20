# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, HelpMessage = "Windows x86_64 internal-symbol build directory")]
    [string]$BuildDir,
    [ValidateSet("Debug", "Release", "RelWithDebInfo")]
    [string]$Config = "RelWithDebInfo",
    [string]$TestFilter = "QnnCPUBackendTests.*:QnnIRBackendTests.*:QnnUnit_*.*:QnnFrameworkOpTraceUnit.*",
    [string]$OpenCppCoveragePath = "",
    [switch]$NoInstallCoverageTools
)

$ErrorActionPreference = "Stop"

function Resolve-OpenCppCoverage {
    param(
        [string]$RequestedPath,
        [bool]$AllowInstall
    )

    $candidates = @($RequestedPath, $env:OPENCPPCOVERAGE_EXE, "${env:ProgramFiles}\OpenCppCoverage\OpenCppCoverage.exe") |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) }

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    if (-not $AllowInstall) {
        throw "OpenCppCoverage was not found. Install OpenCppCoverage.OpenCppCoverage, set OPENCPPCOVERAGE_EXE, or pass -OpenCppCoveragePath."
    }

    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if ($null -eq $winget) {
        throw "OpenCppCoverage was not found and winget.exe is unavailable. Install OpenCppCoverage.OpenCppCoverage, set OPENCPPCOVERAGE_EXE, or pass -OpenCppCoveragePath."
    }

    Write-Host "OpenCppCoverage not found. Installing OpenCppCoverage.OpenCppCoverage with winget..."
    & $winget.Source install --id OpenCppCoverage.OpenCppCoverage --exact --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "winget failed to install OpenCppCoverage.OpenCppCoverage (exit code $LASTEXITCODE)."
    }

    $defaultPath = "${env:ProgramFiles}\OpenCppCoverage\OpenCppCoverage.exe"
    if (-not (Test-Path -LiteralPath $defaultPath -PathType Leaf)) {
        throw "OpenCppCoverage installation completed but '$defaultPath' was not found. Set OPENCPPCOVERAGE_EXE or pass -OpenCppCoveragePath."
    }
    return (Resolve-Path -LiteralPath $defaultPath).Path
}

$resolvedBuildDir = (Resolve-Path -LiteralPath $BuildDir).Path
$binDir = Join-Path $resolvedBuildDir $Config
$providerTestExe = Join-Path $binDir "onnxruntime_provider_test.exe"
$providerDll = Join-Path $binDir "onnxruntime_providers_qnn.dll"
$providerPdb = Join-Path $binDir "onnxruntime_providers_qnn.pdb"

foreach ($requiredPath in @($providerTestExe, $providerDll, $providerPdb)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required coverage input was not found: $requiredPath"
    }
}

$openCppCoverageExe = Resolve-OpenCppCoverage -RequestedPath $OpenCppCoveragePath -AllowInstall (-not $NoInstallCoverageTools)
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..\..")).Path
$qnnSources = Join-Path $repoRoot "onnxruntime\core\providers\qnn"
$coverageDir = Join-Path $binDir "coverage"

# This directory is owned exclusively by this script. Do not clean build artifacts.
if (Test-Path -LiteralPath $coverageDir) {
    Remove-Item -LiteralPath $coverageDir -Recurse -Force
}
New-Item -ItemType Directory -Path $coverageDir | Out-Null

$coverageXml = Join-Path $coverageDir "coverage.xml"
$coverageHtml = Join-Path $coverageDir "html"
$stdoutLog = Join-Path $coverageDir "run.stdout.log"
$stderrLog = Join-Path $coverageDir "run.stderr.log"

# QNN CPU is opt-in on Windows x86_64 and is the only backend covered by this local pipeline.
$env:ORT_QNN_ENABLE_CPU_BACKEND = "1"

$arguments = @(
    "--optimized_build",
    "--working_dir", $binDir,
    "--modules", $providerDll,
    "--sources", $qnnSources,
    "--export_type", "cobertura:$coverageXml",
    "--export_type", "html:$coverageHtml",
    "--",
    $providerTestExe,
    "--gtest_filter=$TestFilter",
    "--gtest_color=no"
)

Write-Host "Collecting QNN CPU coverage with filter: $TestFilter"
# OpenCppCoverage emits informational messages on stderr. PowerShell otherwise
# promotes those messages to errors when ErrorActionPreference is Stop.
$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $openCppCoverageExe @arguments 1> $stdoutLog 2> $stderrLog
$coverageExitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference
if ($coverageExitCode -ne 0) {
    throw "OpenCppCoverage/test execution failed with exit code $coverageExitCode. See '$stdoutLog' and '$stderrLog'."
}

if (-not (Test-Path -LiteralPath $coverageXml -PathType Leaf) -or
    -not (Test-Path -LiteralPath (Join-Path $coverageHtml "index.html") -PathType Leaf)) {
    throw "OpenCppCoverage completed without the expected report artifacts in '$coverageDir'."
}

Write-Host "Coverage XML: $coverageXml"
Write-Host "Coverage HTML: $(Join-Path $coverageHtml 'index.html')"
Write-Host "Test stdout: $stdoutLog"
Write-Host "Test stderr: $stderrLog"
