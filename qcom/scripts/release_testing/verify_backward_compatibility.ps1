# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

param(
    [Parameter(Mandatory=$true)]
    [string]$PythonVersion,
    [Parameter(Mandatory=$true)]
    [string]$WheelArch,
    [Parameter(Mandatory=$true)]
    [string]$WheelDirectory,
    [Parameter(Mandatory=$true)]
    [string]$OnnxruntimeVersion,
    [Parameter(Mandatory=$true)]
    [string]$SamplePath
)

$ErrorActionPreference = 'Stop'

# Force UTF-8 for native-process I/O so ONNX Runtime's wide-char log output
# isn't misrendered with spaces between every character.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding           = [System.Text.Encoding]::UTF8

# Suppress ANSI color escape codes ([0;93m...[m) in ORT's log output —
# they render as literal text in the GitHub Actions log viewer.
$env:NO_COLOR = "1"

$pyNoDot = $PythonVersion.Replace(".", "")
$envName = "py${pyNoDot}_release_backward_compatibility_env"

# Find the wheel that matches both the Python version and the platform
$wheel = Get-ChildItem -Path $WheelDirectory `
  -Filter "*cp${pyNoDot}-cp${pyNoDot}-${WheelArch}.whl" |
  Select-Object -First 1

if (-not $wheel) {
    Write-Host "No wheel found matching cp${pyNoDot}-${WheelArch} in $WheelDirectory" -ForegroundColor Red
    exit 1
}
Write-Host "Found wheel: $($wheel.FullName)" -ForegroundColor Cyan

# Create venv using the py launcher to pick the requested Python version
$pyTag = if ($WheelArch -eq "win_arm64") { "$PythonVersion-arm64" } else { "$PythonVersion" }
if (Test-Path $envName) { Remove-Item -Path $envName -Recurse -Force }
py -$pyTag -m venv $envName
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to create venv (exit $LASTEXITCODE)" -ForegroundColor Red
    exit $LASTEXITCODE
}

try {
    # Activate venv (dot-source so the PATH update persists in this script's scope)
    . "$envName/Scripts/Activate.ps1"

    # Upgrade pip
    python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        Write-Host "pip upgrade FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }

    # Install onnxruntime-qnn from the local wheel (pulls onnxruntime as a dependency)
    python -m pip install $wheel.FullName
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Wheel install FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }

    # Replace the bundled onnxruntime with the older version to verify backward compatibility
    python -m pip uninstall -y onnxruntime
    if ($LASTEXITCODE -ne 0) {
        Write-Host "onnxruntime uninstall FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }

    python -m pip install "onnxruntime==$OnnxruntimeVersion"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "onnxruntime==$OnnxruntimeVersion install FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }

    # Run the sample test against the older onnxruntime
    python $SamplePath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Backward Compatibility test FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "Backward Compatibility test PASSED" -ForegroundColor Green
} finally {
    Remove-Item -Path $envName -Recurse -Force -ErrorAction SilentlyContinue
}
