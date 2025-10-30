# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

param(
    [Parameter(HelpMessage = "The build config")]
    [ValidateSet("Debug", "Release", "RelWithDebInfo")]
    [string]$Config = "Release",

    [Parameter(Mandatory = $true, HelpMessage = "The target architecture")]
    [ValidateSet("arm64", "x86_64")]
    [string]$Arch,

    [Parameter(Mandatory = $true, HelpMessage = "Path to onnx/models models")]
    [string]$OnnxModelsRoot,

    [Parameter(HelpMessage = "Path to Python executable to use for testing.")]
    [string]$PyExePath = $null
)

$ScriptDir = (Resolve-Path -Path "$(Split-Path -Parent $MyInvocation.MyCommand.Definition)").Path
$RepoRoot = (Resolve-Path -Path ("$ScriptDir\..\..\.."))

. "$RepoRoot\qcom\scripts\windows\tools.ps1"

$InQdc = (Test-Path "C:\Temp\TestContent")  # a crude but effective test

$BuildRoot = (Join-Path (Join-Path (Join-Path $RepoRoot "build") "windows-$Arch") $Config)
$BuildBinDir = $BuildRoot
if ((Test-Path (Join-Path $BuildRoot $Config))) {
    # Multi-config generators like Visual Studio 2022 put executables one level deeper.
    $BuildBinDir = (Join-Path $BuildRoot $Config)
}

$CTestTestFile = (Join-Path $BuildRoot "CTestTestfile.cmake")
if (-not (Test-Path $CTestTestFile)) {
    throw "$CTestTestFile not found"
}

$CTestExe = (Join-Path $(Get-CMakeBinDir) "ctest.exe")
$OnnxTestRunnerExe = (Join-Path $BuildBinDir "onnx_test_runner.exe")

# Extract the build path from CTestTestfile.cmake
$OldBuildDirectoryRegex = (Select-String -Path $CTestTestFile -Pattern "# Build directory: (.*)$").Matches.Groups[1].Value
$OldBuildDirectoryBackslashesRegex = ($OldBuildDirectoryRegex -replace "/", "\\\\")

# Substitutions to point CTest at the build directory
$NewBuildDirectory = ($BuildRoot -replace "\\", "/")
$NewBuildDirectoryBackslashes = ($NewBuildDirectory -replace "/", "\\")

# Rewrite CTestTestfile.cmake
Copy-Item $CTestTestFile "$CTestTestFile.bak"
(Get-Content $CTestTestFile) `
    -replace $OldBuildDirectoryRegex, $NewBuildDirectory `
    -replace $OldBuildDirectoryBackslashesRegex, $NewBuildDirectoryBackslashes |
    Out-File -Encoding ascii $CTestTestFile

    # Figure out if HTP is available
if ((Get-CimInstance Win32_operatingsystem).OSArchitecture -eq "ARM 64-bit Processor") {
    $QdqBackend = "htp"
} else {
    $QdqBackend = "cpu"
}

$Failed = @()

# Run CTest
Push-Location $BuildRoot
Write-Host "--=-=-=- Running unit tests -=--=-=-"
& $CTestExe --build-config $Config --verbose --timeout $(60 * 60)

if (-not $?) {
    Write-Host "Unit tests failed. Will exit with error after running model tests."
    $Failed += ("Unit tests")
}

if ($PyExePath) {
    Write-Host "--=-=-=- Running Python tests -=--=-=-"
    Push-Location $BuildBinDir
    $PythonTestFilesPath = "$RepoRoot\qcom\scripts\all\python_test_files.txt"

    if (Test-Path $PythonTestFilesPath) {
        $PythonTestFiles = Get-Content $PythonTestFilesPath

        foreach ($PythonFile in $PythonTestFiles) {
            $PythonFile = $PythonFile.Trim()
            if ($PythonFile -and (Test-Path $PythonFile)) {

                # TODO - AISW-139802 - Tests in the following files hang on Windows - skip them for now
                if ($PythonFile -like "*onnxruntime_test_python_backend.py" -or
                        $PythonFile -like "*onnxruntime_test_python_global_threadpool.py") {
                    Write-Host "Skipping $PythonFile - contains a test that hangs on Windows"
                    continue
                }

                Write-Host "Running $PythonFile..."
                & $PyExePath $PythonFile
                if (-not $?) {
                    Write-Error "Python test $PythonFile failed."
                    $Failed += ($PythonFile)
                }
            } else {
                Write-Warning "Failed to find $PythonFile."
                $Failed += ("Failed to find $PythonFile")
            }
        }
    } else {
        Write-Error "Python test files list not found at $PythonTestFilesPath"
        $Failed += ("Python test discovery ($PythonTestFilesPath)")
    }

    # TODO: [AISW-157198] Quantization Python tests fail on QDC WoS devices
    if (-not $InQdc) {
        if (Test-Path "quantization" -PathType Container) {
            Write-Host "Running quantization tests..."
            & $PyExePath -m unittest discover -s quantization
        } else {
            Write-Warning "Failed to find directory 'quantization'."
            $Failed += ("Quantization python test discovery")
        }
        if (-not $?) {
            Write-Error "Quantization tests failed."
            $Failed += ("Quantization")
        }
    } else {
        Write-Warning "Skipping quantization tests in QDC."
    }
} else {
    Write-Host "Not running Python tests."
}

Push-Location $BuildRoot
Write-Host "--=-=-=- Running ONNX model tests -=--=-=-"
& $OnnxTestRunnerExe `
    -j 1 `
    -e qnn `
    -i "backend_type|cpu" `
    "$RepoRoot\cmake\external\onnx\onnx\backend\test\data\node"
if (-not $?) {
    $Failed += ("ONNX node models")
}

Write-Host "-=-=-=- Running onnx/models float32 tests -=-=-=-"
Push-Location $OnnxModelsRoot
if (-not $?) {
    throw "Could not cd to $OnnxModelsRoot"
}

& $OnnxTestRunnerExe `
    -j 1 `
    -e qnn `
    -i "backend_type|cpu" `
    "testdata\float32"
if (-not $?) {
    $Failed += ("float32 CPU models")
}

Write-Host "-=-=-=- Running onnx/models qdq tests -=-=-=-"
& $OnnxTestRunnerExe `
    -j 1 `
    -e qnn `
    -i "backend_type|$QdqBackend" `
    "testdata\qdq"
if (-not $?) {
    $Failed += ("QDQ models")
}

if ($QdqBackend -ne "cpu") {
    Write-Host "-=-=-=- Running onnx/models qdq tests with context cache enabled -=-=-=-"
    # Scrub old context caches
    Get-ChildItem -Path "testdata\qdq-with-context-cache" -Recurse -Filter "*_ctx.onnx" | Remove-Item -Force
    & $OnnxTestRunnerExe `
        -j 1 `
        -e qnn `
        -f -i "backend_type|$QdqBackend" `
        "testdata\qdq-with-context-cache"
    if (-not $?) {
        $Failed = ("QDQ models with a context cache")
    }
} else {
    Write-Host "Not running onnx/models qdq tests with context cache enabled on CPU backend."
}

# If it looks like we're running in QDC, copy logs to the directory they'll scan to find them.
if (Test-Path "C:\Temp\TestContent") {
    $QdcLogsDir = "C:\Temp\QDC_logs"
    if (Test-Path $QdcLogsDir) {
        Remove-Item -Recurse -Force -Path $QdcLogsDir
        if (-not $?) {
            throw "Failed to clear old QDC logs dir $QdcLogsDir"
        }
    }

    New-Item -ItemType Directory -Force -Path $QdcLogsDir | Out-Null
    if (-not $?) {
        throw "Failed to create QDC logs dir $QdcLogsDir"
    }

    Write-Host "Copying logs $BuildBinDir\*.xml --> $QdcLogsDir"
    Copy-Item $BuildBinDir\*.xml $QdcLogsDir
}

if ($Failed.Count -gt 0) {
    throw "Tests failed: " + ($Failed -join ", ")
}
