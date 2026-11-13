# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

param(
    [Parameter(HelpMessage = "The build config")]
    [ValidateSet("Debug", "Release", "RelWithDebInfo")]
    [string]$Config = "Release",
    [Parameter(HelpMessage = "Path to onnx/models models")]
    $OnnxModelsRoot = $null,
    [Parameter(HelpMessage = "Number of sequential provider-test processes")]
    [ValidateSet(1, 2)]
    [int]$ProviderTestProcesses = 1
)

$RootDir = (Resolve-Path -Path "$(Split-Path -Parent $MyInvocation.MyCommand.Definition)").Path
$RepoRoot= (Resolve-Path -Path ("$RootDir\..\..\.."))

. (Join-Path $RootDir "utils.ps1")

# QNN CPU is not advertised by default; tests use it as the x86 attach point, so opt in.
$env:ORT_QNN_ENABLE_CPU_BACKEND = "1"

if (-not $OnnxModelsRoot) {
    $OnnxModelsRoot = (Join-Path $RootDir (Join-Path "model_tests" "onnx_models"))
}

$TimeoutSec = 60 * 60

$CTestExe = (Join-Path $RootDir "ctest.exe")

# Single-config generators like Ninja put runners in the parent directory
# of the rest of the ONNX Runtime build.
if (Test-Path (Join-Path $RootDir "onnxruntime_plugin_ep_onnx_test.exe")) {
    $OnnxEpTestRunnerExe = (Join-Path $RootDir "onnxruntime_plugin_ep_onnx_test.exe")
    $ResultsXmlDir = $RootDir
} else {
    $OnnxEpTestRunnerExe = (Join-Path (Join-Path $RootDir $Config) "onnxruntime_plugin_ep_onnx_test.exe")
    $ResultsXmlDir = (Join-Path $RootDir $Config)
}

$CTestTestFile = (Join-Path $RootDir "CTestTestfile.cmake")

if (-not (Test-Path $CTestTestFile)) {
    throw "$CTestTestFile not found"
}

# Extract the build path from CTestTestfile.cmake
$OldBuildDirectoryRegex = (Select-String -Path $CTestTestFile -Pattern "# Build directory: (.*)$").Matches.Groups[1].Value
$OldBuildDirectoryBackslashesRegex = ($OldBuildDirectoryRegex -replace "/", "\\\\")

# Substitutions to point CTest at this directory
$NewBuildDirectory = ($RootDir -replace "\\", "/")
$NewBuildDirectoryBackslashes = ($NewBuildDirectory -replace "/", "\\")

# Rewrite CTestTestfile.cmake
(Get-Content $CTestTestFile) `
    -replace $OldBuildDirectoryRegex, $NewBuildDirectory `
    -replace $OldBuildDirectoryBackslashesRegex, $NewBuildDirectoryBackslashes |
    Out-File -Encoding ascii $CTestTestFile

# Figure out if HTP is available
if ((Get-HostArch) -eq "arm64") {
    $QdqBackend = "htp"
} else {
    $QdqBackend = "cpu"
}

$Failed = $false

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# CTest
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

Push-Location $RootDir
Write-Host "--=-=-=- Running unit tests -=--=-=-"
if ($ProviderTestProcesses -eq 1) {
    & $CTestExe --build-config $Config --verbose --timeout $TimeoutSec
    if ($LASTEXITCODE -ne 0) { $Failed = $true }
} else {
    # Keep the registered CTest command/filter; split the native ARM64 internal
    # workload across two sequential processes while session cleanup is investigated.
    $PreviousTotalShards = $env:GTEST_TOTAL_SHARDS
    $PreviousShardIndex = $env:GTEST_SHARD_INDEX
    $ProviderReport = Join-Path $ResultsXmlDir "onnxruntime_provider_test.exe.$Config.results.xml"
    try {
        $env:GTEST_TOTAL_SHARDS = "2"
        for ($Shard = 0; $Shard -lt 2; $Shard++) {
            $env:GTEST_SHARD_INDEX = "$Shard"
            if (Test-Path -LiteralPath $ProviderReport) { Remove-Item -LiteralPath $ProviderReport }
            Write-Host "Running provider-test process $($Shard + 1)/2 through CTest"
            & $CTestExe --build-config $Config --verbose --timeout $TimeoutSec
            if ($LASTEXITCODE -ne 0) { $Failed = $true }
            if (Test-Path -LiteralPath $ProviderReport) {
                $ShardReport = Join-Path $ResultsXmlDir "onnxruntime_provider_test.exe.$Config.shard$Shard.results.xml"
                Move-Item -LiteralPath $ProviderReport -Destination $ShardReport -Force
            } else {
                Write-Warning "Provider-test process $($Shard + 1)/2 did not produce its XML report."
                $Failed = $true
            }
        }
    } finally {
        $env:GTEST_TOTAL_SHARDS = $PreviousTotalShards
        $env:GTEST_SHARD_INDEX = $PreviousShardIndex
    }
}
if ($Failed) {
    Write-Host "Unit tests failed. Will exit with error after running model tests."
}

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Python tests
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# AISW-152430 - Do not run Python tests on Windows ARM for now due to frankenstein build process for ARM Python wheel

if ((Get-CimInstance Win32_Processor).Architecture -ne 12) {  # Architecture code 12 corresponds to ARM64
    Write-Host "--=-=-=- Running Python tests -=--=-=-"
    $PythonTestFilesPath = (Join-Path $RootDir "python_test_files.txt")

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
                & python $PythonFile
                if (-not $?) {
                    Write-Error "Python test $PythonFile failed."
                    $Failed = $true
                }
            } else {
                Write-Warning "Failed to find $PythonFile - may be OK on platforms which do not support Python."
            }
        }
    } else {
        Write-Error "Python test files list not found at $PythonTestFilesPath"
        $Failed = $true
    }

    if (Test-Path "quantization" -PathType Container) {
        Write-Host "Running quantization tests..."
        & python -m unittest discover -s quantization
    } else {
        Write-Warning "Failed to find directory 'quantization' - may be OK on platforms which do not support Python."
    }
    if (-not $?) {
        Write-Error "Quantization tests failed."
        $Failed = $true
    }
} else {
    Write-Warning "Host is Windows ARM - skipping Python testing for now."
}

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Model tests
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

$TestModelsViaEpPlugin = {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet("cpu", "gpu", "htp")]
        [string]$Backend,
        [Parameter(Mandatory = $true)]
        [string]$Suite,
        [string]$TestPath = "testdata/$Suite"
    )

    Write-Host "--=-=-=- Running ONNX model $Suite tests with the ABI-stable EP plugin -=--=-=-"

    $ModelLog = Join-Path $ResultsXmlDir "${Suite}_model_tests.log"
    $ModelXml = Join-Path $ResultsXmlDir "${Suite}_model_tests.results.xml"

    if (Test-Path $ModelLog) { Remove-Item $ModelLog }
    if (Test-Path $ModelXml) { Remove-Item -Force $ModelXml }

    $RunnerTestPath = $TestPath
    $FilteredTestPath = $null
    $TestResult = $false
    try {
        if ($Suite -eq "node") {
            # The upstream plugin runner takes a whole suite directory and has no
            # per-case skip flag. Filter QNN-owned exclusions into a temporary copy.
            $FilteredTestPath = Join-Path ([System.IO.Path]::GetTempPath()) ("qnn-model-tests-" + [guid]::NewGuid())
            & py "${RepoRoot}\qcom\scripts\all\model_test_filter.py" `
                --source $TestPath `
                --destination $FilteredTestPath `
                --suite $Suite `
                --backend $Backend | Tee-Object -FilePath $ModelLog | Write-Host
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to prepare filtered QNN model suite '$Suite'."
            }
            $RunnerTestPath = $FilteredTestPath
        }

        & $OnnxEpTestRunnerExe `
            -j 1 `
            --plugin_ep_libs "qnn|onnxruntime_providers_qnn.dll" `
            --plugin_eps qnn `
            -i "backend_type|$Backend" `
            $RunnerTestPath | Tee-Object -FilePath $ModelLog -Append | Write-Host
        $TestResult = $LASTEXITCODE -eq 0
    } finally {
        if ($null -ne $FilteredTestPath -and (Test-Path -LiteralPath $FilteredTestPath)) {
            try {
                Remove-Item -LiteralPath $FilteredTestPath -Recurse -Force -ErrorAction Stop
            } catch {
                Write-Warning "Failed to remove filtered QNN model suite '$FilteredTestPath': $_"
            }
        }
    }

    if (Test-Path $ModelLog) {
        py "${RepoRoot}\qcom\scripts\all\model_test_log_to_junit_xml.py" `
            $ModelLog | Out-File -FilePath $ModelXml -Encoding utf8
    }

    if (-not $TestResult) {
        return $true
    }
    return $false
}

Write-Host "--=-=-=- Running ONNX model tests -=--=-=-"

Push-Location $OnnxModelsRoot
if (-not $?) {
    throw "Could not cd to $OnnxModelsRoot"
}

foreach ($RunModelTests in ($TestModelsViaEpPlugin)) {
    $Failed = (& $RunModelTests -Backend cpu -Suite node -TestPath "$RepoRoot\cmake\external\onnx\onnx\backend\test\data\node") -or $Failed
    $Failed = (& $RunModelTests -Backend cpu -Suite float32) -or $Failed
    $Failed = (& $RunModelTests -Backend $QdqBackend -Suite qdq) -or $Failed

    if ($QdqBackend -ne "cpu") {
        Write-Host "-=-=-=- Running onnx/models qdq tests with context cache enabled -=-=-=-"
        # Scrub old context caches
        Get-ChildItem -Path "testdata\qdq-with-context-cache" -Recurse -Filter "*_ctx.onnx" | Remove-Item -Force

        $Failed = (& $RunModelTests -Backend $QdqBackend -Suite qdq-with-context-cache) -or $Failed
    } else {
        Write-Host "Not running onnx/models qdq tests with context cache enabled on CPU backend."
    }
}

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# QDC Logs
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

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

    # This location depends on whether we built with Ninja or Visual Studio.
    $LocalLogsDir = (Join-Path $RootDir $Config)
    if (-not (Test-Path $LocalLogsDir)) {
        $LocalLogsDir = $RootDir
    }
    Write-Host "Copying logs $LocalLogsDir\*.xml --> $QdcLogsDir"
    Copy-Item $LocalLogsDir\*.xml $QdcLogsDir
}

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# All done
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

if ($Failed) {
    throw "Tests failed"
}
