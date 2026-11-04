# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

# QDC entry point. The existing build archive keeps its repo-relative layout.
$ErrorActionPreference = "Stop"
$PayloadRoot = $PSScriptRoot
$LogDir = "C:\Temp\QDC_logs"
$ExitCode = 1
$PythonInstallStarted = $false
$TranscriptStarted = $false
$Installer = Join-Path $PayloadRoot "installers\python-arm64.exe"
$PythonRoot = $null
$BinaryRoot = $null

function Get-RegisteredPython {
    param([string]$Version)
    $ShortVersion = ([version]$Version).ToString(2)
    $PythonKey = "Software\Python\PythonCore\$ShortVersion-arm64"
    $UninstallKey = "Software\Microsoft\Windows\CurrentVersion\Uninstall"
    # Include other patch versions and MSI components: the bundle may upgrade or repair them.
    $ProductName = "^Python $([regex]::Escape($ShortVersion))\.\d.*\(ARM64\)$"
    $Registered = $false
    $Executable = $null
    foreach ($Hive in @([Microsoft.Win32.RegistryHive]::CurrentUser, [Microsoft.Win32.RegistryHive]::LocalMachine)) {
        foreach ($View in @([Microsoft.Win32.RegistryView]::Registry64, [Microsoft.Win32.RegistryView]::Registry32)) {
            $Registry = [Microsoft.Win32.RegistryKey]::OpenBaseKey($Hive, $View)
            try {
                $Python = $Registry.OpenSubKey($PythonKey)
                if ($null -ne $Python) {
                    try {
                        $Registered = $true
                        $InstallPath = $Python.OpenSubKey("InstallPath")
                        if ($null -ne $InstallPath) {
                            try {
                                $Candidate = $InstallPath.GetValue("ExecutablePath")
                                if (-not $Candidate -and $InstallPath.GetValue("")) {
                                    $Candidate = Join-Path $InstallPath.GetValue("") "python.exe"
                                }
                                if (-not $Executable -and $Candidate) { $Executable = $Candidate }
                            } finally { $InstallPath.Dispose() }
                        }
                    } finally { $Python.Dispose() }
                }
                # Burn can remain registered even when the interpreter/InstallPath is missing.
                $Uninstall = $Registry.OpenSubKey($UninstallKey)
                if ($null -ne $Uninstall) {
                    try {
                        foreach ($Name in $Uninstall.GetSubKeyNames()) {
                            $Product = $Uninstall.OpenSubKey($Name)
                            if ($null -eq $Product) { throw "Uninstall registration changed while checking Python: $Name" }
                            try {
                                if ($Product.GetValue("DisplayName") -match $ProductName) { $Registered = $true }
                            } finally { $Product.Dispose() }
                        }
                    } finally { $Uninstall.Dispose() }
                }
            } finally { $Registry.Dispose() }
        }
    }
    # Registry read errors propagate; they cannot establish that a fresh install is safe.
    return [pscustomobject]@{ Registered = $Registered; Executable = $Executable }
}

function Assert-HtpResults {
    param([string]$Report, [string]$RequiredTest = "")
    if (-not (Test-Path -LiteralPath $Report)) {
        throw "HTP result report is missing: $Report"
    }
    [xml]$Results = Get-Content -LiteralPath $Report -Raw
    $Cases = @($Results.SelectNodes("//testcase[starts-with(@classname, 'QnnHTP')]"))
    if ($RequiredTest) {
        $Cases = @($Cases | Where-Object { $_.name -eq $RequiredTest })
    }
    $Passed = @($Cases | Where-Object {
        $_.status -eq "run" -and $_.result -in @("completed", "complete") -and
        $null -eq $_.SelectSingleNode("skipped") -and $null -eq $_.SelectSingleNode("failure")
    })
    Write-Host "HTP test cases: $($Cases.Count); passed: $($Passed.Count)"
    if ($Passed.Count -eq 0) {
        throw "No HTP tests passed (missing, failed, or all skipped)."
    }
}

try {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    # QDC devices can be reused; avoid publishing reports from a previous job.
    Get-ChildItem -LiteralPath $LogDir -File | Remove-Item -Force
    Start-Transcript -Path (Join-Path $LogDir "bootstrap_stdout.txt") | Out-Null
    $TranscriptStarted = $true
    $Settings = Get-Content (Join-Path $PayloadRoot "qdc_windows_config.json") -Raw | ConvertFrom-Json
    if ((Get-FileHash -LiteralPath $Installer -Algorithm SHA256).Hash -ne $Settings.python_sha256) {
        throw "Python installer SHA-256 mismatch."
    }
    $PythonRoot = "C:\Temp\ORT-QDC-Python-$($Settings.python_version)"
    $PythonExe = Join-Path $PythonRoot "python.exe"
    $ExistingPython = Get-RegisteredPython -Version $Settings.python_version
    if ($ExistingPython.Registered) {
        if (-not $ExistingPython.Executable -or -not (Test-Path -LiteralPath $ExistingPython.Executable -PathType Leaf)) {
            throw "ARM64 Python is already registered but has no usable interpreter; leaving the existing product unchanged."
        }
        $PythonExe = $ExistingPython.Executable
        Write-Host "Reusing registered Python: $PythonExe"
    } elseif (-not (Test-Path -LiteralPath $PythonExe)) {
        $InstallArgs = @(
            "/quiet", "/norestart", "/log", "`"$(Join-Path $LogDir 'python-install.log')`"",
            # QDC runs as SYSTEM; use machine installation rather than per-user MSI
            # registration in SYSTEM's incomplete user profile.
            "InstallAllUsers=1", "TargetDir=`"$PythonRoot`"", "Include_launcher=0", "Include_pip=0",
            "Include_doc=0", "Include_tcltk=0", "Include_test=0", "Shortcuts=0", "AssociateFiles=0",
            "PrependPath=0"
        )
        $Process = Start-Process -FilePath $Installer -ArgumentList $InstallArgs -PassThru
        # Only a new, unregistered product belongs to this job, including failed install attempts.
        $PythonInstallStarted = $true
        if (-not $Process.WaitForExit(600000)) {
            & taskkill.exe /PID $Process.Id /T /F
            throw "Python installation timed out after 10 minutes."
        }
        # 3010 means installation succeeded and a reboot was requested. Require a usable interpreter below.
        if ($Process.ExitCode -notin @(0, 3010)) {
            throw "Python installation failed with exit code $($Process.ExitCode)."
        }
    }
    $Version = & $PythonExe -c "import platform; print(platform.python_version()); print(platform.machine())"
    if ($LASTEXITCODE -ne 0 -or $Version[0] -ne $Settings.python_version -or $Version[1] -ne "ARM64") {
        throw "The requested ARM64 Python is unavailable: $Version"
    }
    Write-Host "Helper Python: $PythonExe ($Version)"

    $Config = $Settings.build_config
    $TestRoot = Join-Path $PayloadRoot "build\windows-arm64\$Config"
    $BinaryRoot = $TestRoot
    if (Test-Path (Join-Path $TestRoot "$Config\onnxruntime_provider_test.exe")) {
        $BinaryRoot = Join-Path $TestRoot $Config
    }
    . (Join-Path $TestRoot "utils.ps1")
    if ((Get-HostArch) -ne "arm64") {
        throw "QDC Windows tests require an ARM64 device."
    }
    Get-CimInstance Win32_Processor | Select-Object Name, Architecture | Format-List | Out-Host

    # This small QDQ Add test requires full QNN assignment, runs inference and checks output accuracy.
    $ProbeTest = "BinaryOp_Add4D"
    $ProbeXml = Join-Path $LogDir "htp-smoke.results.xml"
    $ProbeLog = Join-Path $LogDir "htp-smoke_stdout.txt"
    $ProbeErrorLog = Join-Path $LogDir "htp-smoke-stderr_stdout.txt"
    $ProbeExitCodeFile = Join-Path $LogDir "htp-smoke.exitcode.txt"
    $ProbeWrapper = Join-Path $LogDir "run-htp-smoke.cmd"
    $ProbeExecutable = Join-Path $BinaryRoot "onnxruntime_provider_test.exe"
    $env:ORT_UNIT_TEST_MAIN_LOG_LEVEL = "1"
    @"
@echo off
"$ProbeExecutable" --gtest_filter=QnnHTPBackendTests.$ProbeTest "--gtest_output=xml:$ProbeXml"
set "PROBE_EXIT_CODE=%ERRORLEVEL%"
> "$ProbeExitCodeFile" echo %PROBE_EXIT_CODE%
exit /b %PROBE_EXIT_CODE%
"@ | Set-Content -LiteralPath $ProbeWrapper -Encoding Ascii
    $Probe = Start-Process -FilePath $env:ComSpec -WorkingDirectory $BinaryRoot `
        -ArgumentList @("/d", "/s", "/c", "`"$ProbeWrapper`"") `
        -RedirectStandardOutput $ProbeLog -RedirectStandardError $ProbeErrorLog -PassThru
    if (-not $Probe.WaitForExit(300000)) {
        & taskkill.exe /PID $Probe.Id /T /F
        throw "HTP smoke test timed out after 5 minutes."
    }
    Get-Content -LiteralPath $ProbeLog | Out-Host
    Get-Content -LiteralPath $ProbeErrorLog | Out-Host
    # Capture the process state before interpreting it. This is needed to diagnose
    # the observed QDC session where a completed smoke test exposed no ExitCode.
    $ProbeExitCode = $null
    $ProbeExitCodeReadError = $null
    try {
        $ProbeExitCode = $Probe.ExitCode
    } catch {
        $ProbeExitCodeReadError = $_.Exception.Message
    }
    $ProbeExitCodeValue = if ($null -eq $ProbeExitCode) {
        "<null>"
    } elseif ([string]::IsNullOrWhiteSpace([string]$ProbeExitCode)) {
        "<empty>"
    } else {
        $ProbeExitCode
    }
    $ProbeExitCodeType = if ($null -eq $ProbeExitCode) { "<null>" } else { $ProbeExitCode.GetType().FullName }
    $ProbeExitCodeError = if ($ProbeExitCodeReadError) { $ProbeExitCodeReadError } else { "<none>" }
    $ProbeDiagnostic = "HTP smoke process diagnostic: Type=$($Probe.GetType().FullName); PID=$($Probe.Id); " +
        "HasExited=$($Probe.HasExited); ExitCodeRaw=$ProbeExitCodeValue; ExitCodeType=$ProbeExitCodeType; " +
        "ExitCodeReadError=$ProbeExitCodeError"
    Write-Host $ProbeDiagnostic

    # Persist the child's ERRORLEVEL because QDC can expose a completed Process
    # whose ExitCode property is null. Require both this marker and the JUnit
    # report so a post-test crash cannot be mistaken for a pass.
    if (-not (Test-Path -LiteralPath $ProbeExitCodeFile -PathType Leaf)) {
        throw "HTP smoke exit-code marker is missing: $ProbeExitCodeFile"
    }
    $ProbeChildExitCode = (Get-Content -LiteralPath $ProbeExitCodeFile -Raw).Trim()
    if ($ProbeChildExitCode -ne "0") {
        throw "HTP smoke test failed with exit code marker '$ProbeChildExitCode'."
    }
    Assert-HtpResults -Report $ProbeXml -RequiredTest $ProbeTest
    $ProbeOutput = Get-Content -LiteralPath $ProbeLog -Raw
    if ($ProbeOutput -notmatch "HTP arch: $($Settings.expected_htp_arch)\b") {
        throw "Device did not report the required HTP architecture v$($Settings.expected_htp_arch)."
    }
    Write-Host "Verified HTP v$($Settings.expected_htp_arch) inference on QDC."

    if ($Settings.test_mode -eq "full") {
        & (Join-Path $TestRoot "run_tests.ps1") -Config $Config -PythonExe $PythonExe -TestRepoRoot $PayloadRoot
        Assert-HtpResults -Report (Join-Path $BinaryRoot "onnxruntime_provider_test.exe.$Config.results.xml")
    } elseif ($Settings.test_mode -ne "smoke") {
        throw "Unsupported test mode: $($Settings.test_mode)"
    }
    $ExitCode = 0
} catch {
    Write-Host "QDC bootstrap failed: $_"
} finally {
    # A failed diagnostic operation must not prevent resource cleanup or outcome reporting.
    try {
        if ($BinaryRoot -and (Test-Path -LiteralPath $BinaryRoot)) {
            Get-ChildItem -LiteralPath $BinaryRoot -File |
                Where-Object { $_.Name -like "*.results.xml" -or $_.Extension -eq ".log" } |
                Copy-Item -Destination $LogDir -Force
            $CTestLog = Join-Path $TestRoot "Testing\Temporary\LastTest.log"
            if (Test-Path -LiteralPath $CTestLog) {
                Copy-Item -LiteralPath $CTestLog -Destination (Join-Path $LogDir "ctest_stdout.txt") -Force
            }
        }
    } catch {
        Write-Host "Diagnostic collection failed: $_"
        $ExitCode = 1
    }
    if ($PythonInstallStarted) {
        try {
            $Cleanup = Start-Process -FilePath $Installer -ArgumentList @(
                "/quiet", "/uninstall", "/norestart", "/log",
                "`"$(Join-Path $LogDir 'python-uninstall.log')`""
            ) -PassThru
            if (-not $Cleanup.WaitForExit(300000)) {
                & taskkill.exe /PID $Cleanup.Id /T /F
                throw "Python cleanup timed out."
            }
            if ($Cleanup.ExitCode -notin @(0, 3010)) {
                throw "Python cleanup failed with exit code $($Cleanup.ExitCode)."
            }
        } catch {
            Write-Host "Python cleanup failed: $_"
            $ExitCode = 1
        }
    }
    if ($TranscriptStarted) {
        try {
            Stop-Transcript | Out-Null
        } catch {
            Write-Host "Transcript shutdown failed: $_"
            $ExitCode = 1
        }
    }
    try {
        New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
        $Failure = if ($ExitCode -ne 0) { '<failure message="QDC setup, device tests, or cleanup failed; see bootstrap_stdout.txt" />' } else { '' }
        @"
<testsuites><testsuite name="QDCWindows" tests="1" failures="$ExitCode"><testcase classname="QDCWindows" name="BootstrapAndTests">$Failure</testcase></testsuite></testsuites>
"@ | Set-Content -LiteralPath (Join-Path $LogDir "bootstrap.results.xml") -Encoding UTF8
    } catch {
        Write-Host "Bootstrap result publication failed: $_"
        $ExitCode = 1
    }
}
exit $ExitCode
