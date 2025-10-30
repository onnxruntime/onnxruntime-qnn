# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

<#
    .DESCRIPTION
    Runs a pre-defined build_and_test.py command using a specific Python version.
    If an appropriate Python iterpreter is not available, it is installed.
#>

$RepoRoot = (Resolve-Path -Path "$(Split-Path -Parent $MyInvocation.MyCommand.Definition)").Path

. "$RepoRoot\qcom\scripts\windows\tools.ps1"

$BootstrapPythonInstallDir = "$RepoRoot\build\bootstrap-tools\python"
$BootstrapPackageCache = "$RepoRoot\build\package-cache"

function Get-PythonExePath() {
    <#
        .DESCRIPTION
        Get the path to a python interpreter, installing it if necessary.
    #>
    param(
        [Parameter(Mandatory = $true)]
        [PSCustomObject]$PyInfo
    )

    $Status = Get-PythonStatus -PyVersion $PyInfo.launcher_version

    switch ($Status.Status) {
        Installed {
            return $Status.ExePath
        }
        NotInstalled {
            Install-Python -PyInfo $PyInfo -TargetDir $BootstrapPythonInstallDir
            return Join-Path $BootstrapPythonInstallDir "python.exe"
        }
        Broken {
            throw "Repairing broken Python installations is not supported."
        }
        default {
            throw "Unknown PythonStatus: $($Status.Status)"
        }
    }
}

function Install-Python() {
    <#
        .DESCRIPTION
        Install Python
    #>
    param(
        [Parameter(Mandatory = $true)]
        [PSCustomObject]$PyInfo,

        [Parameter(Mandatory = $true)]
        [string]$TargetDir
    )

    Write-Host "Installing Python from $($PyInfo.installer_path)"

    $InstallerArgs = @() + $PyInfo.install_args

    $TargetDirSet = $false
    for ($i = 0; $i -lt $InstallerArgs.Count; $i++) {
        if ($InstallerArgs[$i].StartsWith("TargetDir=")) {
            $InstallerArgs[$i] = "TargetDir=$TargetDir"
            $TargetDirSet = $true
        }
    }

    if (-not $TargetDirSet) {
        $InstallerArgs += "TargetDir=$TargetDir"
    }

    Start-Process -Wait -FilePath $PyInfo.installer_path -ArgumentList $InstallerArgs
    if (-not $?) {
        throw "Failed to install Python"
    }
}

$BootstrapJsonPath = (Join-Path $RepoRoot "bootstrap.json")
$BootstrapInfo = Get-Content -Path $BootstrapJsonPath -Raw | ConvertFrom-Json
if (-not $?) {
    throw "Could not load $BootstrapJsonPath"
}

$PythonExe = Get-PythonExePath $BootstrapInfo.python

$env:ORT_BUILD_PACKAGE_CACHE_PATH = $BootstrapPackageCache
& $PythonExe "$RepoRoot\qcom\build_and_test.py" $BootstrapInfo.build_and_test.args
