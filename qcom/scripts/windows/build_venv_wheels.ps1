# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

param (
    [Parameter(Mandatory = $true,
               HelpMessage = "The Python virtual environment whose packages to build into wheels.")]
    [string]$VEnvPath,

    [Parameter(Mandatory = $true,
               HelpMessage = "Where to put the wheels.")]
    [string]$WheelDir
)

$RepoRoot = (Resolve-Path -Path "$(Split-Path -Parent $MyInvocation.MyCommand.Definition)\..\..\..").Path

. "$RepoRoot\qcom\scripts\windows\utils.ps1"

Use-PyVenv -PyVenv $VEnvPath {
    if ($null -ne $WheelDir) {
        python.exe -m pip list | Select-Object -Skip 2 | ForEach-Object {
            $parts = $_ -split '\s+'
            $package = "$($parts[0])==$($parts[1])"
            Write-Host "Building wheel of $package into $WheelDir"
            Assert-Success -ErrorMessage "Failed to build wheel of $package" {
                python.exe -m pip wheel --wheel-dir $WheelDir $package
            }
        }
    }
}
