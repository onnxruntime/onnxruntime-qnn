# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

$RepoRoot = (Resolve-Path -Path "$(Split-Path -Parent $MyInvocation.MyCommand.Definition)\..\..\..").Path

. "$RepoRoot\qcom\scripts\windows\tools.ps1"
. "$RepoRoot\qcom\scripts\windows\utils.ps1"
$TargetNugetDir = Get-NugetBinDir
$env:Path = "$TargetNugetDir;" + $env:Path
