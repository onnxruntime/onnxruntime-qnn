# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

param (
    [Parameter(Mandatory = $true,
               HelpMessage = "Path to input directory containing nupkg files.")]
    [string]$InputDir
)

$RepoRoot = (Resolve-Path -Path "$(Split-Path -Parent $MyInvocation.MyCommand.Definition)\..\..\..").Path

. "$RepoRoot\qcom\scripts\upleveling\prepare_nuget.ps1"

# Verify that credentials are provided via environment variables
if (-not $env:ARTIFACTORY_USERNAME -or -not $env:ARTIFACTORY_PASSWORD) {
    Write-Error "Credentials must be provided via ARTIFACTORY_USERNAME and ARTIFACTORY_PASSWORD environment variables"
    exit 1
}

$nugetFiles = Get-ChildItem -Path $InputDir -Filter "*.nupkg" -File -Recurse

foreach ($file in $nugetFiles) {
    $fileBasename = $file.BaseName
    $version = $fileBasename -replace "^Qualcomm\.ML\.OnnxRuntime\.QNN\.", ""
    Write-Host "${fileBasename}: $version"

    $sourceName = "re-artifactory-nuget-test-project-$version"
    $sourceUrl = "https://re-artifactory.qualcomm.com/artifactory/api/nuget/aisw-nuget-test-project/onnxruntime-qnn/$version"

    # Use PackageSourceCredentials environment variables for secure authentication
    # NuGet automatically reads credentials from environment variables in the format:
    # NuGetPackageSourceCredentials_<SourceName> with value "Username={Username};Password={Password}"
    # Sanitize source name to make it valid for environment variable names (alphanumeric and underscores only)
    $sourceNameSafe = $sourceName -replace '[^a-zA-Z0-9]', '_'
    $sourceCredentialsVar = "NuGetPackageSourceCredentials_${sourceNameSafe}"
    Set-Item -Path "env:$sourceCredentialsVar" -Value "Username=$env:ARTIFACTORY_USERNAME;Password=$env:ARTIFACTORY_PASSWORD"

    nuget sources Add `
        -Name $sourceNameSafe `
        -Source $sourceUrl

    nuget push "$($file.FullName)" -Source $sourceNameSafe

    nuget sources Remove -Name $sourceNameSafe

    # Clean up environment variables
    Remove-Item -Path "env:$sourceCredentialsVar" -ErrorAction SilentlyContinue
}
