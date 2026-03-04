# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

param ( 
    [Parameter(Mandatory = $true,
               HelpMessage = "Product name.")]
    [string]$ProductName,

    [Parameter(Mandatory = $true,
               HelpMessage = "The format of artifact. Choose one of [wheel, nuget, zip].")]
    [string]$ArtifactFormat,

    [Parameter(Mandatory = $true,
               HelpMessage = "Source version of artifact.")]
    [string]$VersionFrom,

    [Parameter(Mandatory = $true,
               HelpMessage = "Target version of artifact.")]
    [string]$VersionTo,

    [Parameter(Mandatory = $true,
               HelpMessage = "From which server the artifact is located.")]
    [string]$IndexServerFrom,

    [Parameter(Mandatory = $true,
               HelpMessage = "To which server to store the artifact.")]
    [string]$IndexServerTo
)

function Set-NuGetCredentials {
    param($server, $version)

    if ($server -eq "testnuget") {
        $source_name = "testnuget.org"
    } elseif ($server -eq "nuget") {
        $source_name = "nuget.org"
    } else {
        $source_name = "re-artifactory-nuget-$server-$version"
    }

    # Sanitize source name for environment variable
    $source_name_safe = $source_name -replace '-','_' -replace '\.','_'

    # Set PackageSourceCredentials environment variable
    $source_credentials_var = "NuGetPackageSourceCredentials_${source_name_safe}"

    if ($server -eq "testnuget") {
        Set-Item -Path "env:$source_credentials_var" -Value "Username=$env:TEST_NUGET_API_KEY;Password=$env:TEST_NUGET_API_KEY"
    } elseif ($server -eq "nuget") {
        Set-Item -Path "env:$source_credentials_var" -Value "Username=$env:NUGET_API_KEY;Password=$env:NUGET_API_KEY"
    } else {
        Set-Item -Path "env:$source_credentials_var" -Value "Username=$env:ARTIFACTORY_USERNAME;Password=$env:ARTIFACTORY_PASSWORD"
    }

    Write-Host "Set PackageSourceCredentials for $source_name_safe"
    return $source_credentials_var
}

$RepoRoot = (Resolve-Path -Path "$(Split-Path -Parent $MyInvocation.MyCommand.Definition)\..\..\..").Path

. "$RepoRoot\qcom\scripts\upleveling\prepare_nuget.ps1"

# Set NuGet PackageSourceCredentials environment variables for both source and target
$source_cred_var = Set-NuGetCredentials -server $IndexServerFrom -version $VersionFrom
$target_cred_var = Set-NuGetCredentials -server $IndexServerTo -version $VersionTo

python $RepoRoot\qcom\scripts\upleveling\qnn_ep_uplevel.py `
    --product_name $ProductName `
    --artifact_format "nuget" `
    --version_from $VersionFrom `
    --version_to $VersionTo `
    --index_server_from $IndexServerFrom `
    --index_server_to $IndexServerTo

# Clean up the environment variables
Remove-Item -Path "env:$source_cred_var" -ErrorAction SilentlyContinue
Remove-Item -Path "env:$target_cred_var" -ErrorAction SilentlyContinue
