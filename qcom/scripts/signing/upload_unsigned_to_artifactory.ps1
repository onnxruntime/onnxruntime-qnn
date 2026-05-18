# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

# Script to upload unsigned signing artifacts to Artifactory
# Usage: upload_unsigned_to_artifactory.ps1 -ZipFile <zip_file> -Filename <filename> -Version <version> -NetrcFile <netrc_file>

param(
    [Parameter(Mandatory=$true)]
    [string]$ZipFile,
    [Parameter(Mandatory=$true)]
    [string]$Filename,
    [Parameter(Mandatory=$true)]
    [string]$Version,
    [Parameter(Mandatory=$true)]
    [string]$NetrcFile
)

if (-not (Test-Path $ZipFile)) {
    Write-Error "Error: Zip file '$ZipFile' does not exist"
    exit 1
}

if (-not (Test-Path $NetrcFile)) {
    Write-Error "Error: Netrc file '$NetrcFile' does not exist"
    exit 1
}

# Get the repository root directory
$repoRoot = git rev-parse --show-toplevel

Write-Host "Uploading $Filename"

$caCertPath = Join-Path $repoRoot "qcom/scripts/upleveling/certs/artifactory-ca.pem"
$uploadUrl = "https://re-artifactory.qualcomm.com/artifactory/aisw-zip-test-project/onnxruntime-qnn/$Version/unsigned_libs/$Filename"

curl.exe -T "$ZipFile" --fail `
    --cacert "$caCertPath" `
    --netrc-file "$NetrcFile" `
    -s `
    "$uploadUrl" | Out-Null

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to upload $Filename"
    exit 1
}

Write-Host "Successfully uploaded $Filename"
