# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("wheel", "zip", "nuget")]
    [string]$ArtifactType,
    [Parameter(Mandatory=$true)]
    [string]$SourceDirectory,
    [Parameter(Mandatory=$true)]
    [string]$ExpectedVersion
)

$ErrorActionPreference = 'Stop'

# Force UTF-8 so any native tool output renders correctly in the GHA log viewer.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding           = [System.Text.Encoding]::UTF8

# =============================================================================
# Shared helpers
# =============================================================================

function Normalize-Version([string]$Version) {
    # Reduce to major.minor.patch — handles four-part FileVersion ("2.5.0.0"),
    # pre-release suffixes ("2.5.0.rc3"), and plain three-part ("2.5.0").
    $parts = $Version.Split('.')
    return "$($parts[0]).$($parts[1]).$($parts[2])"
}

function Test-QnnDll {
    param(
        [Parameter(Mandatory=$true)] [string]$DllPath,
        [Parameter(Mandatory=$true)] [string]$Label,
        [Parameter(Mandatory=$true)] [string]$ExpectedVersion
    )

    $result = [pscustomobject]@{ CertPass = $false; VersionPass = $false }

    if (-not (Test-Path $DllPath)) {
        Write-Host "  [$Label] CERTIFICATE FAIL: DLL not found" -ForegroundColor Red
        Write-Host "  [$Label] VERSION FAIL: DLL not found" -ForegroundColor Red
        return $result
    }

    # Certificate check
    $signature = Get-AuthenticodeSignature -FilePath $DllPath
    if ($signature.Status -ne 'Valid') {
        Write-Host "  [$Label] CERTIFICATE FAIL: Invalid signature ($($signature.Status))" -ForegroundColor Red
    } elseif ($signature.SignerCertificate.Subject -like "*QUALCOMM INCORPORATED*") {
        Write-Host "  [$Label] CERTIFICATE PASS" -ForegroundColor Green
        $result.CertPass = $true
    } else {
        Write-Host "  [$Label] CERTIFICATE FAIL: Not signed by QUALCOMM INCORPORATED (Subject: $($signature.SignerCertificate.Subject))" -ForegroundColor Red
    }

    # Version check — normalise to major.minor.patch to handle both
    # "2.5.0" (native DLLs) and "2.5.0.0" (managed DLL) against the
    # same ExpectedVersion string (e.g. "2.5.0" or "2.5.0.rc3").
    $rawVersion = [System.Diagnostics.FileVersionInfo]::GetVersionInfo($DllPath).FileVersion
    $normFile   = Normalize-Version $rawVersion
    $normExp    = Normalize-Version $ExpectedVersion

    if ($normFile -eq $normExp) {
        Write-Host "  [$Label] VERSION PASS: $rawVersion" -ForegroundColor Green
        $result.VersionPass = $true
    } else {
        Write-Host "  [$Label] VERSION FAIL: Expected $ExpectedVersion, got $rawVersion" -ForegroundColor Red
    }

    return $result
}

function Expand-Artifact([string]$ArtifactPath, [string]$ExtractDir) {
    # .whl and .nupkg are zip archives — copy to .zip first so Expand-Archive works.
    $ext = [System.IO.Path]::GetExtension($ArtifactPath).ToLower()
    if ($ext -eq '.zip') {
        Expand-Archive -Path $ArtifactPath -DestinationPath $ExtractDir -Force
    } else {
        $zipPath = Join-Path ([System.IO.Path]::GetDirectoryName($ArtifactPath)) `
                             "$([System.IO.Path]::GetFileNameWithoutExtension($ArtifactPath)).zip"
        Copy-Item -Path $ArtifactPath -Destination $zipPath -Force
        try {
            Expand-Archive -Path $zipPath -DestinationPath $ExtractDir -Force
        } finally {
            Remove-Item -Path $zipPath -Force -ErrorAction SilentlyContinue
        }
    }
}

# =============================================================================
# Verify source directory
# =============================================================================

if (-not (Test-Path $SourceDirectory -PathType Container)) {
    Write-Host ""
    Write-Error "Directory not found: $SourceDirectory"
    exit 1
}

# =============================================================================
# Per-artifact-type verification
# =============================================================================

$certPassCount    = 0
$certFailCount    = 0
$versionPassCount = 0
$versionFailCount = 0
$nuspecPass        = $null   # only set for nuget

switch ($ArtifactType) {

    "wheel" {
        $arm64Wheels = @(Get-ChildItem -Path $SourceDirectory -Recurse -Filter "*.whl" |
            Where-Object { $_.Name -match "win_arm64\.whl$" })
        $amdWheels   = @(Get-ChildItem -Path $SourceDirectory -Recurse -Filter "*.whl" |
            Where-Object { $_.Name -match "win_amd64\.whl$" })

        if (($arm64Wheels.Count + $amdWheels.Count) -eq 0) {
            Write-Host ""
            Write-Error "No wheels found matching win_amd64.whl or win_arm64.whl"
            exit 1
        }

        Write-Host ""
        Write-Host "Found $($arm64Wheels.Count) ARM64 wheel(s) and $($amdWheels.Count) AMD wheel(s)" -ForegroundColor Cyan

        # --- ARM64 wheels: 1 DLL per wheel ---
        foreach ($wheel in $arm64Wheels) {
            Write-Host ""
            Write-Host "Processing: $($wheel.Name)" -ForegroundColor Yellow

            $extractDir = Join-Path $wheel.DirectoryName "$($wheel.BaseName)_extracted"
            try {
                Expand-Artifact $wheel.FullName $extractDir

                $dll = Get-ChildItem -Path $extractDir -Recurse -Filter "onnxruntime_providers_qnn.dll" |
                    Select-Object -First 1
                $dllPath = if ($dll) { $dll.FullName } else { "" }

                $r = Test-QnnDll -DllPath $dllPath -Label "arm64" -ExpectedVersion $ExpectedVersion
                if ($r.CertPass)    { $certPassCount++ }    else { $certFailCount++ }
                if ($r.VersionPass) { $versionPassCount++ } else { $versionFailCount++ }
            }
            catch {
                Write-Host "  [arm64] ERROR: $($_.Exception.Message)" -ForegroundColor Red
                $certFailCount++
                $versionFailCount++
            }
            finally {
                Remove-Item -Path $extractDir -Recurse -Force -ErrorAction SilentlyContinue
            }
        }

        # --- AMD64 wheels: 2 DLLs per wheel (amd64 + arm64ec), AND aggregation ---
        foreach ($wheel in $amdWheels) {
            Write-Host ""
            Write-Host "Processing: $($wheel.Name)" -ForegroundColor Yellow

            $extractDir = Join-Path $wheel.DirectoryName "$($wheel.BaseName)_extracted"

            $amd64CertPass      = $false
            $amd64VersionPass   = $false
            $arm64ecCertPass    = $false
            $arm64ecVersionPass = $false

            try {
                Expand-Artifact $wheel.FullName $extractDir

                $allDlls    = Get-ChildItem -Path $extractDir -Recurse -Filter "onnxruntime_providers_qnn.dll"
                $amd64Dll   = $allDlls | Where-Object { $_.FullName -match "[\\/]libs[\\/]amd64[\\/]"   } | Select-Object -First 1
                $arm64ecDll = $allDlls | Where-Object { $_.FullName -match "[\\/]libs[\\/]arm64ec[\\/]" } | Select-Object -First 1

                # libs/amd64/onnxruntime_providers_qnn.dll
                $amd64Path = if ($amd64Dll) { $amd64Dll.FullName } else { "" }
                $r = Test-QnnDll -DllPath $amd64Path -Label "amd64  " -ExpectedVersion $ExpectedVersion
                $amd64CertPass    = $r.CertPass
                $amd64VersionPass = $r.VersionPass

                # libs/arm64ec/onnxruntime_providers_qnn.dll
                $arm64ecPath = if ($arm64ecDll) { $arm64ecDll.FullName } else { "" }
                $r = Test-QnnDll -DllPath $arm64ecPath -Label "arm64ec" -ExpectedVersion $ExpectedVersion
                $arm64ecCertPass    = $r.CertPass
                $arm64ecVersionPass = $r.VersionPass
            }
            catch {
                Write-Host "  ERROR: $($_.Exception.Message)" -ForegroundColor Red
            }
            finally {
                Remove-Item -Path $extractDir -Recurse -Force -ErrorAction SilentlyContinue
            }

            # Wheel-level verdict: both DLLs must pass for the wheel to pass
            if ($amd64CertPass -and $arm64ecCertPass) {
                $certPassCount++
            } else {
                $certFailCount++
            }

            if ($amd64VersionPass -and $arm64ecVersionPass) {
                $versionPassCount++
            } else {
                $versionFailCount++
            }
        }
    }

    "zip" {
        $zipArchs = @("win-arm64", "win-arm64x", "win-x64")

        foreach ($arch in $zipArchs) {
            $zips = @(Get-ChildItem -Path $SourceDirectory -Filter "*.zip" |
                Where-Object { $_.Name -match "${arch}\.zip$" })

            if ($zips.Count -ne 1) {
                Write-Host ""
                Write-Host "Expected 1 ${arch} zip in $SourceDirectory, found $($zips.Count)" -ForegroundColor Red
                $certFailCount++
                $versionFailCount++
                continue
            }

            $zip = $zips[0]
            Write-Host ""
            Write-Host "Processing [$arch]: $($zip.Name)" -ForegroundColor Yellow

            $extractDir = Join-Path $zip.DirectoryName "$($zip.BaseName)_extracted"

            try {
                Expand-Archive -Path $zip.FullName -DestinationPath $extractDir -Force

                $dll = Get-ChildItem -Path $extractDir -Recurse -Filter "onnxruntime_providers_qnn.dll" |
                    Select-Object -First 1
                $dllPath = if ($dll) { $dll.FullName } else { "" }

                $r = Test-QnnDll -DllPath $dllPath -Label $arch -ExpectedVersion $ExpectedVersion
                if ($r.CertPass)    { $certPassCount++ }    else { $certFailCount++ }
                if ($r.VersionPass) { $versionPassCount++ } else { $versionFailCount++ }
            }
            catch {
                Write-Host "  [$arch] ERROR: $($_.Exception.Message)" -ForegroundColor Red
                $certFailCount++
                $versionFailCount++
            }
            finally {
                Remove-Item -Path $extractDir -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
    }

    "nuget" {
        $nupkg = Get-ChildItem -Path $SourceDirectory -Filter "*.nupkg" | Select-Object -First 1
        if (-not $nupkg) {
            Write-Host ""
            Write-Host "ERROR: No .nupkg found in $SourceDirectory" -ForegroundColor Red
            exit 1
        }

        Write-Host ""
        Write-Host "Processing: $($nupkg.Name)" -ForegroundColor Yellow

        $extractDir = Join-Path $nupkg.DirectoryName "$($nupkg.BaseName)_extracted"
        $nuspecPass = $false

        try {
            Expand-Artifact $nupkg.FullName $extractDir

            # --- Nuspec version check ---
            Write-Host ""
            Write-Host "--- nuspec ---" -ForegroundColor Cyan
            $nuspecFile = Get-ChildItem -Path $extractDir -Filter "*.nuspec" | Select-Object -First 1
            if (-not $nuspecFile) {
                Write-Host "  NUSPEC FAIL: .nuspec not found in package" -ForegroundColor Red
            } else {
                [xml]$nuspec = Get-Content $nuspecFile.FullName -Encoding UTF8
                $nuspecVersion = $nuspec.package.metadata.version
                if ($nuspecVersion -eq $ExpectedVersion) {
                    Write-Host "  NUSPEC VERSION PASS: $nuspecVersion" -ForegroundColor Green
                    $nuspecPass = $true
                } else {
                    Write-Host "  NUSPEC VERSION FAIL: Expected $ExpectedVersion, got $nuspecVersion" -ForegroundColor Red
                }
            }

            # --- DLL checks ---
            $dllsToCheck = @(
                @{ Label = "managed (netstandard2.0)"; RelPath = "lib\netstandard2.0\Qualcomm.ML.OnnxRuntime.QNN.dll" },
                @{ Label = "native win-arm64";         RelPath = "runtimes\win-arm64\native\onnxruntime_providers_qnn.dll" },
                @{ Label = "native win-x64";           RelPath = "runtimes\win-x64\native\onnxruntime_providers_qnn.dll" }
            )

            foreach ($entry in $dllsToCheck) {
                Write-Host ""
                Write-Host "--- $($entry.Label) ---" -ForegroundColor Cyan

                $dllPath = Join-Path $extractDir $entry.RelPath
                $r = Test-QnnDll -DllPath $dllPath -Label $entry.Label -ExpectedVersion $ExpectedVersion
                if ($r.CertPass)    { $certPassCount++ }    else { $certFailCount++ }
                if ($r.VersionPass) { $versionPassCount++ } else { $versionFailCount++ }
            }
        }
        catch {
            Write-Host ""
            Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
        }
        finally {
            Remove-Item -Path $extractDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

# =============================================================================
# Summary
# =============================================================================

Write-Host ""
if ($null -ne $nuspecPass) {
    Write-Host "=== Nuspec Summary ===" -ForegroundColor Cyan
    if ($nuspecPass) {
        Write-Host "Nuspec version: PASS" -ForegroundColor Green
    } else {
        Write-Host "Nuspec version: FAIL" -ForegroundColor Red
    }
}
Write-Host "=== Certificate Summary ===" -ForegroundColor Cyan
Write-Host "Total:  $($certPassCount + $certFailCount)"
Write-Host "Passed: $certPassCount" -ForegroundColor Green
Write-Host "Failed: $certFailCount" -ForegroundColor Red
Write-Host "=== Version Summary ===" -ForegroundColor Cyan
Write-Host "Total:  $($versionPassCount + $versionFailCount)"
Write-Host "Passed: $versionPassCount" -ForegroundColor Green
Write-Host "Failed: $versionFailCount" -ForegroundColor Red
Write-Host "=== End of Summary ===" -ForegroundColor Cyan

$failed = $certFailCount -gt 0 -or $versionFailCount -gt 0
if ($null -ne $nuspecPass -and -not $nuspecPass) { $failed = $true }
if ($failed) { exit 1 }
