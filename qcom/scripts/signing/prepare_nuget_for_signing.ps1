# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
param(
    [Parameter(Mandatory=$true)]
    [string]$NuGetDirectory,
    [Parameter(Mandatory=$true)]
    [string]$OutputDirectory
)

# Verify the directories exist or create output directory
if (-not (Test-Path $NuGetDirectory -PathType Container)) {
    Write-Host ""
    Write-Error "Directory not found: $NuGetDirectory"
    exit 1
}

if (-not (Test-Path $OutputDirectory -PathType Container)) {
    Write-Host ""
    New-Item -Path $OutputDirectory -ItemType Directory -Force | Out-Null
}

# Create unsigned/nuget subdirectories
$unsignedDir = Join-Path $OutputDirectory "unsigned_libs"
$nugetDir = Join-Path $unsignedDir "nuget"

if (-not (Test-Path $nugetDir -PathType Container)) {
    New-Item -Path $nugetDir -ItemType Directory -Force | Out-Null
}

$OutputDirectory = $nugetDir

# Find all NuGet packages matching the pattern
$nugetPackages = Get-ChildItem -Path $NuGetDirectory -Filter "*.nupkg"

if ($nugetPackages.Count -eq 0) {
    Write-Host ""
    Write-Error "No NuGet packages found matching *.nupkg"
    exit 1
}

Write-Host ""
Write-Host "Found $($nugetPackages.Count) NuGet package(s)" -ForegroundColor Cyan
Write-Host ""

$processedPackages = @()
$failedPackages = 0
$failedPackageNames = @()
$totalPackages = $nugetPackages.Count

# Process each NuGet package
foreach ($package in $nugetPackages) {
    Write-Host "Processing: $($package.Name)" -ForegroundColor Yellow

    $packageBaseName = $package.BaseName
    $zipPath = Join-Path $OutputDirectory "$packageBaseName.zip"
    $tempExtractDir = Join-Path $OutputDirectory "$packageBaseName`_temp"
    $finalExtractDir = Join-Path $OutputDirectory "$packageBaseName"

    try {
        # Copy NuGet package and rename to .zip
        Copy-Item -Path $package.FullName -Destination $zipPath -Force

        # Extract the zip to temporary directory
        Expand-Archive -Path $zipPath -DestinationPath $tempExtractDir -Force

        # Create final extraction directory
        if (-not (Test-Path $finalExtractDir)) {
            New-Item -Path $finalExtractDir -ItemType Directory -Force | Out-Null
        }

        # The architecture-neutral managed helper assembly is present in every package.
        $managedDll = @{
            Source = "lib\netstandard2.0\Qualcomm.ML.OnnxRuntime.QNN.dll"
            Dest   = "Qualcomm.ML.OnnxRuntime.QNN.dll"
        }

        # The native provider DLL lives under a per-architecture runtimes folder. A single-arch
        # package contains exactly one of these (arm64 OR x64). The arm64 and x64 copies share the
        # same filename but are different binaries, so each is extracted into an arch-named
        # subfolder to avoid colliding when both packages are staged into the same output dir.
        $providerDlls = @(
            "runtimes\win-arm64\native\onnxruntime_providers_qnn.dll",
            "runtimes\win-x64\native\onnxruntime_providers_qnn.dll"
        )

        $allDllsFound = $true

        # Managed assembly: required in every package.
        $managedSource = Join-Path $tempExtractDir $managedDll.Source
        if (Test-Path -LiteralPath $managedSource) {
            Copy-Item -LiteralPath $managedSource -Destination (Join-Path $finalExtractDir $managedDll.Dest) -Force
        }
        else {
            Write-Host "  ERROR: DLL not found at $($managedDll.Source)" -ForegroundColor Red
            $allDllsFound = $false
        }

        # Native provider assembly: extract whichever architecture(s) this package contains,
        # each into its own arch-named subfolder. At least one must be present.
        $providerFound = $false
        foreach ($providerPath in $providerDlls) {
            $sourceDllPath = Join-Path $tempExtractDir $providerPath

            if (Test-Path -LiteralPath $sourceDllPath) {
                # e.g. "runtimes\win-arm64\native\..." -> "win-arm64"
                $arch = Split-Path (Split-Path (Split-Path $providerPath -Parent) -Parent) -Leaf
                $archDir = Join-Path $finalExtractDir $arch
                if (-not (Test-Path $archDir)) {
                    New-Item -Path $archDir -ItemType Directory -Force | Out-Null
                }
                Copy-Item -LiteralPath $sourceDllPath -Destination (Join-Path $archDir "onnxruntime_providers_qnn.dll") -Force
                $providerFound = $true
            }
        }

        if (-not $providerFound) {
            Write-Host "  ERROR: onnxruntime_providers_qnn.dll not found under runtimes\win-arm64\native or runtimes\win-x64\native" -ForegroundColor Red
            $allDllsFound = $false
        }

        if (-not $allDllsFound) {
            Write-Host "  ERROR: Not all required DLLs found in package" -ForegroundColor Red
            Remove-Item -Path $tempExtractDir -Recurse -Force -ErrorAction SilentlyContinue
            Remove-Item -Path $finalExtractDir -Recurse -Force -ErrorAction SilentlyContinue
            $failedPackages++
            $failedPackageNames += $package.Name
            continue
        }

        # Clean up temporary extraction
        Remove-Item -Path $tempExtractDir -Recurse -Force -ErrorAction SilentlyContinue

        # Store package information for signing
        $packageInfo = @{
            PackageName = $package.Name
            OriginalPath = $package.FullName
            ExtractedPath = $finalExtractDir
            ZipPath = $zipPath
        }

        $processedPackages += $packageInfo
        Write-Host "  Ready for signing" -ForegroundColor Green

        # Delete the zip file
        Remove-Item -Path $zipPath -Force -ErrorAction SilentlyContinue
    }
    catch {
        Write-Host "  ERROR: Failed to process package - $($_.Exception.Message)" -ForegroundColor Red
        Remove-Item -Path $tempExtractDir -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item -Path $finalExtractDir -Recurse -Force -ErrorAction SilentlyContinue
        $failedPackages++
        $failedPackageNames += $package.Name
    }
}

Write-Host ""
Write-Host "=== Preparation Summary ===" -ForegroundColor Cyan
Write-Host "Total NuGet packages: $totalPackages"
Write-Host "Prepared for signing: $($processedPackages.Count)" -ForegroundColor Green
Write-Host "Failures: $failedPackages" -ForegroundColor Red
if ($failedPackageNames.Count -gt 0) {
    foreach ($failedName in $failedPackageNames) {
        Write-Host "  - $failedName" -ForegroundColor Red
    }
}

if ($failedPackages -ne 0) {
    Write-Host "Preparation failed: $failedPackages NuGet package(s) could not be processed" -ForegroundColor Red
    Write-Host "=== End of Summary ===" -ForegroundColor Cyan
    exit 1
}
Write-Host "Preparation successful: $($processedPackages.Count) NuGet package(s) prepared for signing" -ForegroundColor Green
Write-Host "=== End of Summary ===" -ForegroundColor Cyan

# Compress unsigned NuGet libs for signing
Write-Host ""
Write-Host "Compressing unsigned NuGet libs" -ForegroundColor Cyan
$nugetLibsZip = Join-Path $unsignedDir "nuget.zip"
try {
    Compress-Archive -Path "$nugetDir\*" -DestinationPath $nugetLibsZip -Force
    Write-Host "Successfully created nuget.zip" -ForegroundColor Green
    # Move nuget.zip to the output directory
    $outputZip = Join-Path $OutputDirectory "..\nuget.zip"
    Move-Item -Path $nugetLibsZip -Destination $outputZip -Force
}
catch {
    Write-Host "ERROR: Failed to create nuget.zip - $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
