# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

param (
    [Parameter(Mandatory = $true,
               HelpMessage = "The architecture for which to build.")]
    [ValidateSet("aarch64", "arm64", "arm64ec", "x86_64")]
    [string]$Arch,

    [Parameter(Mandatory = $false,
               HelpMessage = "If true, build for ARM64x.")]
    [bool]$BuildAsX = $false,

    [Parameter(Mandatory = $false,
               HelpMessage = "Path to QAIRT SDK.")]
    [string]$QairtSdkRoot,

    [Parameter(Mandatory = $false,
               HelpMessage = "What to do: build|archive|test|generate_sln.")]
    [ValidateSet("build", "archive", "test", "generate_sln")]
    [string]$Mode = "build",

    [Parameter(Mandatory = $false,
               HelpMessage = "The configuration to build.")]
    [ValidateSet("Debug", "Release", "RelWithDebInfo")]
    [string]$Config = "Release",

    [Parameter(Mandatory = $false,
               HelpMessage = "Force regeneration of build system.")]
    [bool]$Update = $false,

    [Parameter(Mandatory = $false,
               HelpMessage = "Build a wheel targeting this Python version.")]
    [ValidateSet("", "3.11", "3.12", "3.13")]
    [string]$TargetPyVersion = "",

    [Parameter(Mandatory = $true,
               HelpMessage = "Python virtual environment to activate.")]
    [string]$PyVEnv
)

function Get-QairtSdkRoot() {
    if ($QairtSdkRoot -eq "") {
        return Get-QairtRoot
    }
    else {
        return Resolve-Path -Path $QairtSdkRoot
    }
}

$RepoRoot = (Resolve-Path -Path "$(Split-Path -Parent $MyInvocation.MyCommand.Definition)\..\..\..").Path

. "$RepoRoot\qcom\scripts\windows\tools.ps1"
. "$RepoRoot\qcom\scripts\windows\utils.ps1"

$BuildRoot = (Join-Path $RepoRoot "build")
$BuildDirArch = $Arch

if ($Mode -eq "generate_sln") {
    $BuildDir = (Join-Path $BuildRoot "vs")
}
else {
    if ($BuildAsX) {
        switch ($Arch) {
            "ARM64" { $BuildDirArch = "arm64-x-slice" }
            "ARM64ec" { $BuildDirArch = "arm64x" }
            Default { throw "Invalid arch $Arch for ARM64x" }
        }
    }
    $BuildDir = (Join-Path $BuildRoot "windows-$BuildDirArch")
}

if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

Enter-PyVenv $PyVEnv

$BuildIsDirty = $false
$CMakeGenerator = (Get-DefaultCMakeGenerator -Arch $Arch)

if ($Mode -eq "generate_sln") {
    $CMakeGenerator = "Visual Studio 17 2022"
    $BuildIsDirty = $true
} elseif ($Mode -eq "build") {
    if (Test-UpdateNeeded -BuildDir $BuildDir -Config $Config `
            -TargetPyVersion $TargetPyVersion -QairtSdkRoot $(Get-QairtSdkRoot) `
            -CMakeGenerator $CMakeGenerator -Update $Update) {
        $BuildIsDirty = $true
        Save-QairtSdkFilePath -BuildDir $BuildDir -Config $Config
        Save-TargetPyVersion -BuildDir $BuildDir -Config $Config -TargetPyVersion $TargetPyVersion
    }
}

$GenerateBuild = $false
$DoBuild = $false
$BuildWheel = $false
$MakeTestArchive = $false
$RunTests = $false
$TestRunner = "$RepoRoot\qcom\scripts\windows\run_tests.ps1"

switch ($Mode) {
    "build" {
        if ($BuildIsDirty) {
            $GenerateBuild = $true
        }

        $DoBuild = $true
    }
    "generate_sln" {
        $GenerateBuild = $true
    }
    "test" {
        $RunTests = $true
    }
    "archive" {
        $MakeTestArchive = $true
    }
    default {
        throw "Unknown build mode $Mode."
    }
}

$ArchArgs = @()
$CommonArgs = `
    "--build_dir", $BuildDir, `
    "--build_shared_lib", `
    "--cmake_generator", $CMakeGenerator, `
    "--config", $Config, `
    "--parallel"

$TargetPyExe = $null
$BuildDepsWheelHouse = $null
if ($TargetPyVersion -ne "")
{
    # Wheels only supported when we can run Python for the target arch.
    $TargetPyExe = (Join-Path (Get-PythonBinDir -Version $TargetPyVersion -Arch $Arch) "python.exe")
    $BuildWheel = $true
    $ArchArgs += "--enable_pybind"
    $BuildVEnv = (Join-Path $BuildDir "venv-$TargetPyVersion")
    $BuildDepsWheelHouse = (Join-Path $BuildDir "wheels-$TargetPyVersion")
    $TestPyExe = (Join-Path (Join-Path $BuildVEnv "Scripts") "python.exe")
    Write-Host "Building Python wheel using $TargetPyExe"
}
else {
    $BuildVEnv = $PyVEnv
    $TestPyExe = $null
    Write-Host "Not building a Python wheel"
}

if ($BuildAsX) {
    $CommonArgs += "--buildasx"
}

# The ORT build incorrectly enables use of Kleidiai when using Ninja on Windows,
# even if ArmNN is not requested. Manually turn it off.
$PlatformArgs = @("--no_kleidiai")

$CmakeBinDir = (Get-CMakeBinDir)
$env:Path = "$CmakeBinDir;" + $env:Path

if ($null -eq $env:ORT_BUILD_PRUNE_PACKAGES -or 1 -eq $env:ORT_BUILD_PRUNE_PACKAGES) {
    Optimize-ToolsDir
}

Push-Location $RepoRoot

$failed = $false
if ($MakeTestArchive) {
    python.exe "$RepoRoot\qcom\scripts\all\archive_tests.py" `
        "--config=$Config" `
        "--qairt-sdk-root=$(Get-QairtSdkRoot)" `
        "--target-platform=windows-$BuildDirArch"
    if (-not $?) {
        $failed = $true
    }
}
else {
    if ($GenerateBuild -or $DoBuild) {
        # Don't miss the cache due to __TIME__, __DATE__, or __TIMESTAMP__.
        $env:CCACHE_SLOPPINESS = "time_macros"
        if ($CMakeGenerator -eq "Ninja") {
            $env:Path = "$(Get-NinjaBinDir);$(Get-CCacheBinDir);" + $env:Path
            $env:Path = "$(Get-CCacheBinDir);" + $env:Path

            # The default somehow gives us paths that are too long in CI
            $PlatformArgs += "--cmake_extra_defines", "CMAKE_OBJECT_PATH_MAX=240"

            # We don't have Visual Studio to set up the build environment so do it
            # manually with somthing akin to vcvarsall.bat.
            Enter-MsvcEnv -TargetArch $Arch
        }
        else {
            # Tell the EP build that we're cross-compiling to ARM64.
            # We do not do this when using Ninja because our fake vcvars handles
            # cross-compilation flags.
            $ArchArgs += "--$Arch"

            # https://github.com/ccache/ccache/wiki/MS-Visual-Studio#usage-with-cmake
            Assert-Success -ErrorMessage "Failed to copy ccache.exe to $BuildDir\cl.exe" {
                Copy-Item "$(Get-CCacheBinDir)\ccache.exe" "$BuildDir\cl.exe"
            }
            $FakeClCcacheDir = $BuildDir.Replace("\", "/")
            $CommonArgs += `
                "--cmake_extra_defines", "CMAKE_VS_GLOBALS=CLToolExe=cl.exe;CLToolPath=$FakeClCcacheDir;UseMultiToolTask=true", `
                "--cmake_extra_defines", 'CMAKE_MSVC_DEBUG_INFORMATION_FORMAT=$"<"$"<"CONFIG:Debug,RelWithDebInfo">":Embedded">"'
        }
    }

    if (-not (Test-Path $BuildVEnv)) {
        Assert-Success -ErrorMessage "Failed to create build virtual environment" {
            & $TargetPyExe -m venv $BuildVEnv
        }
    }

    Use-PyVenv -PyVenv $BuildVEnv {
        Assert-Success { python.exe -m pip install uv }
        Assert-Success {
            $FindLinks = @()
            if ($null -ne $BuildDepsWheelHouse -and (Test-Path $BuildDepsWheelHouse)) {
                $FindLinks += "--find-links", $BuildDepsWheelHouse
            }
            uv.exe pip install -r "$RepoRoot\tools\ci_build\github\windows\python\requirements.txt" --native-tls $FindLinks
        }
    }

    try {
        if ($GenerateBuild -or $DoBuild) {
            python.exe "$RepoRoot\qcom\scripts\all\fetch_cmake_deps.py"
        }
        if ($GenerateBuild) {
            $QnnArgs = "--use_qnn", "--qnn_home", (Get-QairtSdkRoot)

            Use-PyVenv -PyVenv $BuildVEnv {
                Assert-Success -ErrorMessage "Failed to generate build" {
                    .\build.bat --update $ArchArgs $CommonArgs $QnnArgs $PlatformArgs
                }
            }
        }
        if ($DoBuild) {
            Assert-Success -ErrorMessage "Failed to build" {
                & cmake --build (Join-Path $BuildDir $Config) --config $Config
            }

            if ($BuildWheel) {
                $BuildOutputDir = (Join-Path $BuildDir $Config)
                if ($CMakeGenerator -eq "Visual Studio 17 2022") {
                    $BuildOutputDir = (Join-Path $BuildOutputDir $Config)
                }

                if ($env:ORT_NIGHTLY_BUILD) {
                    $PyNightlyArg = "--nightly_build"
                }
                Use-PyVenv -PyVenv $BuildVEnv {
                    Use-WorkingDir -Path $BuildOutputDir {
                        Assert-Success -ErrorMessage "Failed to build wheel" {
                            python.exe (Join-Path $RepoRoot "setup.py") `
                                bdist_wheel --wheel_name_suffix=qnn_qcom_internal $PyNightlyArg
                        }
                    }
                }
            }
        }
    }
    finally {
        # Whatever happens, blow away mirror to avoid it showing up in git; it's okay, it's
        # very cheap to regenerate.
        if (Test-Path (Join-Path $RepoRoot "mirror")) {
            Remove-Item -Recurse -Force (Join-Path $RepoRoot "mirror")
        }
    }

    if ($RunTests) {
        Push-Location (Join-Path $BuildDir $Config)
        $OnnxModelsRoot = (Get-OnnxModelsRoot)
        & $TestRunner -Config $Config -Arch $BuildDirArch -PyExePath $TestPyExe -OnnxModelsRoot $OnnxModelsRoot

        if (-not $?) {
            $failed = $true
        }
    }
}

if ($failed) {
    throw "Build failure"
}

Pop-Location
