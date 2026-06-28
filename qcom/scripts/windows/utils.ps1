# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT


function Assert-Success() {
    param(
        [scriptblock]$Code,
        [Parameter(Mandatory = $false)]
        [string]$ErrorMessage = "Execution failed"
    )
    Invoke-Command -ScriptBlock $Code

    # This has some limitations. In particular, not every command indicates error
    # by $LASTEXITCODE other than 0. This is especially true of built-in commands
    # such as New-Item, but also some native things like robocopy. Still, we choose
    # to use $LASTEXITCODE because Invoke-Command does not propogate the success
    # of the command it invoked.
    if ($LASTEXITCODE -ne 0) {
        throw $ErrorMessage
    }
}

function Enter-MsvcEnv() {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TargetArch
    )

    switch ($TargetArch) {
        "arm64" { $MsvcArch = "arm64" }
        "x86_64" { $MsvcArch = "amd64"}
        default { throw "Unknown target arch $TargetArch." }
    }

    $VsInstall = Get-InstalledVsGenerator
    & $VsInstall.DevShell -Arch $MsvcArch -SkipAutomaticLocation

    if (-not $?) {
        throw "Could not activate MSVC environment for target arch $TargetArch"
    }
}

function Enter-PyVenv() {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PyVEnv
    )

    if ($env:VIRTUAL_ENV) {
        $LoadedVenv = (Resolve-Path "$env:VIRTUAL_ENV\").Path
        $DesiredVenv = (Resolve-Path "$PyVEnv\").Path

        if ($LoadedVenv -ne $DesiredVenv) {
            throw "Refusing to activate different Python venv ($LoadedVenv vs $DesiredVenv)."
        }
        else {
            Write-Host "Python venv $PyVEnv already activated."
        }
    }
    else {
        Write-Host "Activating Python venv $PyVEnv."
        . (Join-Path $PyVEnv "Scripts\Activate.ps1")
    }
}

function Exit-PyVenv() {
    if (-not $env:VIRTUAL_ENV) {
        throw "Cannot deactivate: no virtual environment is active."
    }

    # If we simply deactivate, any changes we've made to $env:PATH since we activated will be lost.

    # Figure out what $env:Path should be
    $PathNoVenv = (($env:Path.Split(";") | Where-Object { $_ -ne "${env:VIRTUAL_ENV}\Scripts"}) -join ";")

    deactivate

    $env:Path = $PathNoVenv
}

# Enumerate installed Visual Studio roots, newest first. Single source of truth for the
# supported VS versions and editions consumed by Get-InstalledVsGenerator and Get-VcRedistDir.
# Each result carries the install Dir token, the CMake Generator name, and the install Root path.
function Get-VsInstallRoots() {
    $VsInstalls = @(
        @{ Dir = "18"; Generator = "Visual Studio 18 2026" },
        @{ Dir = "2022"; Generator = "Visual Studio 17 2022" }
    )
    $Editions = @("Enterprise", "Professional", "Community", "Preview", "BuildTools")
    $Roots = @()
    foreach ($vs in $VsInstalls) {
        foreach ($edition in $Editions) {
            $Root = "$env:ProgramW6432\Microsoft Visual Studio\$($vs.Dir)\$edition"
            if (Test-Path $Root) {
                $Roots += [PSCustomObject]@{ Dir = $vs.Dir; Generator = $vs.Generator; Root = $Root }
            }
        }
    }
    return $Roots
}

function Get-InstalledVsGenerator() {
    foreach ($vs in (Get-VsInstallRoots)) {
        $DevShell = "$($vs.Root)\Common7\Tools\Launch-VsDevShell.ps1"
        if (Test-Path $DevShell) {
            return [PSCustomObject]@{ Generator = $vs.Generator; DevShell = $DevShell }
        }
    }
    throw "No supported Visual Studio installation found (2026 or 2022)."
}

# Locate the MSVC redistributable directory (VCToolsRedistDir). archive_tests.py uses this
# to bundle msvcp140*.dll and vcruntime140*.dll into the Windows test archive so QA machines
# don't need vc_redist preinstalled to run tests that load onnxruntime.dll.
function Get-VcRedistDir() {
    # Trailing backslashes must be stripped: Launch-VsDevShell.ps1 sets $env:VCToolsRedistDir
    # with a trailing '\'. When the value is then passed through to python.exe as a quoted
    # argument, Windows' CRT argv parser interprets the final '\"' as an escaped quote (per
    # CommandLineToArgvW rules), embedding a stray '"' into the value. The downstream Path
    # join then yields ...\14.42.34433"\x64 and the arch dir lookup fails.
    if ($env:VCToolsRedistDir -and (Test-Path $env:VCToolsRedistDir)) {
        return (Resolve-Path $env:VCToolsRedistDir).Path.TrimEnd('\')
    }

    # Fall back to the VS install's pinned redist version for cross-compile builds (VS
    # generator) that never ran Launch-VsDevShell.ps1 in this session.
    foreach ($vs in (Get-VsInstallRoots)) {
        $VersionFile = "$($vs.Root)\VC\Auxiliary\Build\Microsoft.VCRedistVersion.default.txt"
        if (Test-Path $VersionFile) {
            $Version = (Get-Content $VersionFile -Raw).Trim()
            $RedistDir = "$($vs.Root)\VC\Redist\MSVC\$Version"
            if (Test-Path $RedistDir) {
                return (Resolve-Path $RedistDir).Path.TrimEnd('\')
            }
        }
    }
    throw "Could not locate VC redist directory (VCToolsRedistDir unset and no VS install found)."
}

function Get-DefaultCMakeGenerator() {
    param (
        [Parameter(Mandatory = $true)]
        [string]$Arch
    )
    $HostArch = (Get-HostArch)
    # It's entirely possible that $Arch is "arm64ec" and $HostArch is "arm64".
    # Unfortunately, Launch-VsDevShell.ps1 doesn't support arm64ec so we cannot
    # use Ninja.
    if ($Arch -eq $HostArch) {
        "Ninja"
    } else {
        Write-Host "Cross compiling for $Arch on $HostArch host. Cannot use Ninja."
        (Get-InstalledVsGenerator).Generator
    }
}

function Get-HostArch() {
    # PROCESSOR_ARCHITEW6432 is set on WOW64 / x64-emulated processes and reports the
    # real host arch. Fall back to machine-scope PROCESSOR_ARCHITECTURE (locale-independent).
    $arch = [System.Environment]::GetEnvironmentVariable("PROCESSOR_ARCHITEW6432", "Process")
    if ([string]::IsNullOrEmpty($arch)) {
        $arch = [System.Environment]::GetEnvironmentVariable("PROCESSOR_ARCHITECTURE", "Machine")
    }
    switch ($arch) {
        "ARM64" { "arm64" }
        "AMD64" { "x86_64" }
        default { throw "Unknown OS Architecture $arch." }
    }
}

function Get-QairtSdkFilePath() {
    param (
        [Parameter(Mandatory = $true)]
        [string]$BuildDir,
        [Parameter(Mandatory = $true)]
        [string]$Config
    )
    "$BuildDir\qairt-sdk-path-$Config.txt"
}

function Get-QairtSdkVersion() {
    param(
        [Parameter(Mandatory = $true)]
        [string]$QairtSdkRoot
    )

    (Select-String `
        -Path (Join-Path $QairtSdkRoot "sdk.yaml") `
        -Pattern "^version: (\d+\.\d+\.\d+)" `
    ).Matches[0].Groups[1].Value
}

function Get-TargetPyVersionFilePath() {
    param (
        [Parameter(Mandatory = $true)]
        [string]$BuildDir,
        [Parameter(Mandatory = $true)]
        [string]$Config
    )
    "$BuildDir\target-py-version-$Config.txt"
}

function Save-QairtSdkFilePath() {
    param (
        [Parameter(Mandatory = $true)]
        [string]$BuildDir,
        [Parameter(Mandatory = $true)]
        [string]$Config
    )

    $SdkFilePath = (Get-QairtSdkFilePath -BuildDir $BuildDir -Config $Config)
    if (-Not (Test-Path "$SdkFilePath\..")) {
        New-Item -Path "$SdkFilePath\.." -ItemType Directory | Out-Null
    }
    $QairtSdkRoot | Out-File -FilePath $SdkFilePath
}

function Save-TargetPyVersion() {
    param (
        [Parameter(Mandatory = $true)]
        [string]$BuildDir,
        [Parameter(Mandatory = $true)]
        [string]$Config,
        [Parameter(Mandatory = $false)]
        [string]$TargetPyVersion = ""
    )

    $TargetPyVersionFilePath = (Get-TargetPyVersionFilePath -BuildDir $BuildDir -Config $Config)
    if (-Not (Test-Path "$TargetPyVersionFilePath\..")) {
        New-Item -Path "$TargetPyVersionFilePath\.." -ItemType Directory | Out-Null
    }
    $TargetPyVersion | Out-File -FilePath $TargetPyVersionFilePath
}

function Test-QairtSdkDiffers() {
    param (
        [Parameter(Mandatory = $true)]
        [string]$BuildDir,
        [Parameter(Mandatory = $true)]
        [string]$Config,
        [Parameter(Mandatory = $true)]
        [string]$QairtSdkRoot
    )

    $QairtSdkPathPath = (Get-QairtSdkFilePath -BuildDir $BuildDir -Config $Config)
    if (-Not (Test-Path -Path $QairtSdkPathPath)) {
        return $True
    }

    $LastSdkPath = Get-Content -Path $QairtSdkPathPath
    return $LastSdkPath -ne $QairtSdkRoot
}

function Test-TargetPyVersionDiffers() {
    param (
        [Parameter(Mandatory = $true)]
        [string]$BuildDir,
        [Parameter(Mandatory = $true)]
        [string]$Config,
        [Parameter(Mandatory = $false)]
        [string]$TargetPyVersion = ""
    )

    $TargetPyVersionFilePath = (Get-TargetPyVersionFilePath -BuildDir $BuildDir -Config $Config)
    if (-Not (Test-Path -Path $TargetPyVersionFilePath)) {
        return $True
    }

    $LastTargetPyVersion = Get-Content -Path $TargetPyVersionFilePath
    return $LastTargetPyVersion -ne $TargetPyVersion
}

function Test-UpdateNeeded() {
    param (
        [Parameter(Mandatory = $true)]
        [string]$BuildDir,
        [Parameter(Mandatory = $true)]
        [string]$Config,
        [Parameter(Mandatory = $false)]
        [string]$TargetPyVersion = "",
        [Parameter(Mandatory = $true)]
        [string]$QairtSdkRoot,
        [Parameter(Mandatory = $true)]
        [string]$CMakeGenerator,
        [Parameter(Mandatory = $true)]
        [bool]$Update
    )

    if ($Update) {
        Write-Host "Build system update was requested."
        return $True
    }

    if ($CMakeGenerator -eq "Ninja") {
        $BuildNinjaPath = "$BuildDir\$Config\build.ninja"
        if (-Not (Test-Path -Path $BuildNinjaPath)) {
            Write-Host "$BuildNinjaPath does not exist."
            return $True
        }
    } else {
        $SlnPath = "$BuildDir\$Config\onnxruntime_qnn.sln"
        if (-Not (Test-Path -Path $SlnPath)) {
            Write-Host "VS Solution $SlnPath does not exist."
            return $True
        }
    }

    if (Test-TargetPyVersionDiffers -BuildDir $BuildDir -Config $Config -TargetPyVersion $TargetPyVersion) {
        Write-Host "Previous build used a different Python version."
        return $True
    }

    if (Test-QairtSdkDiffers -BuildDir $BuildDir -Config $Config -QairtSdkRoot $QairtSdkRoot) {
        Write-Host "Previous build used a different QAIRT SDK."
        return $True
    }

    Write-Host "No need to update build system."
    return $False
}

function Use-PyVEnv() {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PyVEnv,
        [scriptblock]$Code
    )

    if ($env:VIRTUAL_ENV) {
        $PrevVenv = $env:VIRTUAL_ENV
        Exit-PyVenv
    }

    try {
        Enter-PyVenv -PyVEnv $PyVEnv
        Invoke-Command $Code
    }
    finally {
        Exit-PyVenv
        if ($null -ne $PrevVenv) {
            Enter-PyVenv $PrevVenv
        }
    }
}

function Use-WorkingDir {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [scriptblock]$Code
    )

    Push-Location $Path
    try {
        Invoke-Command $Code
    }
    finally {
        Pop-Location
    }
}
