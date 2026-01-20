
<#
Rebuild venv → Build project → Install dist artifacts → Set runtime PATH
Run this script from qcom/gen_ai (same location as original Bash version).
#>

$ErrorActionPreference = "Stop"

# ----- SETTINGS -----
$PythonVersion = "3.12"
$VenvDir = "ort-genie-venv"
$BuildScript = "qcom\build_and_test.py"
$BuildRoot   = "build"
$BuildArch = "windows-arm64"
$BuildConfig = "Release"
$SourceRootRel = "..\.."

# ----- 0) Check for tools -----
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not installed. Install from: https://github.com/astral-sh/uv"
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "python is not available on PATH."
}

# ----- 1) Deactivate active venv -----
if ($env:VIRTUAL_ENV) {
    throw "Virtual environment active at $env:VIRTUAL_ENV - please deactivate."
}

# ----- 2) Remove venv -----
Write-Host "Removing old venv: $VenvDir"
Remove-Item -Recurse -Force $VenvDir -ErrorAction Ignore

# ----- 3) Clean & Build -----
Write-Host "Changing location to source root..."
Set-Location $SourceRootRel

Write-Host "Removing old build folder at $BuildRoot"
Remove-Item -Recurse -Force $BuildRoot -ErrorAction Ignore

Write-Host "Running build script..."
python $BuildScript build

# ----- 4) Install wheels/sdists -----
Set-Location "qcom\gen_ai"

Write-Host "Creating new venv using uv..."
uv venv -p $PythonVersion $VenvDir

# Activate it
$activateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
Write-Host "Activating virtual environment..."
& $activateScript

Set-Location $SourceRootRel

$DistDir = Join-Path $BuildRoot "$BuildArch\$BuildConfig\dist"

if (-not (Test-Path $DistDir)) {
    throw "Distribution directory not found: $DistDir"
}

Write-Host "Installing artifacts from $DistDir"

$Artifacts = `
    Get-ChildItem $DistDir\* -File -Include *.whl, *.tar.gz |
    Sort-Object Name

if ($Artifacts.Count -eq 0) {
    throw "No wheel (.whl) or sdist (.tar.gz) found."
}

uv pip install $Artifacts.FullName

# ----- 5) Configure runtime DLL search path -----
# Windows resolves dlls from PATH
$RuntimeDir = Split-Path $DistDir -Parent

Write-Host "Adding runtime directory to PATH: $RuntimeDir"
$env:PATH = "$RuntimeDir;$env:PATH"

# ----- 6) Return to qcom/gen_ai -----
Set-Location "qcom\gen_ai"

Write-Host "Done. Active venv."
