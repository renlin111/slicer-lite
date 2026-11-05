param(
    [string]$Python = "python"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot
try {
    $env:PYTHONPATH = Join-Path $PSScriptRoot ".vendor\python"
    & $Python -m pip show pyinstaller | Out-Null
    if ($LASTEXITCODE -ne 0) {
        & $Python -m pip show pyinstaller --path ".vendor\python" | Out-Null
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "PyInstaller is not installed. Run: $Python -m pip install --target .vendor\python pyinstaller"
        exit 1
    }
    & $Python -m PyInstaller --noconfirm SlicerLite.spec
    Write-Host "Build finished: dist\SlicerLite\SlicerLite.exe"
}
finally {
    Pop-Location
}
