$ErrorActionPreference = "Stop"
$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RootDir
$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
$VenvDir = if ($env:VENV_DIR) { $env:VENV_DIR } else { ".venv-rewrite-build" }
& $PythonBin -m venv $VenvDir
. (Join-Path $VenvDir "Scripts\Activate.ps1")
python -m pip install --upgrade pip
python -m pip install -r requirements-build.txt -r requirements-dev.txt
python -m compileall -q split3r_rewrite tests_rewrite
python -m pytest -q tests_rewrite
pyinstaller packaging/split3r_rewrite.spec --clean --noconfirm
Write-Host ""
Write-Host "Rewrite build ready at: $RootDir\dist\Split3rRewrite"
