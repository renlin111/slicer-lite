@echo off
setlocal
cd /d "%~dp0"
set "CODEX_PY=C:\Users\77692\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%CODEX_PY%" (
  "%CODEX_PY%" run_app.py
) else (
  python run_app.py
)
