@echo off
pushd "%~dp0"
if exist ".venv-research\Scripts\python.exe" (
  ".venv-research\Scripts\python.exe" -m local_agent %*
) else (
  python -m local_agent %*
)
if errorlevel 1 pause
popd
