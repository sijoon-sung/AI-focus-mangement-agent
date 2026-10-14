@echo off
pushd "%~dp0"
python -m local_agent %*
if errorlevel 1 pause
popd
