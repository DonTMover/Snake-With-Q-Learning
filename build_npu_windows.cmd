@echo off
setlocal ENABLEDELAYEDEXPANSION

REM build_npu_windows.cmd
REM Build the NPU/DirectML (ONNX Runtime) variant for Windows.
REM Usage: build_npu_windows.cmd [release|debug]

set "PROFILE=%~1"
if /I "%PROFILE%"=="" set "PROFILE=release"
set "CARGO_FLAGS="
if /I "%PROFILE%"=="release" set "CARGO_FLAGS=--release"
if /I NOT "%PROFILE%"=="release" set "PROFILE=debug"

call cargo build %CARGO_FLAGS% --no-default-features --features npu-directml
if errorlevel 1 exit /b 1

set "OUTDIR=target\%PROFILE%"
for %%F in ("%OUTDIR%\Snake-*.exe") do (
  copy /Y "%%~fF" "%OUTDIR%\snake-npu.exe" >nul
  goto :done
)
for %%F in ("%OUTDIR%\*.exe") do (
  copy /Y "%%~fF" "%OUTDIR%\snake-npu.exe" >nul
  goto :done
)

echo [build] Error - no exe found in %OUTDIR%
exit /b 1

:done
echo [build] Done - output at %OUTDIR%\snake-npu.exe
exit /b 0
