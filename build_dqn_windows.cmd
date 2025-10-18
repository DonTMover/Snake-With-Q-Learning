@echo off
setlocal ENABLEDELAYEDEXPANSION

REM build_dqn_windows.cmd
REM Detect CUDA on Windows and build DQN variant accordingly.
REM - If CUDA nvcc is available, build with features: dqn-gpu dqn-gpu-cuda
REM - Else, build CPU DQN: dqn-gpu
REM Usage: build_dqn_windows.cmd [release|debug]

set "PROFILE=%~1"
if /I "%PROFILE%"=="" set "PROFILE=release"
set "CARGO_FLAGS="
if /I "%PROFILE%"=="release" set "CARGO_FLAGS=--release"
if /I NOT "%PROFILE%"=="release" set "PROFILE=debug"

REM Try to locate nvcc in PATH or via CUDA_PATH
where nvcc >nul 2>&1
set "HAS_CUDA=%ERRORLEVEL%"
if not "%HAS_CUDA%"=="0" (
  if defined CUDA_PATH (
    set "PATH=%CUDA_PATH%\bin;%PATH%"
    where nvcc >nul 2>&1
    set "HAS_CUDA=%ERRORLEVEL%"
  )
)

if "%HAS_CUDA%"=="0" goto build_cuda
goto build_cpu

:build_cuda
echo [build] CUDA detected - building DQN with CUDA features
call cargo build %CARGO_FLAGS% --no-default-features --features "dqn-gpu dqn-gpu-cuda"
if errorlevel 1 goto fallback_cpu
goto finalize

:fallback_cpu
echo [build] CUDA build failed - falling back to CPU DQN
call cargo build %CARGO_FLAGS% --no-default-features --features dqn-gpu
if errorlevel 1 exit /b 1
goto finalize

:build_cpu
echo [build] CUDA not detected - building CPU DQN variant
call cargo build %CARGO_FLAGS% --no-default-features --features dqn-gpu
if errorlevel 1 exit /b 1
goto finalize

:finalize
REM Rename/copy the produced exe to snake-dqn.exe for artifact consistency
set "OUTDIR=target\%PROFILE%"
set "SRCEXE="
for %%F in ("%OUTDIR%\Snake-*.exe") do (
  set "SRCEXE=%%~fF"
  goto :copyit
)
for %%F in ("%OUTDIR%\*.exe") do (
  set "SRCEXE=%%~fF"
  goto :copyit
)
echo [build] Error - no exe found in %OUTDIR%
exit /b 1

:copyit
copy /Y "%SRCEXE%" "%OUTDIR%\snake-dqn.exe" >nul
set "SRCPDB=%SRCEXE:.exe=.pdb%"
if exist "%SRCPDB%" copy /Y "%SRCPDB%" "%OUTDIR%\snake-dqn.pdb" >nul

if not exist "%OUTDIR%\snake-dqn.exe" (
  echo [build] Error - failed to create snake-dqn.exe
  exit /b 1
)
echo [build] Done - output at %OUTDIR%\snake-dqn.exe
exit /b 0
