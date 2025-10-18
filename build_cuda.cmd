@echo off
setlocal
rem Enter MSVC dev environment for cl.exe
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -no_logo -arch=amd64
where cl >nul 2>nul
if errorlevel 1 (
  echo ERROR: cl.exe not found after VsDevCmd. Ensure C++ build tools are installed.
  exit /b 1
)

rem Detect the newest installed CUDA (prefer 12.6/12.5/12.4, fallback to 12.0)
set CUDAROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA
set CUDA_PATH=
if exist "%CUDAROOT%\v12.6" set CUDA_PATH=%CUDAROOT%\v12.6
if "%CUDA_PATH%"=="" if exist "%CUDAROOT%\v12.5" set CUDA_PATH=%CUDAROOT%\v12.5
if "%CUDA_PATH%"=="" if exist "%CUDAROOT%\v12.4" set CUDA_PATH=%CUDAROOT%\v12.4
if "%CUDA_PATH%"=="" if exist "%CUDAROOT%\v12.0" set CUDA_PATH=%CUDAROOT%\v12.0
if "%CUDA_PATH%"=="" (
  echo ERROR: Could not locate CUDA under %CUDAROOT%.
  exit /b 1
)
setx CUDA_PATH "%CUDA_PATH%" >nul 2>nul
set PATH=%CUDA_PATH%\bin;%PATH%
set CUDACXX=%CUDA_PATH%\bin\nvcc.exe

echo Using CUDA at %CUDA_PATH%
set RUST_BACKTRACE=1
echo Building (logs -> build_cuda.log)...
cargo build -vv --release --features "dqn-gpu dqn-gpu-cuda" > build_cuda.log 2>&1
set BUILD_ERR=%ERRORLEVEL%
if not %BUILD_ERR%==0 (
  echo ---- BUILD FAILED (last 200 lines) ----
  powershell -NoProfile -Command "Get-Content -Path 'build_cuda.log' -Tail 200"
  exit /b %BUILD_ERR%
)
type build_cuda.log | powershell -NoProfile -Command "$input | Select-Object -Last 20"
echo ---- BUILD SUCCEEDED ----
endlocal
