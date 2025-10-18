@echo off
setlocal
set REAL_NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe
if not exist "%REAL_NVCC%" (
  echo ERROR: nvcc not found at %REAL_NVCC%
  exit /b 1
)
"%REAL_NVCC%" --allow-unsupported-compiler %*
endlocal
