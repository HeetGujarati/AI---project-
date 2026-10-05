@echo off
rem ============================================================
rem Enforce all temporary files, Hugging Face caches, PyTorch hub,
rem pip downloads, and Python bytecode to D:\Lora++\.cache
rem ============================================================

set BASE_CACHE=D:\Lora++\.cache
set TEMP=%BASE_CACHE%\temp
set TMP=%BASE_CACHE%\temp
set TMPDIR=%BASE_CACHE%\temp
set HF_HOME=%BASE_CACHE%\huggingface
set TRANSFORMERS_CACHE=%BASE_CACHE%\huggingface
set HF_DATASETS_CACHE=%BASE_CACHE%\huggingface\datasets
set TORCH_HOME=%BASE_CACHE%\torch
set PIP_CACHE_DIR=%BASE_CACHE%\pip
set PYTHONPYCACHEPREFIX=%BASE_CACHE%\pycache

set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set KMP_DUPLICATE_LIB_OK=TRUE
set TOKENIZERS_PARALLELISM=false

if not exist "%TEMP%" mkdir "%TEMP%"
if not exist "%HF_HOME%" mkdir "%HF_HOME%"
if not exist "%TORCH_HOME%" mkdir "%TORCH_HOME%"
if not exist "%PIP_CACHE_DIR%" mkdir "%PIP_CACHE_DIR%"
if not exist "%PYTHONPYCACHEPREFIX%" mkdir "%PYTHONPYCACHEPREFIX%"

echo [*] D: Drive Cache and Memory Protection Active!
echo [*] TEMP: %TEMP%
echo [*] HF_HOME: %HF_HOME%

if "%~1"=="" (
    cmd /k
) else (
    %*
)
