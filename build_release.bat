@echo off
chcp 65001 >nul
setlocal
title OnmyojiHelper - Native Binary Release Builder
echo ============================================================================
echo   Onmyoji Tactical Helper - Hardened Native .pyd + Single-Exe Release Builder
echo ============================================================================
echo   [1] Lite Build (Recommended: 0-VRAM deterministic rule engine, fast package)
echo   [2] Full Build (Bundles 3 GB local Gemma-4 GGUF weights + llama-cpp-python)
echo ============================================================================
set "choice="
set /p choice=Select build tier (1 or 2, press Enter for default [1]): 

if "%choice%"=="2" (
    python build_release.py --full
) else (
    python build_release.py
)

echo.
pause
endlocal
