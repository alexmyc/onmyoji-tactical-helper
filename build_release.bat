@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title OnmyojiHelper - Multi-Edition Release Builder

if not "%~1"=="" (
    python build_release.py %*
    goto :done
)

echo ============================================================================
echo   Onmyoji Tactical Helper - Hardened Cython .pyd ^& Release Builder
echo ============================================================================
echo   [1] Personal Full Edition        ^(dist\OnmyojiHelper_Personal^)
echo       - 33 native .pyd modules + local .env + best.pt + local .gguf
echo.
echo   [2] HR / Recruiter Demo Edition  ^(dist\OnmyojiHelper_HR_Demo + Win64.zip^)
echo       - 33 native .pyd modules + ZERO .env + auto-opens Offline Demo on click
echo.
echo   [3] Friend Full Edition - CUDA   ^(dist\OnmyojiHelper_Friend_Full^)
echo       - Hardened .pyd + WARNING log + whitelist .env + bundled local .gguf
echo.
echo   [4] Friend Lite Edition - No GPU ^(dist\OnmyojiHelper_Friend_Lite^)
echo       - Hardened .pyd + whitelist .env + excludes llama_cpp + FORCE_LITE_ENGINE=1
echo ============================================================================
set "choice="
set /p choice=Select edition [1-4, press Enter for default 2 (HR Demo)]: 

if "%choice%"=="1" (
    python build_release.py --profile personal --workers 2
) else if "%choice%"=="3" (
    python build_release.py --profile friend-full --workers 2
) else if "%choice%"=="4" (
    python build_release.py --profile friend-lite --workers 2
) else (
    python build_release.py --profile hr-demo --workers 2
)

:done
echo.
pause
endlocal
