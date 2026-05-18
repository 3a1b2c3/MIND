@echo off
:: Score a test video set against MIND-Data ground truth.
::
:: Usage:
::   run_mind.bat                                          scores matrix-game-3, all metrics, both perspectives
::   run_mind.bat <test_subdir>                            scores MIND-tests\<test_subdir>
::   run_mind.bat <test_subdir> <metrics>                  custom metrics
::   run_mind.bat <test_subdir> <metrics> <gpus>           multi-GPU
::   run_mind.bat <test_subdir> <metrics> <gpus> <person>  person = 1st | 3rd | both (default: both)
::
:: Examples:
::   run_mind.bat matrix-game-3 lcm,visual
::   run_mind.bat dreamx-world lcm,visual,dino,action,gsc 1
::   run_mind.bat dreamx-world lcm,visual,dino,gsc 1 1st

setlocal enableextensions

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
:: Strip cross-venv pollution. The PowerShell session that launched this bat
:: often has VIRTUAL_ENV / PYTHONHOME / PYTHONPATH set from another project
:: (e.g. scope's uv-managed 3.12 venv). MIND uses python 3.10, so if those
:: leak through they make Python try to graft scope's 3.12 site-packages onto
:: MIND's 3.10 stdlib path, causing _sre.MAGIC mismatch at startup.
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONSTARTUP="
set "VIRTUAL_ENV="
set "VIRTUAL_ENV_PROMPT="
set "UV_PYTHON="
set "UV_PROJECT_ENVIRONMENT="

set PY=%~dp0.venv\Scripts\python.exe
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests

set TEST_SUBDIR=%~1
set METRICS=%~2
set NUM_GPUS=%~3
set PERSON=%~4

if not defined TEST_SUBDIR set TEST_SUBDIR=matrix-game-3
if not defined METRICS set METRICS=lcm,visual,dino,action,gsc
if not defined NUM_GPUS set NUM_GPUS=1

:: 4th positional = perspective filter. Accepts: 1st, 3rd, both (default = both).
set PERSPECTIVES=
if /I "%PERSON%"=="1st"  set PERSPECTIVES=1st_data
if /I "%PERSON%"=="3rd"  set PERSPECTIVES=3rd_data
if /I "%PERSON%"=="both" set PERSPECTIVES=
if /I "%PERSON%"==""     set PERSPECTIVES=
if defined PERSON if not defined PERSPECTIVES if /I not "%PERSON%"=="both" (
    echo ERROR: 4th arg must be 1st, 3rd, or both ^(got: %PERSON%^)
    exit /b 2
)

:: Accept either a bare subdir name (resolved under MIND_TESTS) or an absolute
:: path. Detection: an absolute path contains ":\" or starts with "\\" (UNC).
:: This lets you tab-complete MIND-tests\dreamx-world_small in the shell without
:: tripping the prefix-prepend below.
echo %TEST_SUBDIR% | findstr /R /C:":\\" /C:"^\\\\" >nul
if errorlevel 1 (
    set TEST_ROOT=%MIND_TESTS%\%TEST_SUBDIR%
) else (
    set TEST_ROOT=%TEST_SUBDIR%
)

if not exist "%PY%" (
    echo ERROR: venv python not found: %PY%
    echo Make sure MIND is set up.
    exit /b 2
)
if not exist "%GT_ROOT%" (
    echo ERROR: gt_root not found: %GT_ROOT%
    exit /b 2
)
if not exist "%TEST_ROOT%" (
    echo ERROR: test_root not found: %TEST_ROOT%
    echo Stage videos there first via drive_matrix3.py / drive_mind.py / score_matrix3.py.
    exit /b 2
)

echo ============================================================
echo MIND scoring
echo ============================================================
echo   gt_root      : %GT_ROOT%
echo   test_root    : %TEST_ROOT%
echo   metrics      : %METRICS%
echo   gpus         : %NUM_GPUS%
echo   person       : %PERSON% (perspectives=%PERSPECTIVES%)
echo ============================================================

if defined PERSPECTIVES (
    "%PY%" src\process.py --gt_root "%GT_ROOT%" --test_root "%TEST_ROOT%" --metrics %METRICS% --num_gpus %NUM_GPUS% --perspectives %PERSPECTIVES%
) else (
    "%PY%" src\process.py --gt_root "%GT_ROOT%" --test_root "%TEST_ROOT%" --metrics %METRICS% --num_gpus %NUM_GPUS%
)

set EXIT_CODE=%ERRORLEVEL%
if %EXIT_CODE%==0 (
    echo.
    echo ============================================================
    echo Done. Result JSON written next to this script.
    echo ============================================================
) else (
    echo.
    echo ERROR: process.py exited with %EXIT_CODE%
)
exit /b %EXIT_CODE%
