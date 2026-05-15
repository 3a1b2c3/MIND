@echo off
:: Score a test video set against MIND-Data ground truth.
::
:: Usage:
::   run_mind.bat                                  scores matrix-game-3, all metrics
::   run_mind.bat dreamx-world                     scores dreamx-world, all metrics
::   run_mind.bat <test_subdir>                    scores MIND-tests\<test_subdir>
::   run_mind.bat <test_subdir> <metrics>          custom metrics
::   run_mind.bat <test_subdir> <metrics> <gpus>   multi-GPU
::
:: Examples:
::   run_mind.bat matrix-game-3 lcm,visual
::   run_mind.bat dreamx-world lcm,visual,dino,action,gsc 1

setlocal enableextensions

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

set PY=%~dp0.venv\Scripts\python.exe
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests

set TEST_SUBDIR=%~1
set METRICS=%~2
set NUM_GPUS=%~3

if not defined TEST_SUBDIR set TEST_SUBDIR=matrix-game-3
if not defined METRICS set METRICS=lcm,visual,dino,action,gsc
if not defined NUM_GPUS set NUM_GPUS=1

set TEST_ROOT=%MIND_TESTS%\%TEST_SUBDIR%

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
echo   gt_root   : %GT_ROOT%
echo   test_root : %TEST_ROOT%
echo   metrics   : %METRICS%
echo   gpus      : %NUM_GPUS%
echo ============================================================

"%PY%" src\process.py --gt_root "%GT_ROOT%" --test_root "%TEST_ROOT%" --metrics %METRICS% --num_gpus %NUM_GPUS%

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
