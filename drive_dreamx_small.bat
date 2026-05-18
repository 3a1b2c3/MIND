@echo off
:: Stage DreamX-World videos into MIND-tests\dreamx-world_small\ for run_mind.bat scoring.
::
:: Identical to drive_dreamx.bat except --model-name is `dreamx-world_small`, so
:: the staged videos land under a separate test-root subdir and don't collide
:: with the full-resolution `dreamx-world` set.
::
:: Usage:
::   drive_dreamx_small.bat                            stage all samples
::   drive_dreamx_small.bat --dry-run                  preview commands without running inference
::   drive_dreamx_small.bat --limit 5                  first 5 samples only
::   drive_dreamx_small.bat --perspective 1st_data     limit to first-person
::   drive_dreamx_small.bat --test-type mem_test       limit to memory tests
::
:: All flags pass through to src\drive_dreamx.py.

setlocal enableextensions

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

set PY=%~dp0.venv\Scripts\python.exe
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests
set MODEL_NAME=dreamx-world_small
set LOG=%~dp0drive_dreamx_small.log

if not exist "%PY%" (
    echo ERROR: venv python not found: %PY%
    exit /b 2
)
if not exist "%GT_ROOT%" (
    echo ERROR: gt_root not found: %GT_ROOT%
    exit /b 2
)

echo ============================================================
echo DreamX-World staging into MIND-tests (small)
echo ============================================================
echo   gt_root   : %GT_ROOT%
echo   test_root : %MIND_TESTS%
echo   model     : %MODEL_NAME%
echo   log       : %LOG%
echo ============================================================

:: Speed bundle: half-resolution, 30 steps, 81 frames @ 16fps (still 5s output),
:: fp8 transformer weights. ~4-5x faster than the full-res defaults.
:: --perspective 1st_data: only stage first-person samples. Override with an
:: extra `--perspective 3rd_data` arg (argparse last-wins).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_dreamx.ps1" "%LOG%" "%PY%" "src\drive_dreamx.py" "--gt-root" "%GT_ROOT%" "--test-root" "%MIND_TESTS%" "--model-name" "%MODEL_NAME%" "--height" "352" "--width" "640" "--video-length" "81" "--fps" "16" "--steps" "30" "--gpu-memory-mode" "model_full_load_and_qfloat8" "--perspective" "1st_data" %*

set EXIT_CODE=%ERRORLEVEL%
if not %EXIT_CODE%==0 (
    echo.
    echo ERROR: drive_dreamx.py exited with %EXIT_CODE%
    exit /b %EXIT_CODE%
)

echo.
echo ============================================================
echo Generation done. Running scoring: run_mind.bat %MODEL_NAME% (1st only)
echo ============================================================
:: Score only 1st_data to match generation. 4th arg = "1st" (run_mind PERSON flag).
call "%~dp0run_mind.bat" "%MODEL_NAME%" lcm,visual,dino,action,gsc 1 1st
exit /b %ERRORLEVEL%
