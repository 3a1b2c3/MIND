@echo off
:: Stage DreamX-World videos into MIND-tests\dreamx-world\ and auto-score with run_mind.bat.
::
:: Usage:
::   drive_dreamx.bat                            stage all samples then score
::   drive_dreamx.bat --dry-run                  preview commands without running inference
::   drive_dreamx.bat --limit 5                  first 5 samples only
::   drive_dreamx.bat --perspective 1st_data     limit to first-person
::   drive_dreamx.bat --test-type mem_test       limit to memory tests
::
:: All flags pass through to src\drive_dreamx.py.

setlocal enableextensions

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

set PY=%~dp0.venv\Scripts\python.exe
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests
set MODEL_NAME=dreamx-world
set LOG=%~dp0drive_dreamx.log

if not exist "%PY%" (
    echo ERROR: venv python not found: %PY%
    exit /b 2
)
if not exist "%GT_ROOT%" (
    echo ERROR: gt_root not found: %GT_ROOT%
    exit /b 2
)

echo ============================================================
echo DreamX-World staging into MIND-tests
echo ============================================================
echo   gt_root   : %GT_ROOT%
echo   test_root : %MIND_TESTS%
echo   model     : %MODEL_NAME%
echo   log       : %LOG%
echo ============================================================

:: --perspective 1st_data: only stage first-person samples. Override with an
:: extra `--perspective 3rd_data` arg (argparse last-wins).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_dreamx.ps1" "%LOG%" "%PY%" "src\drive_dreamx.py" "--gt-root" "%GT_ROOT%" "--test-root" "%MIND_TESTS%" "--model-name" "%MODEL_NAME%" "--perspective" "1st_data" %*

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
