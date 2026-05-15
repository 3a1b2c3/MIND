@echo off
:: Stage DreamX-World videos into MIND-tests\dreamx-world\ for run_mind.bat scoring.
::
:: Usage:
::   drive_dreamx.bat                            stage all samples
::   drive_dreamx.bat --dry-run                  preview commands without running inference
::   drive_dreamx.bat --limit 5                  first 5 samples only
::   drive_dreamx.bat --perspective 1st_data     limit to first-person
::   drive_dreamx.bat --test-type mem_test       limit to memory tests
::
:: All flags pass through to src\drive_dreamx.py.

setlocal enableextensions

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

set PY=%~dp0.venv\Scripts\python.exe
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests

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
echo   model     : dreamx-world
echo ============================================================

"%PY%" src\drive_dreamx.py --gt-root "%GT_ROOT%" --test-root "%MIND_TESTS%" %*

set EXIT_CODE=%ERRORLEVEL%
if %EXIT_CODE%==0 (
    echo.
    echo ============================================================
    echo Done. Now score with: run_mind.bat dreamx-world
    echo ============================================================
) else (
    echo.
    echo ERROR: drive_dreamx.py exited with %EXIT_CODE%
)
exit /b %EXIT_CODE%
