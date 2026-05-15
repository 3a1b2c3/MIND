@echo off
:: Stage Matrix-Game-3 videos into MIND-tests\matrix-game-3\ for run_mind.bat scoring.
::
:: Usage:
::   drive_matrix3.bat                            stage all samples
::   drive_matrix3.bat --dry-run                  preview commands without running inference
::   drive_matrix3.bat --limit 5                  first 5 samples only
::   drive_matrix3.bat --perspective 1st_data     limit to first-person
::   drive_matrix3.bat --test-type mem_test       limit to memory tests
::
:: All flags pass through to src\drive_matrix3.py.

setlocal enableextensions

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

set PY=%~dp0.venv\Scripts\python.exe
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests
set LOG=%~dp0drive_matrix3.log

if not exist "%PY%" (
    echo ERROR: venv python not found: %PY%
    exit /b 2
)
if not exist "%GT_ROOT%" (
    echo ERROR: gt_root not found: %GT_ROOT%
    exit /b 2
)

echo ============================================================
echo Matrix-Game-3 staging into MIND-tests
echo ============================================================
echo   gt_root   : %GT_ROOT%
echo   test_root : %MIND_TESTS%
echo   model     : matrix-game-3
echo   log       : %LOG%
echo ============================================================

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_dreamx.ps1" "%LOG%" "%PY%" "src\drive_matrix3.py" "--gt-root" "%GT_ROOT%" "--test-root" "%MIND_TESTS%" %*

set EXIT_CODE=%ERRORLEVEL%
if not %EXIT_CODE%==0 (
    echo.
    echo ERROR: drive_matrix3.py exited with %EXIT_CODE%
    exit /b %EXIT_CODE%
)

echo.
echo ============================================================
echo Generation done. Running scoring: run_mind.bat matrix-game-3
echo ============================================================
call "%~dp0run_mind.bat" matrix-game-3
exit /b %ERRORLEVEL%
