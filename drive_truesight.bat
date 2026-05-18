@echo off
:: Stage TrueSight (Wan 2.1 I2V 14B) videos into MIND-tests\truesight\ for run_mind.bat scoring.
::
:: TODO: needs src\drive_truesight.py that:
::   - Walks MIND-Data first frames + action.json
::   - Invokes truesight\inference_i2v.py (uses truesight\.venv\Scripts\python.exe)
::   - Stages output to MIND-tests\truesight\<perspective>\<test_type>\<gt_name>\video.mp4
::
:: Standard params (match MIND-Data 24 fps): 1280x704, 121 frames, 24 fps, 50 steps.

setlocal enableextensions
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

set PY=%~dp0.venv\Scripts\python.exe
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests
set MODEL_NAME=truesight
set TRUESIGHT_REPO=C:\workspace\world\truesight
set LOG=%~dp0drive_truesight.log

if not exist "%PY%" ( echo ERROR: venv python not found: %PY% & exit /b 2 )
if not exist "%GT_ROOT%" ( echo ERROR: gt_root not found: %GT_ROOT% & exit /b 2 )
if not exist "%TRUESIGHT_REPO%\inference_i2v.py" (
    echo ERROR: truesight\inference_i2v.py not found
    echo Expected: %TRUESIGHT_REPO%\inference_i2v.py
    exit /b 2
)
if not exist "%~dp0src\drive_truesight.py" (
    echo ERROR: src\drive_truesight.py not yet written
    echo This bat is a stub; create the driver script first.
    exit /b 2
)

echo ============================================================
echo TrueSight staging into MIND-tests
echo ============================================================
echo   gt_root      : %GT_ROOT%
echo   test_root    : %MIND_TESTS%
echo   model        : %MODEL_NAME%
echo   truesight    : %TRUESIGHT_REPO%
echo   log          : %LOG%
echo ============================================================

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_dreamx.ps1" "%LOG%" "%PY%" "src\drive_truesight.py" "--gt-root" "%GT_ROOT%" "--test-root" "%MIND_TESTS%" "--model-name" "%MODEL_NAME%" "--truesight-repo" "%TRUESIGHT_REPO%" "--perspective" "1st_data" %*

set EXIT_CODE=%ERRORLEVEL%
if not %EXIT_CODE%==0 ( echo. & echo ERROR: drive_truesight.py exited with %EXIT_CODE% & exit /b %EXIT_CODE% )

echo.
echo ============================================================
echo Generation done. Running scoring: run_mind.bat %MODEL_NAME%
echo ============================================================
call "%~dp0run_mind.bat" "%MODEL_NAME%"
exit /b %ERRORLEVEL%
