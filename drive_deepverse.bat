@echo off
:: Stage DeepVerse videos into MIND-tests\deepverse\ for run_mind.bat scoring.
::
:: TODO: needs src\drive_deepverse.py that:
::   - Walks MIND-Data first frames + action.json
::   - Invokes DeepVerse\run.py (uses DeepVerse\.venv\Scripts\python.exe)
::   - Stages output to MIND-tests\deepverse\<perspective>\<test_type>\<gt_name>\video.mp4

setlocal enableextensions
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

set PY=%~dp0.venv\Scripts\python.exe
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests
set MODEL_NAME=deepverse
set DEEPVERSE_REPO=C:\workspace\world\DeepVerse
set LOG=%~dp0drive_deepverse.log
if not defined MIND_FPS set MIND_FPS=24

if not exist "%PY%" ( echo ERROR: venv python not found: %PY% & exit /b 2 )
if not exist "%GT_ROOT%" ( echo ERROR: gt_root not found: %GT_ROOT% & exit /b 2 )
if not exist "%DEEPVERSE_REPO%\run.py" (
    echo ERROR: DeepVerse\run.py not found at %DEEPVERSE_REPO%\run.py
    exit /b 2
)
if not exist "%~dp0src\drive_deepverse.py" (
    echo ERROR: src\drive_deepverse.py not yet written
    echo This bat is a stub; create the driver script first.
    exit /b 2
)

echo ============================================================
echo DeepVerse staging into MIND-tests
echo ============================================================
echo   gt_root      : %GT_ROOT%
echo   test_root    : %MIND_TESTS%
echo   model        : %MODEL_NAME%
echo   deepverse    : %DEEPVERSE_REPO%
echo   log          : %LOG%
echo ============================================================

:: drive_deepverse.py PERSPECTIVES tuple now defaults to ("3rd_data","1st_data"),
:: so omitting --perspective walks both with 3rd-person first. Pass --perspective
:: <p> on the CLI to restrict to one. CLI args after %* override the defaults.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_dreamx.ps1" "%LOG%" "%PY%" "src\drive_deepverse.py" "--gt-root" "%GT_ROOT%" "--test-root" "%MIND_TESTS%" "--model-name" "%MODEL_NAME%" "--deepverse-repo" "%DEEPVERSE_REPO%" "--fps" "%MIND_FPS%" %*

set EXIT_CODE=%ERRORLEVEL%
if not %EXIT_CODE%==0 ( echo. & echo ERROR: drive_deepverse.py exited with %EXIT_CODE% & exit /b %EXIT_CODE% )

echo.
echo ============================================================
echo Generation done. Running scoring: run_mind.bat %MODEL_NAME%
echo ============================================================
if not defined MIND_METRICS set MIND_METRICS=lcm,visual,dino,action,gsc
call "%~dp0run_mind.bat" "%MODEL_NAME%" "%MIND_METRICS%"
exit /b %ERRORLEVEL%
