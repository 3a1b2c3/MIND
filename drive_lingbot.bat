@echo off
:: Stage LingBot-World-Fast videos into MIND-tests\lingbot-fast\ for run_mind.bat scoring.
::
:: Uses lingbot-world's generate.py with the MIND action.json passed through
:: --action_path, so per-frame WASD/ud/lr conditioning flows end-to-end. That
:: makes the `action` MIND metric meaningful for this model (unlike the
:: matrix3 / dreamx drivers which stub a single placeholder action).
::
:: Usage:
::   drive_lingbot.bat                            stage all samples then score
::   drive_lingbot.bat --dry-run                  preview commands without running inference
::   drive_lingbot.bat --limit 5                  first 5 samples only
::   drive_lingbot.bat --perspective 1st_data     limit to first-person
::   drive_lingbot.bat --test-type mem_test       limit to memory tests
::
:: All flags pass through to src\drive_lingbot.py.

setlocal enableextensions

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

set PY=%~dp0.venv\Scripts\python.exe
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests
set MODEL_NAME=lingbot-fast
set CKPT_DIR=C:\workspace\world\lingbot-world\fast
set LOG=%~dp0drive_lingbot.log

if not exist "%PY%" (
    echo ERROR: venv python not found: %PY%
    exit /b 2
)
if not exist "%GT_ROOT%" (
    echo ERROR: gt_root not found: %GT_ROOT%
    exit /b 2
)
if not exist "%CKPT_DIR%" (
    echo WARNING: ckpt_dir not present: %CKPT_DIR%
    echo Run download_fast.bat in lingbot-world first ^(unless using --dry-run^).
    echo.
)

echo ============================================================
echo LingBot-World-Fast staging into MIND-tests
echo ============================================================
echo   gt_root   : %GT_ROOT%
echo   test_root : %MIND_TESTS%
echo   model     : %MODEL_NAME%
echo   ckpt_dir  : %CKPT_DIR%
echo   log       : %LOG%
echo ============================================================

:: --perspective 1st_data: lingbot-fast only stages first-person samples.
:: To include 3rd_data, pass an overriding `--perspective 3rd_data` as an
:: extra arg — argparse's "last wins" rule lets the user override.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_dreamx.ps1" "%LOG%" "%PY%" "src\drive_lingbot.py" "--gt-root" "%GT_ROOT%" "--test-root" "%MIND_TESTS%" "--model-name" "%MODEL_NAME%" "--ckpt-dir" "%CKPT_DIR%" "--perspective" "1st_data" %*

set EXIT_CODE=%ERRORLEVEL%
if not %EXIT_CODE%==0 (
    echo.
    echo ERROR: drive_lingbot.py exited with %EXIT_CODE%
    exit /b %EXIT_CODE%
)

echo.
echo ============================================================
echo Staging done. Scoring with run_mind.bat %MODEL_NAME% ...
echo ============================================================
echo.

:: run_mind.bat defaults PERSON=1st, matching this bat's 1st_data-only generation.
call "%~dp0run_mind.bat" %MODEL_NAME%

set SCORE_EXIT=%ERRORLEVEL%
if not %SCORE_EXIT%==0 (
    echo.
    echo ERROR: run_mind.bat exited with %SCORE_EXIT%
    exit /b %SCORE_EXIT%
)
echo.
echo Done. See result_%MODEL_NAME%_*.json in this directory.
endlocal
