@echo off
:: Stage Infinite-World videos into MIND-tests\infinite-world\.
:: TODO: needs src\drive_infinite.py; no .venv (set INFINITE_PY env).

setlocal enableextensions
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

set PY=%~dp0.venv\Scripts\python.exe
if not defined INFINITE_PY set INFINITE_PY=python
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests
set MODEL_NAME=infinite-world
set INFINITE_REPO=C:\workspace\world\Infinite-World
set LOG=%~dp0drive_infinite.log

if not exist "%PY%" ( echo ERROR: venv python not found: %PY% & exit /b 2 )
if not exist "%INFINITE_REPO%" ( echo ERROR: repo not found: %INFINITE_REPO% & exit /b 2 )
if not exist "%~dp0src\drive_infinite.py" ( echo ERROR: src\drive_infinite.py not yet written & exit /b 2 )

echo === Infinite-World staging into MIND-tests  ^|  model=%MODEL_NAME% ===
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_dreamx.ps1" "%LOG%" "%PY%" "src\drive_infinite.py" "--gt-root" "%GT_ROOT%" "--test-root" "%MIND_TESTS%" "--model-name" "%MODEL_NAME%" "--repo" "%INFINITE_REPO%" "--py" "%INFINITE_PY%" "--perspective" "1st_data" %*
set EXIT_CODE=%ERRORLEVEL%
if not %EXIT_CODE%==0 ( exit /b %EXIT_CODE% )
call "%~dp0run_mind.bat" "%MODEL_NAME%"
exit /b %ERRORLEVEL%
