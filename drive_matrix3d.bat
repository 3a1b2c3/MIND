@echo off
:: Stage Matrix-3D videos into MIND-tests\matrix-3d\.
:: TODO: needs src\drive_matrix3d.py; no .venv (set MATRIX3D_PY env).

setlocal enableextensions
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

set PY=%~dp0.venv\Scripts\python.exe
if not defined MATRIX3D_PY set MATRIX3D_PY=python
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests
set MODEL_NAME=matrix-3d
set MATRIX3D_REPO=C:\workspace\world\Matrix-3D
set LOG=%~dp0drive_matrix3d.log

if not exist "%PY%" ( echo ERROR: venv python not found: %PY% & exit /b 2 )
if not exist "%MATRIX3D_REPO%" ( echo ERROR: repo not found: %MATRIX3D_REPO% & exit /b 2 )
if not exist "%~dp0src\drive_matrix3d.py" ( echo ERROR: src\drive_matrix3d.py not yet written & exit /b 2 )

echo === Matrix-3D staging into MIND-tests  ^|  model=%MODEL_NAME% ===
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_dreamx.ps1" "%LOG%" "%PY%" "src\drive_matrix3d.py" "--gt-root" "%GT_ROOT%" "--test-root" "%MIND_TESTS%" "--model-name" "%MODEL_NAME%" "--repo" "%MATRIX3D_REPO%" "--py" "%MATRIX3D_PY%" "--perspective" "1st_data" %*
set EXIT_CODE=%ERRORLEVEL%
if not %EXIT_CODE%==0 ( exit /b %EXIT_CODE% )
call "%~dp0run_mind.bat" "%MODEL_NAME%"
exit /b %ERRORLEVEL%
