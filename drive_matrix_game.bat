@echo off
:: Stage original Matrix-Game videos into MIND-tests\matrix-game\.
:: NOTE: Matrix-Game-3 already has drive_matrix3.bat. This is for the older Matrix-Game.
:: TODO: needs src\drive_matrix_game.py; no .venv (set MATRIX_GAME_PY env).

setlocal enableextensions
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

set PY=%~dp0.venv\Scripts\python.exe
if not defined MATRIX_GAME_PY set MATRIX_GAME_PY=python
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests
set MODEL_NAME=matrix-game
set MATRIX_GAME_REPO=C:\workspace\world\Matrix-Game
set LOG=%~dp0drive_matrix_game.log

if not exist "%PY%" ( echo ERROR: venv python not found: %PY% & exit /b 2 )
if not exist "%MATRIX_GAME_REPO%" ( echo ERROR: repo not found: %MATRIX_GAME_REPO% & exit /b 2 )
if not exist "%~dp0src\drive_matrix_game.py" ( echo ERROR: src\drive_matrix_game.py not yet written & exit /b 2 )

echo === Matrix-Game staging into MIND-tests  ^|  model=%MODEL_NAME% ===
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_dreamx.ps1" "%LOG%" "%PY%" "src\drive_matrix_game.py" "--gt-root" "%GT_ROOT%" "--test-root" "%MIND_TESTS%" "--model-name" "%MODEL_NAME%" "--repo" "%MATRIX_GAME_REPO%" "--py" "%MATRIX_GAME_PY%" "--perspective" "1st_data" %*
set EXIT_CODE=%ERRORLEVEL%
if not %EXIT_CODE%==0 ( exit /b %EXIT_CODE% )
call "%~dp0run_mind.bat" "%MODEL_NAME%"
exit /b %ERRORLEVEL%
