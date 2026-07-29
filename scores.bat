@echo off
setlocal enableextensions enabledelayedexpansion
REM ==========================================================================
REM Print the combined MIND scores table from the EXISTING result_*.json files.
REM No scoring is run -- this just reads what's already there.
REM   scores.bat                    all result_*.json
REM   scores.bat matrix-game-3      only matching result files
REM ==========================================================================
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
set "PAT=result_*.json"
if not "%~1"=="" set "PAT=result_%~1*.json"

set "JSONS="
for %%J in ("%~dp0%PAT%") do set "JSONS=!JSONS! "%%~fJ""
if not defined JSONS ( echo   no result files matching %PAT% & exit /b 1 )
"%PY%" "%~dp0_scores_table.py" !JSONS!
endlocal
