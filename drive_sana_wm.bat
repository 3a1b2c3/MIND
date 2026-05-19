@echo off
:: Stage Sana world-model videos into MIND-tests\sana-wm\.
:: TODO: needs src\drive_sana_wm.py that:
::   - Walks MIND-Data first frames + action.json
::   - Invokes Sana\inference_video_scripts\inference_sana_video.py (Sana\.venv\Scripts\python.exe)
::   - Stages output to MIND-tests\sana-wm\<perspective>\<test_type>\<gt_name>\video.mp4

setlocal enableextensions
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

set PY=%~dp0.venv\Scripts\python.exe
:: Sana-WM uses a dedicated venv (.venv-wm) created by Sana\environment_setup_sana_wm.bat
:: — distinct from Sana\.venv (the older Sana env, may not exist).
set SANA_PY=C:\workspace\world\Sana\.venv-wm\Scripts\python.exe
set GT_ROOT=C:\workspace\world\MIND-Data
set MIND_TESTS=C:\workspace\world\MIND-tests
set MODEL_NAME=sana-wm
set SANA_REPO=C:\workspace\world\Sana
set SANA_ENTRY=%SANA_REPO%\inference_video_scripts\inference_sana_video.py
set LOG=%~dp0drive_sana_wm.log
if not defined MIND_FPS set MIND_FPS=24

if not exist "%PY%" ( echo ERROR: venv python not found: %PY% & exit /b 2 )
if not exist "%SANA_REPO%" ( echo ERROR: Sana repo not found: %SANA_REPO% & exit /b 2 )
if not exist "%SANA_ENTRY%" ( echo ERROR: Sana entry not found: %SANA_ENTRY% & exit /b 2 )
if not exist "%~dp0src\drive_sana_wm.py" ( echo ERROR: src\drive_sana_wm.py not yet written & exit /b 2 )

echo === Sana-WM staging into MIND-tests  ^|  model=%MODEL_NAME% ===
:: drive_sana_wm.py PERSPECTIVES tuple is ordered ("3rd_data","1st_data"), so the
:: default walk does 3rd-person samples before 1st-person. Pass --perspective on
:: the CLI to restrict to one.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_dreamx.ps1" "%LOG%" "%PY%" "src\drive_sana_wm.py" "--gt-root" "%GT_ROOT%" "--test-root" "%MIND_TESTS%" "--model-name" "%MODEL_NAME%" "--sana-repo" "%SANA_REPO%" "--sana-py" "%SANA_PY%" "--fps" "%MIND_FPS%" %*
set EXIT_CODE=%ERRORLEVEL%
if not %EXIT_CODE%==0 ( exit /b %EXIT_CODE% )

:: Explicit metric list including gsc. gsc requires per-gt_name mirror_test
:: mp4s (10 paths each) which drive_sana_wm.py does NOT yet generate — so gsc
:: rows in the result JSON will be empty until that's wired in. Override by
:: setting MIND_METRICS before calling this bat.
if not defined MIND_METRICS set MIND_METRICS=lcm,visual,dino,action,gsc
call "%~dp0run_mind.bat" "%MODEL_NAME%" "%MIND_METRICS%"
exit /b %ERRORLEVEL%
