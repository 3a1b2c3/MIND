@echo off
:: Stage flashdreams-lingbot videos into MIND-tests\lingbot-flash\ for run_mind.bat scoring.
::
:: Uses the new flashdreams plugin and runs ENTIRELY in-process: the 14B
:: lingbot pipeline loads ONCE in src\drive_lingbot_flash.py and stays
:: resident across all MIND samples. Subprocess-per-sample would reload the
:: model every iteration -- a non-starter at ~100 samples.
::
:: Two flashdreams-lingbot slugs are supported:
::   lingbot-world-fast        (Wan VAE decoder, 4-step) -- default
::   lingbot-world-fast-flash  (LightTAE decoder, tighter streaming window)
::
:: Set the slug via LINGBOT_SLUG env var:
::   set LINGBOT_SLUG=lingbot-world-fast-flash
::
:: Output mp4 fps is fixed at 24 (matches MIND-Data ground truth).
::
:: Usage:
::   drive_lingbot_flash.bat                              stage all samples then score
::   drive_lingbot_flash.bat --dry-run                    preview commands
::   drive_lingbot_flash.bat --limit 5                    first 5 samples only
::   drive_lingbot_flash.bat --perspective 1st_data       limit to first-person
::   drive_lingbot_flash.bat --test-type mem_test         limit to memory tests
::   drive_lingbot_flash.bat --total-blocks 32            longer video
::
:: All flags pass through to src\drive_lingbot_flash.py.
::
:: NOTE: action metric will be loose. MIND's action.json (WASD) is NOT
:: converted -- we synthesize a dummy forward-walk camera trajectory.
:: lcm/visual/dino/gsc are meaningful.

setlocal enableextensions enabledelayedexpansion

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

:: Pin triton-windows to CUDA 12.8 toolkit (matches cu128 torch wheels).
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;%PATH%"

set TORCHDYNAMO_DISABLE=1
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1

:: Strip ambient venv state -- we're spawning into flashdreams's uv env, not MIND's.
set "VIRTUAL_ENV="
set "PYTHONHOME="
set "PYTHONPATH="
set "UV_PYTHON="
set "UV_PROJECT_ENVIRONMENT="

set "UV_EXE=C:\Users\kschmid\.local\bin\uv.exe"
set "FLASHDREAMS=C:\workspace\world\flashdreams"
set "DRIVE_PY=%~dp0src\drive_lingbot_flash.py"

if not defined GT_ROOT    set GT_ROOT=C:\workspace\world\MIND-Data
if not defined MIND_TESTS set MIND_TESTS=C:\workspace\world\MIND-tests
if not defined LINGBOT_SLUG set LINGBOT_SLUG=lingbot-world-fast

set MODEL_NAME=lingbot-flash
if /I "%LINGBOT_SLUG%"=="lingbot-world-fast-flash" set MODEL_NAME=lingbot-flash-flash

set LOG=%~dp0drive_lingbot_flash.log

if not defined MIND_FPS set MIND_FPS=24

if not exist "%UV_EXE%" (
    echo ERROR: uv not found at %UV_EXE%
    exit /b 2
)
if not exist "%FLASHDREAMS%\integrations\lingbot" (
    echo ERROR: flashdreams-lingbot plugin not found at %FLASHDREAMS%\integrations\lingbot
    exit /b 2
)
if not exist "%GT_ROOT%" (
    echo ERROR: gt_root not found: %GT_ROOT%
    exit /b 2
)

echo ============================================================
echo flashdreams-lingbot (in-process, model loads ONCE)
echo ============================================================
echo   gt_root   : %GT_ROOT%
echo   test_root : %MIND_TESTS%
echo   model     : %MODEL_NAME%
echo   slug      : %LINGBOT_SLUG%
echo   fps       : %MIND_FPS%
echo   log       : %LOG%
echo ============================================================

:: Mirror-test generation drives the gsc metric.
if not defined MIND_MIRROR_TEST set MIND_MIRROR_TEST=1
set MIRROR_ARG=
if "%MIND_MIRROR_TEST%"=="1" set MIRROR_ARG=--mirror-test

:: Run the driver INSIDE flashdreams's uv env so lingbot.* + flashdreams.* import.
:: cd into the flashdreams repo so uv resolves the workspace correctly.
pushd "%FLASHDREAMS%"
"!UV_EXE!" run --package flashdreams-lingbot python "%DRIVE_PY%" --gt-root "%GT_ROOT%" --test-root "%MIND_TESTS%" --model-name "%MODEL_NAME%" --slug "%LINGBOT_SLUG%" --fps %MIND_FPS% --perspective 1st_data %MIRROR_ARG% %*
set EXIT_CODE=%ERRORLEVEL%
popd

if not %EXIT_CODE%==0 (
    echo.
    echo ERROR: drive_lingbot_flash.py exited with %EXIT_CODE%
    exit /b %EXIT_CODE%
)

echo.
echo ============================================================
echo Staging done. Scoring with run_mind.bat %MODEL_NAME% ...
echo ============================================================
echo.

if not defined MIND_METRICS set MIND_METRICS=lcm,visual,dino,action,gsc
if "%MIND_METRICS%"=="" set MIND_METRICS=lcm,visual,dino,action,gsc
call "%~dp0run_mind.bat" %MODEL_NAME% "%MIND_METRICS%"

set SCORE_EXIT=%ERRORLEVEL%
if not %SCORE_EXIT%==0 (
    echo.
    echo ERROR: run_mind.bat exited with %SCORE_EXIT%
    exit /b %SCORE_EXIT%
)
echo.
echo Done. See result_%MODEL_NAME%_*.json in this directory.
endlocal
