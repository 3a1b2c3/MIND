@echo off
REM build_vipe.bat - build ViPE in-place inside DeepVerse's existing venv.
REM
REM What this does:
REM   1) Loads MSVC build env via vcvars64.bat (so cl.exe is on PATH).
REM   2) Resolves DeepVerse's .venv python (the one with torch 2.7.0+cu128).
REM   3) Installs ViPE's pip requirements pinned to the matching cu128 wheels.
REM   4) Runs `pip install --no-build-isolation -e .` from MIND\vipe to compile
REM      and link ViPE's native CUDA extensions in-place.
REM
REM ViPE's setup.py downloads Eigen 3.4 automatically (USE_SYSTEM_EIGEN=0 by
REM default), so we don't need a separate Eigen install.
REM
REM Prerequisites:
REM   - Visual Studio 2022 Community/Pro with "Desktop development with C++".
REM   - CUDA Toolkit on PATH (nvcc.exe). NOTE: torch+cu128 expects CUDA 12.x;
REM     CUDA 13.x may or may not work. If the build fails with toolkit-version
REM     errors, install CUDA 12.8 alongside (they coexist) and prepend its
REM     bin folder to PATH before running this bat.
REM
REM Usage:
REM   .\build_vipe.bat              (build)
REM   .\build_vipe.bat /clean       (clear ViPE build cache then build)

setlocal enableextensions enabledelayedexpansion

set "PY=C:\workspace\world\DeepVerse\.venv\Scripts\python.exe"
set "VIPE_DIR=%~dp0vipe"
set "VS_VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
REM Pin CUDA Toolkit to 12.8 (matches torch+cu128). 13.0 is also installed but
REM nvcc 13.x against cu128-built torch leads to mismatch errors at link time.
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%CUDA_HOME%\libnvvp;%PATH%"

if not exist "%PY%" (
    echo ERROR: DeepVerse venv python not found: %PY%
    exit /b 1
)
if not exist "%VIPE_DIR%\setup.py" (
    echo ERROR: ViPE source not found at %VIPE_DIR%
    exit /b 1
)
if not exist "%VS_VCVARS%" (
    echo ERROR: vcvars64.bat not found at %VS_VCVARS%
    echo Adjust VS_VCVARS at the top of this script to your VS 2022 install path.
    exit /b 1
)
if not exist "%CUDA_HOME%\bin\nvcc.exe" (
    echo ERROR: CUDA 12.8 nvcc not found at %CUDA_HOME%\bin\nvcc.exe
    echo Adjust CUDA_HOME at the top of this script if your toolkit is elsewhere.
    exit /b 1
)

if /i "%~1"=="/clean" (
    echo Clearing ViPE build cache...
    if exist "%VIPE_DIR%\build" rmdir /s /q "%VIPE_DIR%\build"
    if exist "%VIPE_DIR%\vipe.egg-info" rmdir /s /q "%VIPE_DIR%\vipe.egg-info"
    for /f "delims=" %%D in ('dir /s /b /ad "%VIPE_DIR%\__pycache__" 2^>nul') do rmdir /s /q "%%D"
    shift
)

echo.
echo === Loading MSVC build env (vcvars64.bat) ===
call "%VS_VCVARS%" >nul
if errorlevel 1 (
    echo [build_vipe] vcvars64.bat failed.
    exit /b %ERRORLEVEL%
)
where cl >nul 2>&1 || (
    echo [build_vipe] cl.exe still not on PATH after vcvars; MSVC env load failed.
    exit /b 1
)

echo.
echo === Tool sanity ===
where nvcc 2>nul && nvcc --version | findstr /R "release"
where cl   2>nul && cl 2>&1 | findstr /R /C:"Version" | findstr /C:"19" || cl 2>&1 | findstr "^Microsoft"
"%PY%" --version
"%PY%" -c "import torch; print('torch', torch.__version__, '(needs CUDA toolkit matching torch.version.cuda =', torch.version.cuda + ')')"

echo.
echo === Filtering known-bad packages from ViPE requirements ===
REM Three classes of exclusions:
REM   1) Linux-only NVIDIA packages with no Windows wheel:
REM      nvidia-cufile-cu12 (GPUDirect Storage), nvidia-nccl-cu12,
REM      nvidia-cusparselt-cu12.
REM   2) triton==X.Y (Linux-only on PyPI; Windows uses triton-windows).
REM   3) calmsize==0.1.3 has a broken sdist that produces 'unknown-0.0.0.whl'
REM      under modern setuptools. We install it separately below with a
REM      workaround. It's only used for human-readable byte size strings.
set "VIPE_REQ_WIN=%TEMP%\vipe_requirements_win.txt"
findstr /V /B /L "nvidia-cufile-cu12 nvidia-nccl-cu12 nvidia-cusparselt-cu12 triton== calmsize==" "%VIPE_DIR%\envs\requirements.txt" > "%VIPE_REQ_WIN%"
if not exist "%VIPE_REQ_WIN%" (
    echo [build_vipe] failed to write filtered requirements file %VIPE_REQ_WIN%
    exit /b 1
)
echo Wrote filtered requirements to: %VIPE_REQ_WIN%

echo.
echo === Step 1/2: pip install -r %VIPE_REQ_WIN% --extra-index-url https://download.pytorch.org/whl/cu128 ===
"%PY%" -m pip install -r "%VIPE_REQ_WIN%" --extra-index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo [build_vipe] pip install of ViPE requirements failed.
    echo If you see another "No matching distribution" line, add the offending
    echo package name to the powershell filter regex above and rerun.
    exit /b %ERRORLEVEL%
)

echo.
echo === Step 2/2: pip install --no-build-isolation -e %VIPE_DIR% ===
echo (This compiles ViPE's native CUDA extensions; first build can take 10+ minutes.)
echo (ViPE will download Eigen 3.4 automatically into its csrc tree.)
pushd "%VIPE_DIR%"
"%PY%" -m pip install --no-build-isolation -e .
set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" (
    echo [build_vipe] ViPE native build failed with exit code %RC%.
    echo Common causes:
    echo   - CUDA Toolkit version mismatch ^(torch wants cu128 -^> CUDA 12.x; you have 13.x?^).
    echo   - Out of disk space in TEMP during Eigen download.
    echo   - Antivirus blocking nvcc.exe.
    exit /b %RC%
)

echo.
echo === Verifying ViPE import + CLI ===
"%PY%" -c "import vipe; print('vipe package:', vipe.__file__)" || (
    echo [build_vipe] post-build import of vipe failed.
    exit /b 1
)

echo.
echo ============================================================
echo build_vipe: done. ViPE installed editable into:
echo   %PY%
echo You should now be able to run from inside MIND:
echo   "%PY%" src\process.py --gt_root ... --test_root ... --metrics lcm,visual,action
echo ============================================================
exit /b 0
