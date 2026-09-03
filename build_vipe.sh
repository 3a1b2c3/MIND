#!/bin/bash
# ============================================================================
# ****************************************************************************
# WARNING: UNVERIFIED / LIKELY BROKEN ON THIS BOX (GB300, aarch64, CUDA 13.2).
# This is a faithful syntactic port of build_vipe.bat, which builds ViPE's
# native CUDA extension via MSVC (vcvars64.bat + cl.exe) -- there is NO Linux
# equivalent of vcvars/cl.exe, so the "load MSVC env" step below is a stub
# that can only ever fail loudly, not actually substitute gcc/nvcc for it.
# Per the mind-benchmark skill's own documented gotchas, the REAL blocker is
# upstream of the compiler anyway: ViPE needs torch 2.10.0+cu128 and a
# Windows/cp310-specific prebuilt flash_attn wheel (mjun0812's release) that
# has no aarch64/Linux equivalent -- matching tonight's repeated finding on
# this exact box that flash-attn/xformers have no aarch64 wheels at all.
# Treat the 'action' (ViPE) metric as PROBABLY UNAVAILABLE on this box,
# regardless of whether this script is ever made to technically run. If this
# is ever actually attempted on Linux, replace the vcvars step with whatever
# this box's native toolchain provides (gcc/g++ + nvcc from the CUDA 13.2
# toolkit), and hunt down/rebuild flash_attn from source for aarch64 first.
# ****************************************************************************
#
# build_vipe.sh - build ViPE in-place inside MIND's existing venv.
#
# What this does (original Windows steps, translated):
#   1) Loads MSVC build env via vcvars64.bat (so cl.exe is on PATH).
#      -- NO LINUX EQUIVALENT. See warning above.
#   2) Resolves MIND's .venv python (torch 2.10.0+cu128, sm_120 in arch_list).
#   3) Ensures matching flash-attn prebuilt wheel is installed (skipped if present).
#   4) Runs `pip install --no-build-isolation -e .` from MIND/vipe to compile
#      and link ViPE's native CUDA extensions in-place with sm_120 support.
#
# Key compile-flag detail: ViPE's specs.py passes -DUSE_CUDA so torch >= 2.10's
# compiled_autograd.h Windows guard fires and skips an if-constexpr block that
# otherwise triggers MSVC C2872 'std: ambiguous symbol' when Eigen is also in
# the include chain. (Windows-specific; may not even apply on Linux/gcc.)
#
# ViPE's setup.py downloads Eigen 3.4 automatically (USE_SYSTEM_EIGEN=0 by
# default), so we don't need a separate Eigen install.
#
# Prerequisites (Windows original):
#   - Visual Studio 2022 Community/Pro with "Desktop development with C++".
#   - CUDA Toolkit 12.x on PATH (nvcc.exe). torch 2.10.0+cu128 needs CUDA 12.x.
# On THIS box, the real prerequisites would be: gcc/g++ toolchain, CUDA 13.2
# toolkit (nvcc.exe -> nvcc), and an aarch64-compatible flash_attn build --
# none of which are confirmed present/working here.
#
# Usage:
#   bash build_vipe.sh              (build)
#   bash build_vipe.sh /clean       (clear ViPE build cache then build)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY="$HERE/.venv/bin/python"
VIPE_DIR="$HERE/vipe"
# No Linux equivalent of vcvars64.bat -- left as a path for documentation /
# error-message purposes only; the "load MSVC env" step below will always
# fail on this box (by design, see warning above).
VS_VCVARS="C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat"
# Pin CUDA Toolkit to 12.8 (matches torch 2.10.0+cu128 that ViPE is pinned to).
# Adjust to this box's CUDA 13.2 toolkit path if actually attempting this build.
CUDA_HOME="/usr/local/cuda-12.8"
CUDA_PATH="$CUDA_HOME"
PATH="$CUDA_HOME/bin:$PATH"
export CUDA_HOME CUDA_PATH PATH
# TORCH_CUDA_ARCH_LIST: include sm_120 so RTX 5090 SASS is baked into vipe_ext
# (Windows original target). On GB300 (aarch64) the relevant arch differs;
# left unpinned here since this box's target SM has not been confirmed.
TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0;10.0;12.0"
export TORCH_CUDA_ARCH_LIST

if [ ! -x "$PY" ]; then
    echo "ERROR: MIND venv python not found: $PY" >&2
    echo "Create the venv with setup_mind_venv.sh from this directory first." >&2
    exit 1
fi
if [ ! -e "$VIPE_DIR/setup.py" ]; then
    echo "ERROR: ViPE source not found at $VIPE_DIR" >&2
    exit 1
fi
if [ ! -e "$VS_VCVARS" ]; then
    echo "ERROR: vcvars64.bat not found at $VS_VCVARS" >&2
    echo "There is no Linux equivalent -- this script cannot build ViPE's" >&2
    echo "native extension on this box as-is. See warning at top of script." >&2
    exit 1
fi
if [ ! -e "$CUDA_HOME/bin/nvcc" ]; then
    echo "ERROR: CUDA nvcc not found at $CUDA_HOME/bin/nvcc" >&2
    echo "Adjust CUDA_HOME at the top of this script if your toolkit is elsewhere." >&2
    exit 1
fi

CLEAN="${1:-}"
if [ "$CLEAN" = "/clean" ] || [ "$CLEAN" = "--clean" ]; then
    echo "Clearing ViPE build cache..."
    rm -rf "$VIPE_DIR/build" "$VIPE_DIR/vipe.egg-info"
    find "$VIPE_DIR" -type d -name "__pycache__" -exec rm -rf {} +
    shift || true
fi

echo
echo "=== Loading MSVC build env (vcvars64.bat) ==="
echo "[build_vipe] No Linux equivalent of vcvars64.bat -- cannot proceed on this box."
exit 1

# --- Everything below is unreachable while the vcvars step above hard-exits,
# --- but is kept as a faithful translation of the rest of the .bat's logic
# --- for reference / in case a future gcc/nvcc-based replacement is spliced
# --- in above this point.

echo
echo "=== Tool sanity ==="
command -v nvcc >/dev/null 2>&1 && nvcc --version | grep -i "release" || true
"$PY" --version
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('arch_list:', torch.cuda.get_arch_list())"
if ! "$PY" -c "import flash_attn; print('flash_attn', flash_attn.__version__)"; then
    echo "[build_vipe] flash_attn import failed."
    echo "Install a prebuilt wheel matching torch + cu128 + cp310 + win, e.g.:"
    echo "  pip install --python $PY --no-build-isolation https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.13/flash_attn-2.8.3+cu128torch2.10-cp310-cp310-win_amd64.whl"
    echo "(NOTE: that wheel is Windows/cp310-specific -- no aarch64 equivalent is known to exist.)"
    exit 1
fi

echo
echo "=== Ensuring in-venv build deps (wheel/setuptools/ninja) for --no-build-isolation ==="
"$PY" -m pip install wheel setuptools ninja

echo
echo "=== Building ViPE editable into MIND venv ==="
echo "(Compiles native CUDA extensions; first build ~5-10 min.)"
(
    cd "$VIPE_DIR"
    "$PY" -m pip install --no-build-isolation -e .
)
RC=$?
if [ "$RC" != "0" ]; then
    echo "[build_vipe] ViPE native build failed with exit code $RC."
    echo "Common causes:"
    echo "  - USE_CUDA define missing (see specs.py)."
    echo "  - flash_attn wheel ABI mismatch with installed torch."
    echo "  - Out of disk space in TEMP during Eigen download."
    exit "$RC"
fi

echo
echo "=== Verifying ViPE import + CLI ==="
if ! "$PY" -c "import vipe; print('vipe package:', vipe.__file__)"; then
    echo "[build_vipe] post-build import of vipe failed."
    exit 1
fi

echo
echo "============================================================"
echo "build_vipe: done. ViPE installed editable into MIND venv:"
echo "  $PY"
echo "You can now run scoring with action enabled:"
echo "  bash run_mind.sh <test_subdir> lcm,visual,dino,action,gsc"
echo "============================================================"
exit 0
