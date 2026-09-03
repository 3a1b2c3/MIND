#!/bin/bash
# Stage Zing-0.5 (action-conditioned ti2v) videos into MIND-tests/zing/ for
# run_mind.sh scoring. Parallel to drive_evoke.sh.
#
# Zing consumes a reference first frame + per-frame keyboard actions, which map directly
# onto MIND's action.json ws/ad/ud/lr ticks (see src/drive_zing.py), so both
# action_space_test AND mem_test run by default, across both 1st_data and 3rd_data.
#
# Unlike Evoke, zing_v0_5 is natively batch-oriented: one JSONL is built for every sample
# and the checkpoint loads exactly once.
#
# Usage:
#   drive_zing.sh                             stage 1st + 3rd, score both
#   drive_zing.sh --dry-run                   preview what would be staged
#   drive_zing.sh --limit 5                   smoke test: first 5 samples
#   drive_zing.sh --test-type mem_test        limit to memory tests
#   drive_zing.sh --perspective 1st_data      override (default = both)
#   drive_zing.sh --num-frames 121            longer rollouts (default 97)
#
# Metric selection (forwarded to run_mind.sh after staging):
#   MIND_METRICS=lcm,visual                pick a subset
#   (unset)                                default = lcm,visual,dino,gsc (no action)
#   MIND_GPUS=2                            multi-GPU scoring
#   MIND_PERSON=1st                        person = 1st | 3rd | both (default both)
#   MIND_MIRROR_TEST=0                     disable mirror_test (default on)
#   MIND_START_INDEX=N                     resume mid-run
#
# Cross-venv knobs:
#   ZING_VENV_PY=<path>                    override zing venv python
#                                           (default $HERE/../zing-world-model/.venv/bin/python)
#   ZING_REPO=<path>                       override the zing checkout
#
# All --flags pass through to src/drive_zing.py; MIND_* env vars stay in this script.
#
# Note: unlike the other drivers here, this one's default MIND_METRICS omits
# "action" already (upstream chose not to request ViPE for zing), so no extra
# ViPE-availability caveat is needed beyond what setup_mind_venv.sh documents.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

PY="$HERE/.venv/bin/python"
GT_ROOT="$HERE/../MIND-Data"
MIND_TESTS="$HERE/../MIND-tests"
LOG="$HERE/drive_zing.log"

# NOTE: zing-world-model's own venv location/setup on this Linux box has not
# been verified -- translated straight from the Windows path shape
# (C:\workspace\world\zing-world-model\.venv\Scripts\python.exe), i.e. a
# sibling of MIND under workspace/world. Adjust below (or export ZING_REPO /
# ZING_VENV_PY) if zing-world-model lives elsewhere on this box.
: "${ZING_REPO:=$HERE/../zing-world-model}"
: "${ZING_VENV_PY:=$ZING_REPO/.venv/bin/python}"

if [ ! -x "$PY" ]; then
  echo "ERROR: MIND venv python not found: $PY" >&2
  exit 2
fi
if [ ! -x "$ZING_VENV_PY" ]; then
  echo "ERROR: zing venv python not found: $ZING_VENV_PY" >&2
  exit 2
fi
if [ ! -e "$GT_ROOT" ]; then
  echo "ERROR: gt_root not found: $GT_ROOT" >&2
  exit 2
fi

echo "============================================================"
echo "Zing-0.5 (action-conditioned ti2v) staging into MIND-tests"
echo "============================================================"
echo "  gt_root      : $GT_ROOT"
echo "  test_root    : $MIND_TESTS"
echo "  model        : zing"
echo "  zing_repo    : $ZING_REPO"
echo "  zing_py      : $ZING_VENV_PY"
echo "  log          : $LOG"
echo "============================================================"

: "${MIND_START_INDEX:=0}"
: "${MIND_MIRROR_TEST:=1}"
MIRROR_ARG=()
if [ "$MIND_MIRROR_TEST" = "1" ]; then
  MIRROR_ARG=(--mirror-test)
fi

T_START=$(date +%s)

set +e
"$PY" "$HERE/run_dreamx.py" "$LOG" "$PY" "src/drive_zing.py" --gt-root "$GT_ROOT" --test-root "$MIND_TESTS" --start-index "$MIND_START_INDEX" "${MIRROR_ARG[@]}" "$@"
EXIT_CODE=$?
set -e

T_END=$(date +%s)
ELAPSED=$((T_END - T_START))
printf -v T_ELAPSED '%02d:%02d:%02d' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))
echo
echo "--- staging elapsed: $T_ELAPSED ---"

if [ "$EXIT_CODE" -ne 0 ]; then
  echo
  echo "ERROR: drive_zing.py exited with $EXIT_CODE"
  exit "$EXIT_CODE"
fi

: "${MIND_PERSON:=both}"
: "${MIND_METRICS:=lcm,visual,dino,gsc}"
if [ -z "$MIND_METRICS" ]; then
  MIND_METRICS=lcm,visual,dino,gsc
fi
: "${MIND_GPUS:=1}"

echo
echo "============================================================"
echo "Generation done. Running scoring: run_mind.sh zing \"$MIND_METRICS\" $MIND_GPUS $MIND_PERSON"
echo "============================================================"
bash "$HERE/run_mind.sh" zing "$MIND_METRICS" "$MIND_GPUS" "$MIND_PERSON"
