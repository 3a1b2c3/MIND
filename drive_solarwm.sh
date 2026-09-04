#!/bin/bash
# Drive SolarWM's MiniMax-H3 base pipeline (h3_infer.py) over MIND: seed each
# sample's first frame with a generic prompt, write to MIND-tests/solarwm/.
# Runs through SolarWM's own .venv-h3 venv.
#
# REAL LIMITATION (see src/drive_solarwm.py's docstring for the full story):
# SolarWM's h3_infer.py drives the raw base MiniMax-H3 pipeline, which has NO
# action-conditioning input at all -- this driver cannot make it follow
# MIND's actions the way drive_abot.sh does. Only lcm/visual/dino/avg_mse are
# meaningful here; `action` AND `gsc` (mirror-test consistency) are both
# meaningless -- gsc scores a go-then-return trajectory this model never
# saw, so --mirror-test runs still produce output but not a real gsc score.
# Score with:
#   run_mind.sh solarwm lcm,visual,dino 1 both
#
# LOAD-ONCE: builds a manifest and makes one h3_infer.py --mind-batch call --
# the ~33B model loads once and loops every sample, same idea as drive_abot.sh.
#
#   drive_solarwm.sh --limit 2                smoke
#   drive_solarwm.sh                          all 1st+3rd person
#   drive_solarwm.sh --mirror-test            mirror clips (gsc not meaningful, see above)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# NOTE: SolarWM's location on this box varies by checkout -- on the GB300 dev
# box it's been seen at /localhome/kschmid/SolarWM (not a sibling of MIND).
# Override with SOLARWM_ROOT= if the sibling-of-MIND default below is wrong.
SOLARWM_ROOT="${SOLARWM_ROOT:-$HERE/../SolarWM}"
GT_ROOT="$HERE/../MIND-Data"
MIND_TESTS="$HERE/../MIND-tests"

PY="$SOLARWM_ROOT/.venv-h3/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: SolarWM H3 venv missing at $PY" >&2
  echo "  -- run 'bash setup_env_h3.sh' in \$SOLARWM_ROOT first, or set" >&2
  echo "  SOLARWM_ROOT=/path/to/SolarWM if it's not at $SOLARWM_ROOT" >&2
  exit 1
fi

# no --perspective -> both 1st_data + 3rd_data. Restrict via: drive_solarwm.sh --perspective 3rd_data
"$PY" "$HERE/src/drive_solarwm.py" --gt-root "$GT_ROOT" --test-root "$MIND_TESTS" \
  --solarwm-root "$SOLARWM_ROOT" "$@"

echo
echo "Videos -> $MIND_TESTS/solarwm/"
echo "Now score (action metric is meaningless here, exclude it):"
echo "  run_mind.sh solarwm lcm,visual,dino 1 both"
