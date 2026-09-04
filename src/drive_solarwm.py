"""Drive SolarWM's MiniMax-H3 base pipeline over MIND.

IMPORTANT LIMITATION, read before trusting results: SolarWM's h3_infer.py
(examples/h3_infer.py-style standalone script) drives the raw base diffusers
MiniMaxH3ModularPipeline (t2va/fl2va workflows) -- NOT an action-conditioned
model. It accepts a prompt and an optional first frame, but there is no
per-frame action-script input anywhere in its API (see h3_infer.py's own
API notes docstring: fl2va only accepts image/last_image/height/width/
prompt/num_frames/generator/...). So unlike drive_abot.py (which converts
MIND's real ws/ad/ud/lr actions into ABot's WASD+IJKL key format), THIS
DRIVER CANNOT MAKE SOLARWM FOLLOW MIND'S ACTIONS AT ALL. It seeds each clip
from the sample's first frame with a generic prompt and lets the model
generate whatever motion it wants.

Consequence for scoring: lcm/visual/dino/avg_mse are still meaningful as a
"does it hold scene identity/quality" check, but the `action` metric is
meaningless here (there's nothing for it to measure control against) and
should be excluded when scoring: run_mind.sh solarwm lcm,visual,dino 1 both

ALSO SLOW: h3_infer.py has no load-once/batch mode (unlike ABot's
--mind-batch) -- this driver subprocesses h3_infer.py once per sample, so
the ~33B model reloads from disk every single sample. Expect this to be far
slower than drive_abot.py. Use --limit for anything beyond a tiny smoke test.

Run via SolarWM's H3 venv (see drive_solarwm.sh) -- SOLARWM_ROOT/.venv-h3.
"""
import argparse
import json
import subprocess
from pathlib import Path

import imageio.v2 as imageio

from utils.mirror_test_utils import MIRROR_ACTIONS, MIRROR_DEFAULT_ACTION, gather_mirror_samples

PERSPECTIVES = ("1st_data", "3rd_data")
TEST_TYPES = ("action_space_test", "mem_test")
GENERIC_PROMPT = (
    "Third-person view moving through the scene, stable forward motion, "
    "consistent environment, realistic camera motion."
)


def first_frame(video: Path, out_png: Path) -> bool:
    try:
        rd = imageio.get_reader(str(video))
        fr = rd.get_data(0)
        rd.close()
        imageio.imwrite(str(out_png), fr)
        return True
    except Exception as exc:
        print(f"[warn] first-frame failed {video}: {exc}")
        return False


def gather(gt_root: Path, perspectives) -> list[dict]:
    out = []
    for p in perspectives:
        for tt in TEST_TYPES:
            td = gt_root / p / "test" / tt
            if not td.is_dir():
                continue
            for sd in sorted(td.iterdir()):
                if (sd / "video.mp4").exists() and (sd / "action.json").exists():
                    out.append({
                        "perspective": p, "test_type": tt, "gt_name": sd.name,
                        "video": sd / "video.mp4",
                    })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-root", type=Path, required=True)
    ap.add_argument("--test-root", type=Path, required=True)
    ap.add_argument("--solarwm-root", type=Path, required=True)
    ap.add_argument("--model-path", default=None,
                    help="defaults to h3_infer.py's own DEFAULT_MODEL_PATH if unset")
    ap.add_argument("--model-name", default="solarwm")
    ap.add_argument("--perspective", default=None)
    ap.add_argument("--num-frames", type=int, default=124,
                    help="rounded to a valid 17n+5 in [120,360] by h3_infer.py itself")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mirror-test", action="store_true", help="run the mirror_test (go-then-return)")
    ap.add_argument("--mirror-action", default=MIRROR_DEFAULT_ACTION, choices=MIRROR_ACTIONS)
    args = ap.parse_args()

    py = args.solarwm_root / ".venv-h3" / "bin" / "python"
    infer = args.solarwm_root / "h3_infer.py"
    if not py.exists():
        raise SystemExit(f"SolarWM H3 venv not found: {py} -- run setup_env_h3.sh first")
    if not infer.exists():
        raise SystemExit(f"h3_infer.py not found: {infer}")

    perspectives = (args.perspective,) if args.perspective else PERSPECTIVES
    if args.mirror_test:
        samples = [s for s in gather_mirror_samples(args.gt_root, args.mirror_action)
                   if not args.perspective or s["perspective"] == args.perspective]
        print(f"[solarwm-mind] MIRROR action='{args.mirror_action}': {len(samples)} samples")
    else:
        samples = gather(args.gt_root, perspectives)
    samples = samples[args.start_index:]
    if args.limit:
        samples = samples[:args.limit]

    tmp_root = args.test_root / args.model_name / "_tmp_frames"
    tmp_root.mkdir(parents=True, exist_ok=True)

    ran, skipped, failed = 0, 0, 0
    for s in samples:
        out_dir = args.test_root / args.model_name / s["perspective"] / s["test_type"] / s["gt_name"]
        out_path = out_dir / "video.mp4"
        if out_path.exists():
            skipped += 1
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        png = tmp_root / f"{s['perspective']}_{s['test_type']}_{s['gt_name']}.png"
        src_png = s.get("frame_png_src")
        if src_png is not None:
            import shutil
            shutil.copy(str(src_png), str(png))
        elif not first_frame(s["video"], png):
            failed += 1
            continue

        cmd = [
            str(py), str(infer),
            "--prompt", GENERIC_PROMPT,
            "--image", str(png),
            "--num-frames", str(args.num_frames),
            "--steps", str(args.steps),
            "--output-dir", str(out_dir),
            "--name", "video",
        ]
        if args.model_path:
            cmd += ["--model-path", args.model_path]

        print(f"[solarwm-mind] {s['perspective']}/{s['test_type']}/{s['gt_name']}", flush=True)
        r = subprocess.run(cmd, cwd=str(args.solarwm_root))
        if r.returncode != 0 or not out_path.exists():
            print(f"[warn] generation failed for {s['gt_name']} (rc={r.returncode})")
            failed += 1
            continue
        ran += 1

    print(f"[solarwm-mind] done: {ran} generated, {skipped} skipped (already existed), {failed} failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
