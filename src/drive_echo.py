"""Drive Echo-WM (JoyAI-Echo, LTX-based) over MIND.

MIND action.json (ws/ad/ud/lr tri-state) -> Echo-WM's WASD/IJKL action-string
DSL ("w-60,a-30,none-12"), which its helpers/action_camera.py parses into a
per-frame key list and integrates into a camera trajectory. Segment durations
are counted in FRAMES (parse_action_string does `frames.extend([keys]*n)`), so
the MIND per-tick key set run-length-encodes directly onto it with no
resampling.

Key semantics match drive_h3world.py / drive_abot.py -- W/S forward-back,
A/D strafe, I/K pitch, J/L yaw -- because Echo's DSL uses the same WASD+IJKL
layout (ALLOWED_ACTION_KEYS = "wsadikjl"). Simultaneous keys concatenate into
one segment ("wl-60"), which the parser accepts as a sorted set.

Two entrypoints exist upstream and this driver exposes both:
  inference_wm.py         non-causal, 1280x704 @ 30 steps  (default here)
  inference_wm_causal.py  causal/streaming, 512x288, few-step  (--causal)

Like drive_h3world.py and drive_evoke.py, there is no load-once/batch mode:
each sample is a fresh subprocess reloading the ~47.8 GB checkpoint plus the
Gemma text encoder. Expect this to be slow.

--num-frames must be 8k+1 (LTX VAE temporal scale, see
ltx-pipelines/src/ltx_pipelines/retake.py:451); default 97.

Run via Echo's own venv (see drive_echo.sh).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import imageio.v2 as imageio

from utils.mirror_test_utils import MIRROR_ACTIONS, MIRROR_DEFAULT_ACTION, gather_mirror_samples

ECHO_ROOT = Path(os.environ.get("ECHO_ROOT", str(Path(__file__).resolve().parent.parent.parent / "JoyAI-Echo")))
ECHO_WM = ECHO_ROOT / "echo_wm"
ECHO_PY = Path(os.environ.get("ECHO_PY", str(ECHO_WM / ".venv" / "bin" / "python")))
GEMMA = ECHO_WM / "checkpoints" / "gemma-3"

PERSPECTIVES = ("1st_data", "3rd_data")
TEST_TYPES = ("action_space_test", "mem_test")
MODEL_NAME = "echo"

# LTX VAE temporal scale is 8: num_frames must satisfy (n - 1) % 8 == 0.
FRAME_SCALE = 8
NUM_FRAMES = 97  # 8*12 + 1; closest valid value to zing's 96, well inside [24, 200]

# MIND has no per-sample caption. Echo's prompt is field-structured (see
# echo_wm/PROMPT_SKILL.md); a single generic one is used for every sample so
# prompt text never varies across models or samples.
GENERIC_SCENE_PROMPT = (
    "Environment: The scene from the first frame, held consistent throughout. "
    "Character: Preserve whatever is present in the first frame without adding people. "
    "Style: Match the first frame's rendering, lighting and color exactly. "
    "Perspective: Continue from the first frame's viewpoint, moving as the actions direct. "
    "Sounds: None. "
    "Speech: None."
)

# Echo checkpoint variants, in the order they are probed on disk.
VARIANTS = ("echo-wm-base", "echo-wm-flash")


def mind_to_action_string(mind_data: list, num_frames: int) -> str:
    """Run-length-encode MIND ticks into Echo's '<keys>-<frames>' DSL.

    Ticks are truncated to num_frames and the last tick is held for any
    padding -- the same convention as drive_h3world.py and drive_zing.py, so
    the trajectories the models see stay comparable. Note MIND's main-test
    action.json files carry thousands of ticks against a ~100-frame clip, so
    this consumes only the opening slice of the trajectory; that is how every
    existing driver behaves and changing it here alone would break
    comparability.
    """
    per_frame: list[str] = []
    for tick in mind_data[:num_frames]:
        keys = []
        if tick.get("ws") == 1:
            keys.append("w")
        elif tick.get("ws") == 2:
            keys.append("s")
        if tick.get("ad") == 1:
            keys.append("a")
        elif tick.get("ad") == 2:
            keys.append("d")
        if tick.get("ud") == 1:
            keys.append("i")
        elif tick.get("ud") == 2:
            keys.append("k")
        if tick.get("lr") == 1:
            keys.append("j")
        elif tick.get("lr") == 2:
            keys.append("l")
        per_frame.append("".join(sorted(keys)) or "none")

    if not per_frame:
        per_frame = ["none"]
    # Pad with idle rather than holding the last key. drive_h3world.py holds
    # (`mat[n:] = mat[n-1]`), which is harmless on the main tests -- their
    # action.json carries thousands of ticks, so padding never triggers -- but
    # wrong on the mirror set, the only place it does. -w.json is 48 ticks of
    # 24-out/24-back; held to 97 frames it becomes "w-24,s-73", reversing three
    # times past the origin. gsc splits the clip at its midpoint expecting the
    # turnaround there, so a held return leg scores as a failed return.
    per_frame.extend(["none"] * (num_frames - len(per_frame)))

    segments: list[str] = []
    run_key, run_len = per_frame[0], 0
    for keys in per_frame:
        if keys == run_key:
            run_len += 1
        else:
            segments.append(f"{run_key}-{run_len}")
            run_key, run_len = keys, 1
    segments.append(f"{run_key}-{run_len}")
    return ",".join(segments)


def snap_frames(n: int) -> int:
    """Smallest valid 8k+1 frame count >= n (minimum 9)."""
    return 1 + FRAME_SCALE * max(1, -(-(n - 1) // FRAME_SCALE))


def resolve_checkpoint(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for variant in VARIANTS:
        candidate = ECHO_WM / "checkpoints" / f"{variant}.safetensors"
        if candidate.exists():
            return candidate
    raise SystemExit(
        f"No Echo-WM checkpoint found in {ECHO_WM / 'checkpoints'} "
        f"(looked for {', '.join(v + '.safetensors' for v in VARIANTS)}) -- "
        "run JoyAI-Echo/echo_wm/download_models.sh first"
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
                    out.append({"perspective": p, "test_type": tt, "gt_name": sd.name,
                                "video": sd / "video.mp4", "action": sd / "action.json"})
    return out


def run_one(png: Path, action_str: str, out_path: Path, args: argparse.Namespace) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    entry = "inference_wm_causal.py" if args.causal else "inference_wm.py"
    cmd = [
        str(ECHO_PY), str(ECHO_WM / entry),
        "--image", str(png),
        "--prompt", args.scene_prompt,
        "--action-str", action_str,
        "--checkpoint", str(args.checkpoint),
        "--gemma-path", str(args.gemma_path),
        "--num-frames", str(args.num_frames),
        "--fps", str(args.fps),
        "--seed", str(args.seed),
        "--output", str(out_path),
    ]
    if args.width:
        cmd += ["--width", str(args.width)]
    if args.height:
        cmd += ["--height", str(args.height)]
    # --steps / --guidance-scale only exist on the non-causal entrypoint; the
    # causal one takes an explicit --timesteps list instead.
    if not args.causal:
        cmd += ["--steps", str(args.steps)]
        if args.guidance_scale is not None:
            cmd += ["--guidance-scale", str(args.guidance_scale)]
    if not args.audio:
        cmd.append("--no-audio")
    # The HUD goes to a SECOND file (<stem>_action.mp4), so video.mp4 is clean
    # either way -- but rendering it costs time and drops a stray mp4 into
    # MIND-tests, so it is off unless asked for.
    cmd.append("--action-overlay" if args.action_overlay else "--no-action-overlay")
    r = subprocess.run(cmd, cwd=str(ECHO_WM))
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-root", type=Path, required=True)
    ap.add_argument("--test-root", type=Path, required=True)
    ap.add_argument("--model-name", default=MODEL_NAME)
    ap.add_argument("--perspective", default=None)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="default: first of echo-wm-base/echo-wm-flash found in echo_wm/checkpoints")
    ap.add_argument("--gemma-path", type=Path, default=GEMMA)
    ap.add_argument("--scene-prompt", default=GENERIC_SCENE_PROMPT,
                    help="MIND has no per-sample caption; one generic scene prompt is used for all samples")
    ap.add_argument("--causal", action="store_true",
                    help="use inference_wm_causal.py (512x288, few-step) instead of inference_wm.py")
    ap.add_argument("--num-frames", type=int, default=None,
                    help=f"must be 8k+1 (97, 105, ..., 193); default {NUM_FRAMES}, or snapped to "
                         "the trajectory length under --mirror-test")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--width", type=int, default=None, help="default: from Echo's config")
    ap.add_argument("--height", type=int, default=None, help="default: from Echo's config")
    ap.add_argument("--steps", type=int, default=30, help="non-causal entrypoint only")
    ap.add_argument("--guidance-scale", type=float, default=None, help="non-causal entrypoint only")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--audio", action="store_true",
                    help="generate the audio track too; off by default since MIND scores video only")
    ap.add_argument("--action-overlay", action="store_true",
                    help="also write <name>_action.mp4 with the WASD HUD; off by default")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the action string per sample and exit without loading the model")
    # Additive, unlike drive_h3world.py's exclusive flag: --mirror-test here
    # selects the mirror set INSTEAD of the main sets, and is opt-in.
    ap.add_argument("--mirror-test", action="store_true",
                    help="run the mirror_test (go-then-return) set instead of action_space/mem")
    ap.add_argument("--mirror-action", default=MIRROR_DEFAULT_ACTION, choices=MIRROR_ACTIONS,
                    help="mirror trajectory (default 'w')")
    args = ap.parse_args()

    explicit_frames = args.num_frames is not None
    if explicit_frames and (args.num_frames - 1) % FRAME_SCALE:
        lower = args.num_frames - ((args.num_frames - 1) % FRAME_SCALE)
        ap.error(f"--num-frames must be 8k+1 (97, 105, 113, ...), got {args.num_frames}; "
                 f"try {lower} or {lower + FRAME_SCALE}")
    if not explicit_frames:
        args.num_frames = NUM_FRAMES

    if not args.dry_run:
        if not ECHO_PY.exists():
            raise SystemExit(f"Echo-WM venv not found: {ECHO_PY} -- run JoyAI-Echo/echo_wm/setup_and_run.sh")
        args.checkpoint = resolve_checkpoint(args.checkpoint)
        if not args.gemma_path.is_dir():
            raise SystemExit(f"Gemma text encoder not found: {args.gemma_path} -- run echo_wm/download_models.sh")

    perspectives = (args.perspective,) if args.perspective else PERSPECTIVES
    if args.mirror_test:
        samples = [s for s in gather_mirror_samples(args.gt_root, args.mirror_action)
                   if not args.perspective or s["perspective"] == args.perspective]
        print(f"[echo-mind] MIRROR action='{args.mirror_action}': {len(samples)} samples")
        # The mirror trajectories are short (-w.json is 48 ticks, 24 out / 24
        # back) and gsc scores by splitting the clip at its midpoint, which is
        # only the turnaround if the clip ends when the trajectory does.
        # Generating the default 97 frames would put the turnaround a quarter
        # of the way in and pad the rest with idle. Fit the clip to the
        # trajectory unless --num-frames was given explicitly.
        if samples and not explicit_frames:
            ticks = len(json.load(open(samples[0]["action"], encoding="utf-8"))["data"])
            args.num_frames = snap_frames(ticks)
            print(f"[echo-mind] mirror trajectory is {ticks} ticks -> --num-frames {args.num_frames} "
                  f"(override with --num-frames)")
    else:
        samples = gather(args.gt_root, perspectives)
    samples = samples[args.start_index:]
    if args.limit:
        samples = samples[:args.limit]

    entry = "inference_wm_causal.py" if args.causal else "inference_wm.py"
    print(f"[echo-mind] {len(samples)} sample(s), {args.num_frames} frames, entrypoint {entry}")

    work = Path(tempfile.mkdtemp(prefix="echo_mind_"))
    done = skipped = 0
    for s in samples:
        out_path = args.test_root / args.model_name / s["perspective"] / s["test_type"] / s["gt_name"] / "video.mp4"
        if out_path.exists():
            skipped += 1
            continue

        mind = json.load(open(s["action"], encoding="utf-8"))["data"]
        action_str = mind_to_action_string(mind, args.num_frames)

        if args.dry_run:
            print(f"[echo-mind] {s['perspective']}/{s['test_type']}/{s['gt_name']}: {action_str}")
            done += 1
            continue

        png = work / f"{s['perspective']}_{s['test_type']}_{s['gt_name']}.png"
        src_png = s.get("frame_png_src")
        if src_png is not None:  # mirror: first frame is already a PNG
            shutil.copy(str(src_png), str(png))
        elif not first_frame(s["video"], png):
            continue

        print(f"[echo-mind] {s['perspective']}/{s['test_type']}/{s['gt_name']} -> {out_path}")
        rc = run_one(png, action_str, out_path, args)
        if rc != 0:
            print(f"[echo-mind] sample failed (rc={rc}): {s['gt_name']}")
            continue
        done += 1

    verb = "planned" if args.dry_run else "generated"
    print(f"[echo-mind] done: {done} {verb}, {skipped} already existed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
