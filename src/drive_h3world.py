"""Drive H3-World (MiniMax-H3 LoRA) over MIND.

MIND action.json (ws/ad/ud/lr tri-state) -> H3-World's raw per-frame action
matrix (numpy [num_frames, abot_action.ACTION_DIM=17], WASD+IJKL layout --
same key semantics as drive_abot.py, since H3-World's LoRA was trained on the
same ABot-World-Explorer action schema). Only the 8 active key columns
(W/A/S/D/I/J/K/L) are set per MIND tick; the Q/E/Space and continuous
rotation/translation columns stay zero, matching how H3-World's own
--action-preset mode drives the model (key-only, no continuous deltas).

H3-World's code/abot/infer.py has no load-once/batch mode (unlike ABot's
--mind-batch): each sample is a fresh subprocess that reloads the ~33B
backbone + LoRA. This is slow (matches drive_evoke.py's per-sample pattern),
but it's the only inference entrypoint the repo exposes.

--num-frames must be 17k+5 (README's example uses 124); MIND action ticks are
truncated/padded (hold last tick) to that length. Run via H3-World's own
venv (see drive_h3world.sh).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

import os

from utils.mirror_test_utils import MIRROR_ACTIONS, MIRROR_DEFAULT_ACTION, gather_mirror_samples

H3_ROOT = Path(os.environ.get("H3_ROOT", str(Path(__file__).resolve().parent.parent.parent / "H3-World")))
H3_PY = Path(os.environ.get("H3_PY", str(H3_ROOT / ".venv" / "bin" / "python")))
INFER = H3_ROOT / "code" / "abot" / "infer.py"
CHECKPOINT = H3_ROOT / "checkpoints" / "H3-World" / "step-10000.safetensors"

PERSPECTIVES = ("1st_data", "3rd_data")
TEST_TYPES = ("action_space_test", "mem_test")
MODEL_NAME = "h3world"

NUM_FRAMES = 124  # 17*7 + 5, matches H3-World README's example
GENERIC_SCENE_PROMPT = "A person moves through the scene, stable forward motion, consistent environment."

# abot_action.ACTION_COLS layout: 11 binary keys, then 3 rotation deltas,
# then 3 translation deltas. Only the active WASD+IJKL columns are set here.
KEY_COLS = ["W", "A", "S", "D", "Q", "E", "I", "J", "K", "L", "Space"]
ACTION_DIM = len(KEY_COLS) + 3 + 3


def mind_to_action_matrix(mind_data: list, num_frames: int = NUM_FRAMES) -> np.ndarray:
    mat = np.zeros((num_frames, ACTION_DIM), dtype=np.float32)
    n = min(len(mind_data), num_frames)
    for i in range(n):
        s = mind_data[i]
        if s.get("ws") == 1:
            mat[i, KEY_COLS.index("W")] = 1
        elif s.get("ws") == 2:
            mat[i, KEY_COLS.index("S")] = 1
        if s.get("ad") == 1:
            mat[i, KEY_COLS.index("A")] = 1
        elif s.get("ad") == 2:
            mat[i, KEY_COLS.index("D")] = 1
        if s.get("ud") == 1:
            mat[i, KEY_COLS.index("I")] = 1
        elif s.get("ud") == 2:
            mat[i, KEY_COLS.index("K")] = 1
        if s.get("lr") == 1:
            mat[i, KEY_COLS.index("J")] = 1
        elif s.get("lr") == 2:
            mat[i, KEY_COLS.index("L")] = 1
    if n < num_frames and n > 0:
        mat[n:] = mat[n - 1]  # hold last tick for any padding
    return mat


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


def run_one(png: Path, action_npy: Path, out_path: Path, args: argparse.Namespace) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(H3_PY), str(INFER),
        "--checkpoint", str(args.checkpoint),
        "--first-frame", str(png),
        "--scene-prompt", args.scene_prompt,
        "--action-file", str(action_npy),
        "--num-frames", str(args.num_frames),
        "--steps", str(args.steps),
        "--seed", str(args.seed),
        "--cfg-scale", str(args.cfg_scale),
        "--out", str(out_path),
    ]
    r = subprocess.run(cmd, cwd=str(H3_ROOT))
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-root", type=Path, required=True)
    ap.add_argument("--test-root", type=Path, required=True)
    ap.add_argument("--model-name", default=MODEL_NAME)
    ap.add_argument("--perspective", default=None)
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    ap.add_argument("--scene-prompt", default=GENERIC_SCENE_PROMPT,
                     help="MIND has no per-sample caption; one generic scene prompt is used for all samples")
    ap.add_argument("--num-frames", type=int, default=NUM_FRAMES, help="must be 17k+5 (124, 243, 481, ...)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    # On by default: the mirror_test is the comparable-across-models measure
    # (gsc splits the clip in half and time-flips the return leg against the
    # outbound one), so it is the run worth getting by default. --no-mirror-test
    # selects the action_space_test / mem_test sets instead.
    ap.add_argument("--mirror-test", action=argparse.BooleanOptionalAction, default=True,
                     help="run the mirror_test (go-then-return); default on")
    ap.add_argument("--mirror-action", default=MIRROR_DEFAULT_ACTION, choices=MIRROR_ACTIONS,
                     help="mirror trajectory (default 'w')")
    args = ap.parse_args()

    if (args.num_frames - 5) % 17:
        ap.error(f"--num-frames must be 17k+5 (124, 243, 481, ...), got {args.num_frames}")
    if not H3_PY.exists():
        raise SystemExit(f"H3-World venv not found: {H3_PY} -- see H3-World/README.md Setup")
    if not args.checkpoint.exists():
        raise SystemExit(f"H3-World checkpoint not found: {args.checkpoint} -- run download_models.sh first")

    perspectives = (args.perspective,) if args.perspective else PERSPECTIVES
    if args.mirror_test:
        samples = [s for s in gather_mirror_samples(args.gt_root, args.mirror_action)
                   if not args.perspective or s["perspective"] == args.perspective]
        print(f"[h3world-mind] MIRROR action='{args.mirror_action}': {len(samples)} samples")
    else:
        samples = gather(args.gt_root, perspectives)
    samples = samples[args.start_index:]
    if args.limit:
        samples = samples[:args.limit]

    work = Path(tempfile.mkdtemp(prefix="h3world_mind_"))
    done = skipped = 0
    for s in samples:
        out_path = args.test_root / args.model_name / s["perspective"] / s["test_type"] / s["gt_name"] / "video.mp4"
        if out_path.exists():
            skipped += 1
            continue
        png = work / f"{s['perspective']}_{s['test_type']}_{s['gt_name']}.png"
        src_png = s.get("frame_png_src")
        if src_png is not None:  # mirror: first frame is already a PNG
            import shutil
            shutil.copy(str(src_png), str(png))
        elif not first_frame(s["video"], png):
            continue
        mind = json.load(open(s["action"], encoding="utf-8"))["data"]
        mat = mind_to_action_matrix(mind, args.num_frames)
        action_npy = work / f"{s['perspective']}_{s['test_type']}_{s['gt_name']}_action.npy"
        np.save(action_npy, mat)

        print(f"[h3world-mind] {s['perspective']}/{s['test_type']}/{s['gt_name']} -> {out_path}")
        rc = run_one(png, action_npy, out_path, args)
        if rc != 0:
            print(f"[h3world-mind] sample failed (rc={rc}): {s['gt_name']}")
            continue
        done += 1

    print(f"[h3world-mind] done: {done} generated, {skipped} already existed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
