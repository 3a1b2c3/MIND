"""Stage Zing-0.5 (action-conditioned ti2v) videos into MIND-tests/zing/ for run_mind.bat scoring.

Zing takes a reference first frame + a per-frame keyboard action array, which maps cleanly onto
MIND's action.json (ws/ad/ud/lr ticks). Unlike the Evoke/Helios drivers -- which spawn a persistent
worker and feed it one sample at a time -- zing_v0_5 is natively batch-oriented: it reads a JSONL of
samples, loads the checkpoint once, and writes <sample_id>.mp4 per line. So this driver builds a
single JSONL for every MIND sample, invokes zing once, then copies the results into the MIND-tests
layout.

Action mapping (same tick convention as drive_lingbot_v2.build_action_dir):
    ws == 1 -> W    ws == 2 -> S
    ad == 1 -> A    ad == 2 -> D
    ud == 1 -> I    ud == 2 -> K
    lr == 1 -> J    lr == 2 -> L
into zing's action_keys order ["w", "a", "s", "d", "i", "j", "k", "l"].
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import av

ZING_REPO = Path(os.environ.get("ZING_REPO", str(Path(__file__).resolve().parent.parent.parent / "zing-world-model")))
DEFAULT_ZING_VENV_PY = ZING_REPO / ".venv" / "Scripts" / "python.exe"
ZING_VENV_PY = Path(os.environ.get("ZING_VENV_PY", str(DEFAULT_ZING_VENV_PY)))

MODEL_NAME = "zing"
TEST_TYPES = ("action_space_test", "mem_test")
PERSPECTIVES = ("1st_data", "3rd_data")

# zing's action_keys order; index i of each per-frame vector corresponds to ACTION_KEYS[i].
ACTION_KEYS = ["w", "a", "s", "d", "i", "j", "k", "l"]
# (mind_field, value) -> index into ACTION_KEYS
ACTION_MAP = {
    ("ws", 1): 0, ("ad", 1): 1, ("ws", 2): 2, ("ad", 2): 3,
    ("ud", 1): 4, ("lr", 1): 5, ("ud", 2): 6, ("lr", 2): 7,
}

# Matches the floor the other MIND drivers enforce (drive_evoke.py) so short samples stay generatable.
MIN_FRAMES = 33


def extract_first_frame(video_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frame.to_image().save(out_path, "PNG")
            return
    raise RuntimeError(f"No frames decoded from {video_path}")


def gather_samples(gt_root: Path) -> list[dict]:
    samples: list[dict] = []
    for perspective in PERSPECTIVES:
        for test_type in TEST_TYPES:
            type_dir = gt_root / perspective / "test" / test_type
            if not type_dir.is_dir():
                continue
            for sample_dir in sorted(type_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue
                video = sample_dir / "video.mp4"
                action = sample_dir / "action.json"
                if not (video.exists() and action.exists()):
                    continue
                samples.append({
                    "perspective": perspective,
                    "test_type": test_type,
                    "gt_name": sample_dir.name,
                    "video": video,
                    "action": action,
                })
    return samples


def sample_id(sample: dict) -> str:
    """Unique, filesystem-safe id. zing sanitises to [A-Za-z0-9._-], which this already satisfies,
    so the emitted <sample_id>.mp4 maps back deterministically."""
    return f"{sample['perspective']}__{sample['test_type']}__{sample['gt_name']}"


def output_path(test_root: Path, sample: dict) -> Path:
    return test_root / MODEL_NAME / sample["perspective"] / sample["test_type"] / sample["gt_name"] / "video.mp4"


def build_prompt(sample: dict) -> str:
    """MIND's action.json carries no caption field -- same perspective-flavored fallback the
    Evoke/Helios drivers use."""
    if sample["perspective"] == "1st_data":
        return "First-person view exploring a 3D virtual environment."
    return "Third-person view of a character exploring a 3D virtual environment."


def build_actions(action_json: Path, num_frames: int) -> list[list[int]]:
    """Convert MIND ticks to zing's per-frame 8-key one-hot rows, truncating or padding (by repeating
    the final tick) to exactly num_frames -- mirroring drive_evoke.py's c2ws handling."""
    ticks = json.loads(action_json.read_text(encoding="utf-8"))["data"]
    rows: list[list[int]] = []
    for tick in ticks[:num_frames]:
        row = [0] * len(ACTION_KEYS)
        for field in ("ws", "ad", "ud", "lr"):
            idx = ACTION_MAP.get((field, tick.get(field)))
            if idx is not None:
                row[idx] = 1
        rows.append(row)
    if not rows:
        rows = [[0] * len(ACTION_KEYS)]
    while len(rows) < num_frames:
        rows.append(list(rows[-1]))
    return rows


def resolve_checkpoint() -> tuple[Path, Path]:
    """Locate the zing snapshot's pretrained dir and generator checkpoint, the same way
    run_all_lowres.bat globs for them."""
    snapshots_dir = ZING_REPO / "pretrained_models" / "models--seedleap--zing-0.5" / "snapshots"
    snapshots = sorted(p for p in snapshots_dir.glob("*") if p.is_dir())
    if not snapshots:
        raise RuntimeError(
            f"No zing snapshots under {snapshots_dir}; run download_models.bat in the zing repo first."
        )
    snapshot = snapshots[-1]
    pretrained = snapshot / "pretrained"
    checkpoint = snapshot / "generator" / "model.pt"
    if not pretrained.is_dir():
        raise RuntimeError(f"zing pretrained dir missing: {pretrained}")
    if not checkpoint.exists():
        raise RuntimeError(f"zing checkpoint missing: {checkpoint}")
    return pretrained, checkpoint


def build_messages(samples: list[dict], args: argparse.Namespace, work_dir: Path) -> Path:
    """Write the batch JSONL zing consumes, extracting each sample's reference frame alongside it."""
    frames_dir = work_dir / "first_frames"
    jsonl_path = work_dir / "mind_zing.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            sid = sample_id(sample)
            ref_png = frames_dir / f"{sid}.png"
            extract_first_frame(sample["video"], ref_png)
            actions = build_actions(sample["action"], args.num_frames)
            record = {
                "schema_version": 2,
                "sample_id": sid,
                "messages": [
                    {"role": "user", "type": "text", "content": build_prompt(sample)},
                    {
                        "role": "target",
                        "type": "video",
                        # zing resolves relative uris against its own cwd, so pass an absolute path.
                        "uri": str(ref_png),
                        "reference_frame_count": 1,
                        "output": {
                            "frames": args.num_frames,
                            "height": args.height,
                            "width": args.width,
                        },
                        "controls": [{
                            "type": "keyboard_direction_frame_interval",
                            "action_keys": ACTION_KEYS,
                            "actions": actions,
                        }],
                    },
                ],
            }
            handle.write(json.dumps(record) + "\n")
    return jsonl_path


def run_zing(jsonl_path: Path, out_dir: Path, args: argparse.Namespace) -> int:
    pretrained, checkpoint = resolve_checkpoint()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cross-venv spawn: drop this venv's markers so zing's interpreter resolves its own site-packages.
    env = dict(os.environ)
    for var in ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH"):
        env.pop(var, None)
    env["PYTHONPATH"] = str(ZING_REPO / "src")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        str(ZING_VENV_PY), "-m", "zing_v0_5",
        "--pretrained-dir", str(pretrained),
        "--checkpoint", str(checkpoint),
        "--messages", str(jsonl_path),
        "--output-dir", str(out_dir),
        "--seed", str(args.seed),
    ]
    print("  " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(ZING_REPO), env=env).returncode


def collect_outputs(samples: list[dict], out_dir: Path, test_root: Path) -> tuple[int, list[str]]:
    staged = 0
    missing: list[str] = []
    for sample in samples:
        produced = out_dir / f"{sample_id(sample)}.mp4"
        if not produced.exists():
            missing.append(sample_id(sample))
            continue
        dest = output_path(test_root, sample)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(produced, dest)
        staged += 1
    return staged, missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage Zing-0.5 videos into MIND-tests for scoring.")
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--test-type", choices=TEST_TYPES)
    parser.add_argument("--perspective", choices=PERSPECTIVES)
    # 97 frames / 352x640 is the low-res config proven out by the zing repo's own examples on this GPU.
    parser.add_argument("--num-frames", type=int, default=97)
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    # Accepted for parity with the other drive_*.bat wrappers, which pass it unconditionally.
    parser.add_argument("--mirror-test", action="store_true")
    args = parser.parse_args()

    if args.num_frames < MIN_FRAMES:
        print(f"ERROR: --num-frames must be >= {MIN_FRAMES}", file=sys.stderr)
        return 2
    if not ZING_VENV_PY.exists():
        print(f"ERROR: zing venv python not found: {ZING_VENV_PY}", file=sys.stderr)
        return 2
    if not args.gt_root.is_dir():
        print(f"ERROR: gt_root not found: {args.gt_root}", file=sys.stderr)
        return 2

    samples = gather_samples(args.gt_root)
    if args.perspective:
        samples = [s for s in samples if s["perspective"] == args.perspective]
    if args.test_type:
        samples = [s for s in samples if s["test_type"] == args.test_type]
    samples = samples[args.start_index:]
    if args.limit:
        samples = samples[: args.limit]

    if not samples:
        print("ERROR: no samples matched the given filters.", file=sys.stderr)
        return 1

    print(f"Samples to stage: {len(samples)}  ({args.num_frames} frames @ {args.width}x{args.height})")
    if args.dry_run:
        for sample in samples:
            print(f"  {sample_id(sample)} -> {output_path(args.test_root, sample)}")
        return 0

    work_dir = args.work_dir or (args.test_root / MODEL_NAME / "_work")
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir = work_dir / "raw_outputs"

    print("Building batch JSONL (extracting reference frames)...", flush=True)
    jsonl_path = build_messages(samples, args, work_dir)

    print(f"Running zing on {len(samples)} samples (checkpoint loads once)...", flush=True)
    code = run_zing(jsonl_path, out_dir, args)
    if code != 0:
        print(f"ERROR: zing exited with {code}", file=sys.stderr)
        return code

    staged, missing = collect_outputs(samples, out_dir, args.test_root)
    print(f"Staged {staged}/{len(samples)} videos into {args.test_root / MODEL_NAME}")
    if missing:
        print(f"WARNING: {len(missing)} sample(s) produced no video:", file=sys.stderr)
        for sid in missing[:5]:
            print(f"  {sid}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
