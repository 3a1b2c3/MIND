"""Drive DreamX-World from a MIND-Data tree into the MIND test layout.

Walks MIND-Data/{1st_data,3rd_data}/test/{action_space_test,mem_test}/<gt_name>/
and, for each sample:
  1. Extracts the first frame from video.mp4 -> a temp PNG.
  2. Maps MIND action.json (ws/ad/ud/lr per-frame ticks) to DreamX-World's
     discrete action_seq letters: w/s/a/d (translate), j/l (yaw), i/k (pitch).
  3. Writes a one-entry eval.json with {image_path, caption, action_seq, action_speed_list}.
  4. Calls inference_dreamx5b.py via DreamX-World's .venv with ulysses=1.
  5. The mp4 lands at test_root/<model_name>/<perspective>/<test_type>/<gt_name>/video.mp4.

Skip-if-exists: samples whose output mp4 already exists are skipped.

Notes:
  - DreamX requires both Wan2.2-TI2V-5B and DreamX-World-5B-Cam to be downloaded
    under C:\\workspace\\world\\DreamX-World\\. Run download_models.py first.
  - DreamX inference always generates 1+4k frames at fixed sample size; we use
    video_length=121 (5s @ 24 FPS) and sample_size=704x1280 to match Matrix-Game-3.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import av

DREAMX_REPO = Path(r"C:\workspace\world\DreamX-World")
DREAMX_VENV_PY = DREAMX_REPO / ".venv" / "Scripts" / "python.exe"
DREAMX_INFER = DREAMX_REPO / "inference_dreamx5b.py"
DREAMX_CONFIG = DREAMX_REPO / "configs" / "wan2.2" / "wan_ti2v_5b.yaml"
DREAMX_WAN = DREAMX_REPO / "Wan2.2-TI2V-5B"
DREAMX_TRANSFORMER = DREAMX_REPO / "DreamX-World-5B-Cam"

TEST_TYPES = ("action_space_test", "mem_test")
PERSPECTIVES = ("1st_data", "3rd_data")

VIDEO_LENGTH = 121
HEIGHT = 704
WIDTH = 1280
FPS = 24
STEPS = 50
GUIDANCE = 3.0
SEED = 42


def extract_first_frame(video_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            img = frame.to_image()
            img.save(out_path, "PNG")
            return
    raise RuntimeError(f"No frames decoded from {video_path}")


def mind_actions_to_dreamx(action_data: list[dict]) -> tuple[list[str], list[int]]:
    """Collapse MIND per-frame (ws/ad/ud/lr) ticks into DreamX (key, speed) segments.

    DreamX takes a list of letter-combo strings and a parallel speed list; each
    segment lasts ceil(video_length / len(action_seq)) frames. We bucket the MIND
    timeline into N segments and emit the dominant action per bucket.
    """
    num_segments = 4
    if not action_data:
        return ["w"], [4]

    bucket_size = max(1, len(action_data) // num_segments)
    seq: list[str] = []
    speeds: list[int] = []
    for i in range(num_segments):
        start = i * bucket_size
        end = len(action_data) if i == num_segments - 1 else (i + 1) * bucket_size
        chunk = action_data[start:end]
        if not chunk:
            seq.append("w")
            speeds.append(4)
            continue
        # require 30% of the bucket to share a direction before emitting that key,
        # else MIND's per-tick noise pollutes the action sequence with every letter.
        thresh = max(1, int(0.3 * len(chunk)))
        ws_pos = sum(1 for d in chunk if d.get("ws", 0) > 0)
        ws_neg = sum(1 for d in chunk if d.get("ws", 0) < 0)
        ad_pos = sum(1 for d in chunk if d.get("ad", 0) > 0)
        ad_neg = sum(1 for d in chunk if d.get("ad", 0) < 0)
        ud_pos = sum(1 for d in chunk if d.get("ud", 0) > 0)
        ud_neg = sum(1 for d in chunk if d.get("ud", 0) < 0)
        lr_pos = sum(1 for d in chunk if d.get("lr", 0) > 0)
        lr_neg = sum(1 for d in chunk if d.get("lr", 0) < 0)

        keys: list[str] = []
        if ws_pos >= thresh: keys.append("w")
        elif ws_neg >= thresh: keys.append("s")
        if ad_pos >= thresh: keys.append("d")
        elif ad_neg >= thresh: keys.append("a")
        if ud_pos >= thresh: keys.append("i")
        elif ud_neg >= thresh: keys.append("k")
        if lr_pos >= thresh: keys.append("l")
        elif lr_neg >= thresh: keys.append("j")

        seq.append("".join(keys) if keys else "w")
        active = max(ws_pos + ws_neg, ad_pos + ad_neg, ud_pos + ud_neg, lr_pos + lr_neg)
        speeds.append(min(8, max(2, int(active * 6 / max(1, len(chunk))))))
    return seq, speeds


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


def output_path(test_root: Path, model_name: str, sample: dict) -> Path:
    return test_root / model_name / sample["perspective"] / sample["test_type"] / sample["gt_name"] / "video.mp4"


def run_one(sample: dict, test_root: Path, model_name: str, work_dir: Path, dry_run: bool) -> int:
    out = output_path(test_root, model_name, sample)
    if out.exists():
        print(f"[skip] {sample['perspective']}/{sample['test_type']}/{sample['gt_name']} -> {out} (exists)")
        return 0

    if sample["perspective"] == "1st_data":
        caption = "First-person view exploring a 3D virtual environment."
    else:
        caption = "Third-person view of a character exploring a 3D virtual environment."

    with open(sample["action"], encoding="utf-8") as f:
        action_json = json.load(f)
    action_seq, action_speed_list = mind_actions_to_dreamx(action_json.get("data", []))

    frame_png = work_dir / sample["perspective"] / sample["test_type"] / f"{sample['gt_name']}.png"
    extract_first_frame(sample["video"], frame_png)

    eval_entry = [{
        "image_path": str(frame_png).replace("\\", "/"),
        "caption": caption,
        "action_seq": action_seq,
        "action_speed_list": action_speed_list,
    }]
    eval_json = work_dir / sample["perspective"] / sample["test_type"] / f"{sample['gt_name']}.eval.json"
    eval_json.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_json, "w", encoding="utf-8") as f:
        json.dump(eval_entry, f, ensure_ascii=False, indent=2)

    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(DREAMX_VENV_PY), str(DREAMX_INFER),
        "--config_path",         str(DREAMX_CONFIG),
        "--model_name",          str(DREAMX_WAN),
        "--transformer_path",    str(DREAMX_TRANSFORMER),
        "--input_dir",           str(eval_json),
        "--output_dir",          str(out.parent),
        "--cam_method",          "prope",
        "--add_control_adapter",
        "--sample_size",         str(HEIGHT), str(WIDTH),
        "--video_length",        str(VIDEO_LENGTH),
        "--fps",                 str(FPS),
        "--guidance_scale",      str(GUIDANCE),
        "--num_inference_steps", str(STEPS),
        "--seed",                str(SEED),
        "--weight_dtype",        "bfloat16",
        "--ulysses_degree",      "1",
        "--ring_degree",         "1",
    ]

    print(f"\n=== {sample['perspective']}/{sample['test_type']}/{sample['gt_name']} ===")
    print(f"caption: {caption}")
    print(f"action_seq: {action_seq}  speeds: {action_speed_list}")
    print(f"out:    {out}")

    if dry_run:
        print("  [dry-run] " + " ".join(cmd))
        return 0

    env = os.environ.copy()
    # Strip anything pointing at a different Python install — host env has uv's
    # cpython-3.10 in PYTHONHOME/PYTHONPATH, which collides with DreamX's venv 3.12
    # and produces "SRE module mismatch" before argparse even loads.
    for k in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONNOUSERSITE",
              "VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT", "UV_PYTHON", "UV_PROJECT_ENVIRONMENT"):
        env.pop(k, None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    env["PYTHONPATH"] = str(DREAMX_REPO)
    env["VIRTUAL_ENV"] = str(DREAMX_REPO / ".venv")

    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=str(DREAMX_REPO), env=env)
    elapsed = time.perf_counter() - t0
    print(f"  rc={rc}  elapsed={elapsed:.1f}s")

    # DreamX names output as {name}_{action_name}.mp4; relocate to video.mp4.
    if rc == 0:
        action_name = "_".join(action_seq)
        produced = out.parent / f"{Path(frame_png).stem}_{action_name}.mp4"
        if produced.exists() and not out.exists():
            produced.rename(out)
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt-root",    type=Path, required=True, help="MIND-Data root")
    parser.add_argument("--test-root",  type=Path, required=True, help="Where to put generated test videos")
    parser.add_argument("--model-name", default="dreamx-world", help="Subfolder name under test-root")
    parser.add_argument("--work-dir",   type=Path, default=None, help="Temp dir for extracted first frames + eval.jsons (default: <test-root>/.frames)")
    parser.add_argument("--only",       nargs="+", help="Only run samples whose gt_name contains any of these substrings")
    parser.add_argument("--perspective", choices=PERSPECTIVES, help="Limit to one perspective")
    parser.add_argument("--test-type",   choices=TEST_TYPES, help="Limit to one test type")
    parser.add_argument("--limit",       type=int, help="Only run first N matched samples")
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()

    if not DREAMX_INFER.exists():
        print(f"FATAL: inference script not found at {DREAMX_INFER}", file=sys.stderr)
        return 2
    if not DREAMX_VENV_PY.exists():
        print(f"FATAL: python.exe not found at {DREAMX_VENV_PY}", file=sys.stderr)
        return 2
    for ckpt in (DREAMX_WAN, DREAMX_TRANSFORMER):
        if not ckpt.exists() and not args.dry_run:
            print(f"FATAL: checkpoint missing: {ckpt}\n  Run `python download_models.py` in {DREAMX_REPO} first.", file=sys.stderr)
            return 2

    work_dir = args.work_dir or (args.test_root / ".frames")
    work_dir.mkdir(parents=True, exist_ok=True)

    samples = gather_samples(args.gt_root)
    if args.perspective:
        samples = [s for s in samples if s["perspective"] == args.perspective]
    if args.test_type:
        samples = [s for s in samples if s["test_type"] == args.test_type]
    if args.only:
        samples = [s for s in samples if any(sub.lower() in s["gt_name"].lower() for sub in args.only)]
    if args.limit:
        samples = samples[: args.limit]

    if not samples:
        print("No samples matched.")
        return 1

    print(f"Will process {len(samples)} sample(s):")
    for s in samples:
        print(f"  - {s['perspective']}/{s['test_type']}/{s['gt_name']}")
    print()

    failures: list[str] = []
    for s in samples:
        rc = run_one(s, args.test_root, args.model_name, work_dir, args.dry_run)
        if rc != 0:
            failures.append(f"{s['perspective']}/{s['test_type']}/{s['gt_name']}")

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for name in failures:
            print(f"  {name}")
        return 1
    print(f"Done. {len(samples)} sample(s) produced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
