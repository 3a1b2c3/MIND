"""Drive SANA-WM_bidirectional from MIND-Data into the MIND test layout.

Walks MIND-Data/{1st_data,3rd_data}/test/{action_space_test,mem_test}/<gt_name>/
and, for each sample whose target video.mp4 isn't already staged:
  1. Extracts the first frame from video.mp4 -> a unique PNG in <work_dir>.
  2. Writes one line to a shared prompts.txt in Sana's I2V format:
        <caption><image>/path/to/frame.png

Then calls inference_video_scripts/inference_sana_video.py ONCE with that
prompts.txt. The 1600M DiT + LTX2VAE + gemma-2-2b-it loads only once.

After inference, walks Sana's save_root for "<global_idx>. <key>.mp4" files
and relocates each to test_root/<model_name>/<perspective>/<test_type>/<gt_name>/video.mp4.

CAVEAT — this driver does NOT yet feed MIND action.json as a camera trajectory.
SANA-WM_bidirectional is a CamCtrl model (Plucker embeddings, BidirectionalGDN);
the generated videos will be first-frame-anchored I2V samples that ignore the
MIND action timeline. Wiring action.json -> Plucker poses is a follow-up.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import av

from utils.stats_logger import log_mp4

TEST_TYPES = ("action_space_test", "mem_test")
PERSPECTIVES = ("1st_data", "3rd_data")

# Defaults matched to SANA-WM_bidirectional/config.yaml (720p, image_size=720,
# DiT=SanaMSVideoCamCtrl_1600M_P1_D20, vae=LTX2VAE_diffusers).
DEFAULT_NUM_FRAMES = 121
DEFAULT_HEIGHT = 720
DEFAULT_WIDTH = 1280
DEFAULT_STEPS = 50
DEFAULT_CFG_SCALE = 6.0
DEFAULT_SEED = 42
DEFAULT_CONFIG_REL = "configs/sana_video_config/Sana_2000M_720px_ltx2vae_AdamW_fsdp.yaml"


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


def output_path(test_root: Path, model_name: str, sample: dict) -> Path:
    return test_root / model_name / sample["perspective"] / sample["test_type"] / sample["gt_name"] / "video.mp4"


def unique_stem(sample: dict) -> str:
    return f"{sample['perspective']}__{sample['test_type']}__{sample['gt_name']}"


def caption_for(sample: dict) -> str:
    if sample["perspective"] == "1st_data":
        return "First-person view exploring a 3D virtual environment."
    return "Third-person view of a character exploring a 3D virtual environment."


def sana_key(prompt_line: str) -> str:
    # Mirror DistributePromptsDataset (inference_sana_video.py L91-94):
    #   key = prompt[:50].split("/")[0] + sha256(prompt)[:10]
    return prompt_line[:50].split("/")[0] + hashlib.sha256(prompt_line.encode()).hexdigest()[:10]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt-root",    type=Path, required=True, help="MIND-Data root")
    parser.add_argument("--test-root",  type=Path, required=True, help="Where to put generated test videos")
    parser.add_argument("--model-name", default="sana-wm", help="Subfolder name under test-root")
    parser.add_argument("--sana-repo",  type=Path, required=True, help="Path to Sana checkout")
    parser.add_argument("--sana-py",    type=Path, required=True, help="Sana .venv python.exe")
    parser.add_argument("--sana-model-path", type=Path, default=None,
                        help="SANA-WM_bidirectional model dir (default: <sana-repo>/output/pretrained_models/SANA-WM_bidirectional)")
    parser.add_argument("--sana-config", type=Path, default=None,
                        help=f"Sana config yaml (default: <sana-repo>/{DEFAULT_CONFIG_REL})")
    parser.add_argument("--work-dir",   type=Path, default=None, help="Temp dir for first frames + prompts.txt (default: <test-root>/.sana_frames)")
    parser.add_argument("--only",       nargs="+", help="Only run samples whose gt_name contains any of these substrings")
    parser.add_argument("--perspective", choices=PERSPECTIVES, help="Limit to one perspective")
    parser.add_argument("--test-type",   choices=TEST_TYPES, help="Limit to one test type")
    parser.add_argument("--limit",       type=int, help="Only run first N matched samples")
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--fps",            type=int,   default=24)
    parser.add_argument("--num-frames",     type=int,   default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--height",         type=int,   default=DEFAULT_HEIGHT)
    parser.add_argument("--width",          type=int,   default=DEFAULT_WIDTH)
    parser.add_argument("--steps",          type=int,   default=DEFAULT_STEPS)
    parser.add_argument("--cfg-scale",      type=float, default=DEFAULT_CFG_SCALE)
    parser.add_argument("--seed",           type=int,   default=DEFAULT_SEED)
    args = parser.parse_args()

    sana_model_path = args.sana_model_path or (args.sana_repo / "output" / "pretrained_models" / "SANA-WM_bidirectional")
    sana_config = args.sana_config or (args.sana_repo / DEFAULT_CONFIG_REL)
    sana_entry = args.sana_repo / "inference_video_scripts" / "inference_sana_video.py"

    for path, label in [
        (sana_entry, "Sana entry"),
        (sana_config, "Sana config"),
        (args.sana_py, "Sana python"),
        (sana_model_path, "SANA-WM model dir"),
    ]:
        if not path.exists() and not args.dry_run:
            print(f"FATAL: {label} not found: {path}", file=sys.stderr)
            return 2

    work_dir = args.work_dir or (args.test_root / ".sana_frames")
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

    pending_lines: list[str] = []
    pending_meta: list[dict] = []
    skipped = 0
    for s in samples:
        target = output_path(args.test_root, args.model_name, s)
        if target.exists():
            skipped += 1
            continue
        stem = unique_stem(s)
        frame_png = work_dir / f"{stem}.png"
        if not frame_png.exists():
            extract_first_frame(s["video"], frame_png)
        prompt = caption_for(s)
        # Sana I2V format: "<caption><image>/abs/path/to/frame.png"
        line = f"{prompt}<image>{str(frame_png).replace(os.sep, '/')}"
        pending_lines.append(line)
        pending_meta.append({
            "sample": s,
            "stem": stem,
            "frame_png": frame_png,
            "target": target,
            "key": sana_key(line),
            "global_idx": len(pending_meta),
        })

    print(f"Will process {len(pending_lines)} sample(s) (skipped {skipped} already-staged):")
    for m in pending_meta:
        s = m["sample"]
        print(f"  - {s['perspective']}/{s['test_type']}/{s['gt_name']}  key={m['key']}")
    print()
    if not pending_lines:
        print("Nothing to do.")
        return 0

    txt_file = work_dir / "prompts.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("\n".join(pending_lines) + "\n")
    print(f"Sana prompts -> {txt_file}\n")

    cmd = [
        str(args.sana_py), str(sana_entry),
        "--config",               str(sana_config),
        "--model_path",           str(sana_model_path),
        "--txt_file",             str(txt_file),
        "--dataset",              "mind_eval",
        "--custom_height_width",  str(args.height), str(args.width),
        "--num_frames",           str(args.num_frames),
        "--fps",                  str(args.fps),
        "--cfg_scale",            str(args.cfg_scale),
        "--step",                 str(args.steps),
        "--seed",                 str(args.seed),
    ]
    print("Inference cmd:\n  " + " ".join(cmd) + "\n")
    print("NOTE: SANA-WM is a camera-controlled world model. This driver passes only")
    print("first-frame + caption. MIND action.json -> camera trajectory wiring is TODO;")
    print("output videos will not follow MIND-specified motion.\n")

    if args.dry_run:
        print("[dry-run] not invoking inference.")
        return 0

    env = os.environ.copy()
    # Strip cross-venv pollution (host VIRTUAL_ENV / PYTHONPATH may point at MIND's env).
    for k in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONNOUSERSITE",
              "VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT", "UV_PYTHON", "UV_PROJECT_ENVIRONMENT"):
        env.pop(k, None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    env["PYTHONPATH"] = str(args.sana_repo)
    env["VIRTUAL_ENV"] = str(args.sana_repo / ".venv")
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["DISABLE_XFORMERS"] = "1"

    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=str(args.sana_repo), env=env)
    elapsed = time.perf_counter() - t0
    print(f"\nInference rc={rc}  elapsed={elapsed:.1f}s  ({elapsed/max(1,len(pending_lines)):.1f}s/sample amortized)\n")

    # Locate produced mp4s. Sana writes to <save_root>/{global_idx}. {key}.mp4.
    # save_root is derived from config + add_label + dataset inside the script; we can't
    # easily predict the path, so search the Sana repo's output tree for our keys.
    sana_output_root = args.sana_repo / "output"
    relocated = 0
    missing: list[str] = []
    for m in pending_meta:
        expected_name = f"{m['global_idx']}. {m['key']}.mp4"
        produced = next(sana_output_root.rglob(expected_name), None) if sana_output_root.exists() else None
        if produced and produced.exists():
            m["target"].parent.mkdir(parents=True, exist_ok=True)
            produced.replace(m["target"])
            relocated += 1
            s = m["sample"]
            log_mp4(args.model_name, s["perspective"], s["test_type"], s["gt_name"], m["target"])
        else:
            missing.append(f"{m['stem']} -> {expected_name}")

    print(f"Relocated {relocated}/{len(pending_meta)} mp4(s) into MIND-tests layout.")
    if missing:
        print(f"MISSING ({len(missing)}):")
        for name in missing:
            print(f"  {name}")
        return 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
