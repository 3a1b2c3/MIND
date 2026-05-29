"""Drive PKU-YuanGroup/Helios (i2v) from a MIND-Data tree.

Parallel to ``drive_matrix3_distilled.py``: walks MIND-Data, for each sample
extracts the first frame + caption, calls Helios's ``infer_helios.py`` in i2v
mode, then stages the produced mp4 to the MIND test layout.

Output layout::

    <test_root>/helios-i2v/<perspective>/<test_type>/<gt_name>/video.mp4

Cross-venv: Helios's ``infer_helios.py`` lives in its own venv
(``C:\\workspace\\world\\Helios\\.venv``) with torch 2.10 + cu128 + Windows
flash-attn-2 via HF kernels. Override with ``HELIOS_VENV_PY``.

Note on per-sample spawn: infer_helios.py loads ~80 GB of distilled weights
on every invocation. For an N-sample run this is slow. If MIND scoring on
Helios becomes a frequent operation, port this to the persistent-worker
pattern used in ``_matrix3_distilled_worker.py``.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import av

from utils.mirror_test_utils import MIRROR_ACTIONS, MIRROR_DEFAULT_ACTION, gather_mirror_samples
from utils.stats_logger import log_mp4

HELIOS_REPO = Path(r"C:\workspace\world\Helios")
HELIOS_INFER = HELIOS_REPO / "infer_helios.py"
HELIOS_HF_REPO = "BestWishYsh/Helios-Distilled"
DEFAULT_HELIOS_VENV_PY = HELIOS_REPO / ".venv" / "Scripts" / "python.exe"
HELIOS_VENV_PY = Path(os.environ.get("HELIOS_VENV_PY", str(DEFAULT_HELIOS_VENV_PY)))

TEST_TYPES = ("action_space_test", "mem_test")
PERSPECTIVES = ("1st_data", "3rd_data")


def extract_first_frame(video_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            img = frame.to_image()
            img.save(out_path, "PNG")
            return
    raise RuntimeError(f"No frames decoded from {video_path}")


def load_action_json(action_path: Path) -> dict:
    with open(action_path, encoding="utf-8") as f:
        return json.load(f)


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


def build_prompt(sample: dict) -> str:
    """Use the action.json caption when present; else perspective-flavored default."""
    try:
        info = load_action_json(sample["action"])
        cap = info.get("caption") or info.get("prompt")
        if cap:
            return str(cap)
    except (OSError, json.JSONDecodeError):
        pass
    if sample["perspective"] == "1st_data":
        return "First-person view exploring a 3D virtual environment."
    return "Third-person view of a character exploring a 3D virtual environment."


def _find_produced_mp4(output_dir: Path, before_set: set[Path]) -> Path | None:
    after = {p for p in output_dir.glob("*.mp4")}
    new = sorted(after - before_set, key=lambda p: p.stat().st_mtime, reverse=True)
    if new:
        return new[0]
    if after:
        return max(after, key=lambda p: p.stat().st_mtime)
    return None


def run_one(sample: dict, test_root: Path, model_name: str, work_dir: Path, dry_run: bool,
            height: int, width: int, num_frames: int, num_inference_steps: int,
            guidance_scale: float, seed: int, low_vram: bool) -> int:
    out = output_path(test_root, model_name, sample)
    if out.exists():
        print(f"[skip] {sample['perspective']}/{sample['test_type']}/{sample['gt_name']} -> {out} (exists)")
        return 0

    frame_png = work_dir / sample["perspective"] / sample["test_type"] / f"{sample['gt_name']}.png"
    if sample.get("frame_png_src") is not None:
        frame_png.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample["frame_png_src"], frame_png)
    else:
        extract_first_frame(sample["video"], frame_png)

    out.parent.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(sample)

    cmd = [
        str(HELIOS_VENV_PY), "-X", "utf8", str(HELIOS_INFER),
        "--base_model_path", HELIOS_HF_REPO,
        "--transformer_path", HELIOS_HF_REPO,
        "--weight_dtype", "bf16",
        "--height", str(height),
        "--width", str(width),
        "--num_frames", str(num_frames),
        "--num_inference_steps", str(num_inference_steps),
        "--fps", "24",
        "--guidance_scale", str(guidance_scale),
        "--seed", str(seed),
        "--sample_type", "i2v",
        "--image_path", str(frame_png),
        "--image_noise_sigma_min", "0.111",
        "--image_noise_sigma_max", "0.135",
        "--prompt", prompt,
        "--output_folder", str(out.parent),
    ]
    if low_vram:
        cmd += ["--enable_low_vram_mode", "--group_offloading_type", "leaf_level", "--num_blocks_per_group", "4"]

    tag = f"{sample['perspective']}/{sample['test_type']}/{sample['gt_name']}"
    print(f"\n=== {tag} ===")
    print(f"prompt: {prompt[:90]}{'...' if len(prompt) > 90 else ''}")
    print(f"out:    {out}")

    if dry_run:
        print("  [dry-run] " + " ".join(cmd))
        return 0

    env = os.environ.copy()
    # Strip cross-venv pollution before spawning Helios's interpreter (matches
    # the env scrub in drive_matrix2.py / drive_matrix3.py — avoids the SRE
    # MAGIC mismatch when MIND's 3.10 stdlib leaks into Helios's 3.11 venv).
    for k in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONNOUSERSITE",
              "VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT", "UV_PYTHON", "UV_PROJECT_ENVIRONMENT"):
        env.pop(k, None)
    # Helios's own Windows tweaks (mirror of run_helios.bat lines 158-165).
    env.setdefault("GLOO_SOCKET_IFNAME", "Wi-Fi")
    env["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    env["USE_LIBUV"] = "0"
    env["TORCH_TCPSTORE_USE_LIBUV"] = "0"
    env["PYTHONIOENCODING"] = "utf-8"

    before = {p for p in out.parent.glob("*.mp4")}
    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=str(HELIOS_REPO), env=env)
    elapsed = time.perf_counter() - t0
    print(f"  rc={rc}  elapsed={elapsed:.1f}s")

    if rc != 0:
        return rc

    # infer_helios.py writes to <output_folder>/<something>.mp4 — diff to find it.
    produced = _find_produced_mp4(out.parent, before)
    if produced is None:
        print(f"  WARN: rc=0 but no mp4 found in {out.parent}; not staged.")
        return 3
    if produced.resolve() != out.resolve():
        if out.exists():
            out.unlink()
        shutil.move(str(produced), str(out))
    log_mp4(model_name, sample["perspective"], sample["test_type"], sample["gt_name"], out)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-root", type=Path, required=True, help="MIND-Data root")
    parser.add_argument("--test-root", type=Path, required=True, help="Where to put generated test videos")
    parser.add_argument("--model-name", default="helios-i2v", help="Subfolder name under test-root")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="Temp dir for extracted first frames (default: <test-root>/.frames-helios)")
    parser.add_argument("--only", nargs="+",
                        help="Only run samples whose gt_name contains any of these substrings")
    parser.add_argument("--perspective", choices=PERSPECTIVES, help="Limit to one perspective")
    parser.add_argument("--test-type", choices=TEST_TYPES, help="Limit to one test type")
    parser.add_argument("--limit", type=int, help="Only run first N matched samples")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Skip first N matched samples after filters; used for mid-run resume.")
    parser.add_argument("--height", type=int, default=384,
                        help="Matches run_helios.bat default. 5090 / 32 GB headroom assumes this.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num-frames", type=int, default=99,
                        help="Helios constraint: must be divisible by 9 (latent chunking).")
    parser.add_argument("--num-inference-steps", type=int, default=4,
                        help="Distilled checkpoint default; bump for ablations only.")
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low-vram", action="store_true",
                        help="Pass --enable_low_vram_mode + group_offloading; required if 5090 OOMs.")
    parser.add_argument("--fps", type=int, default=24,
                        help="Accepted for bat-script consistency; infer_helios uses its own --fps.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mirror-test", action="store_true",
                        help="Also generate mirror_test outputs (additive).")
    parser.add_argument("--mirror-only", action="store_true",
                        help="Skip action_space_test + mem_test; only mirror_test. Implies --mirror-test.")
    parser.add_argument("--mirror-action", default=MIRROR_DEFAULT_ACTION, choices=MIRROR_ACTIONS,
                        help=f"Action prefix for mirror_test (default '{MIRROR_DEFAULT_ACTION}').")
    args = parser.parse_args()
    if args.mirror_only:
        args.mirror_test = True

    if not HELIOS_VENV_PY.exists():
        print(f"FATAL: Helios venv python not found at {HELIOS_VENV_PY}", file=sys.stderr)
        return 2
    if not HELIOS_INFER.exists():
        print(f"FATAL: infer_helios.py not found at {HELIOS_INFER}", file=sys.stderr)
        return 2

    work_dir = args.work_dir or (args.test_root / ".frames-helios")
    work_dir.mkdir(parents=True, exist_ok=True)

    samples = [] if args.mirror_only else gather_samples(args.gt_root)
    if args.mirror_test:
        samples += gather_mirror_samples(args.gt_root, args.mirror_action)
    if args.perspective:
        samples = [s for s in samples if s["perspective"] == args.perspective]
    if args.test_type:
        samples = [s for s in samples if s["test_type"] == args.test_type]
    if args.only:
        samples = [s for s in samples if any(sub.lower() in s["gt_name"].lower() for sub in args.only)]
    if args.start_index:
        if args.start_index >= len(samples):
            print(f"--start-index {args.start_index} is past the end of {len(samples)} matched sample(s); "
                  "nothing to do.")
            return 0
        samples = samples[args.start_index:]
    if args.limit:
        samples = samples[: args.limit]

    if not samples:
        print("No samples matched.")
        return 1

    print(f"Will process {len(samples)} sample(s) via {HELIOS_VENV_PY.name}:")
    for s in samples:
        print(f"  - {s['perspective']}/{s['test_type']}/{s['gt_name']}")
    print()

    failures: list[str] = []
    for s in samples:
        rc = run_one(
            s, args.test_root, args.model_name, work_dir, args.dry_run,
            height=args.height, width=args.width, num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale, seed=args.seed, low_vram=args.low_vram,
        )
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
