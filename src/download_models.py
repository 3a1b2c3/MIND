"""Idempotently download every model MIND scoring needs.

Existing helpers in utils/utils.py shell out to `wget`, which isn't on Windows
by default and fails silently when called via subprocess. This script uses
stdlib urllib + huggingface_hub so it works on any platform.

Skips files that already exist. Safe to re-run.

Env vars:
    MIND_CACHE_DIR  — root for MUSIQ / Aesthetic / CLIP weights
                      (default: ~/.cache/mind)
    DINOV3_DIR      — destination dir for DINOv3 snapshot
                      (default: ./dinov3_vitb16 relative to script repo root)
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent

CACHE_DIR = Path(os.environ.get("MIND_CACHE_DIR", Path.home() / ".cache" / "mind"))
DINOV3_DIR = Path(os.environ.get("DINOV3_DIR", REPO_ROOT / "dinov3_vitb16"))

DINOV3_REPO = "facebook/dinov3-vitb16-pretrain-lvd1689m"

DIRECT_DOWNLOADS = [
    (
        "MUSIQ",
        "https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth",
        CACHE_DIR / "pyiqa_model" / "musiq_spaq_ckpt-358bb6af.pth",
    ),
    (
        "Aesthetic",
        "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_l_14_linear.pth",
        CACHE_DIR / "vitl_model" / "sa_0_4_vit_l_14_linear.pth",
    ),
    (
        "ViT-L/14 (CLIP)",
        "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
        CACHE_DIR / "clip_model" / "ViT-L-14.pt",
    ),
]


def download_one(url: str, dest: Path, label: str) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {label}: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[get ] {label}: {url}")
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=label) as bar:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)
    print(f"[done] {label}: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return True


def download_dinov3(target: Path) -> bool:
    sentinel = target / "config.json"
    if sentinel.exists():
        print(f"[skip] DINOv3: {target} already populated")
        return True
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    print(f"[get ] DINOv3 {DINOV3_REPO} -> {target}")
    snapshot_download(repo_id=DINOV3_REPO, local_dir=str(target))
    print(f"[done] DINOv3: {target}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-dinov3", action="store_true", help="Skip the DINOv3 snapshot (~1 GB).")
    parser.add_argument("--skip-direct", action="store_true", help="Skip MUSIQ/Aesthetic/CLIP.")
    args = parser.parse_args()

    print(f"CACHE_DIR = {CACHE_DIR}")
    print(f"DINOV3_DIR = {DINOV3_DIR}")
    print()

    failures: list[str] = []

    if not args.skip_direct:
        for label, url, dest in DIRECT_DOWNLOADS:
            try:
                download_one(url, dest, label)
            except Exception as e:
                print(f"[FAIL] {label}: {e}", file=sys.stderr)
                failures.append(label)

    if not args.skip_dinov3:
        try:
            download_dinov3(DINOV3_DIR)
        except Exception as e:
            print(f"[FAIL] DINOv3: {e}", file=sys.stderr)
            failures.append("DINOv3")

    print()
    if failures:
        print(f"FAILED ({len(failures)}): {', '.join(failures)}")
        return 1
    print("All models present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
