"""Shared helpers for mirror_test sample gathering.

MIND-Data's mirror_test/ is FLAT:
  - 50 first-frame PNGs: data-25.png ... data-74.png (one per mem_test gt_name)
  - 10 directional "go-then-return" action JSONs: -w, -s, -a, -d, -u, -down, -wl, -wr, -sl, -sr
    each is a 47-frame trajectory in standard MIND action.json format.

The gsc metric scores by splitting the generated mp4 in half and time-flipping the
second half against the first. For comparable numbers across models, each driver
generates ONE canonical mirror mp4 per first-frame (default action prefix is "w"
= forward-out / back-return). Use --mirror-action to override per-run.

Sample dict returned by gather_mirror_samples mirrors the existing per-sample
shape used by the drivers, plus:
  - "video" is None (no source video — first frame is already a PNG)
  - "frame_png_src" is the path to the PNG (drivers should copy it instead of
    calling av-extract_first_frame).
"""

from pathlib import Path

PERSPECTIVES = ("1st_data", "3rd_data")
MIRROR_ACTIONS = ("w", "s", "a", "d", "u", "down", "wl", "wr", "sl", "sr")
MIRROR_DEFAULT_ACTION = "w"


def gather_mirror_samples(gt_root: Path, action: str = MIRROR_DEFAULT_ACTION) -> list[dict]:
    """One sample per data-NN.png × <perspective>, paired with the chosen action JSON.

    Returns sample dicts compatible with each driver's existing per-sample loop:
        {perspective, test_type='mirror_test', gt_name, video=None, action, frame_png_src}
    Output path is <test_root>/<model>/<perspective>/mirror_test/<gt_name>/video.mp4.
    """
    if action not in MIRROR_ACTIONS:
        raise ValueError(f"action {action!r} not in {MIRROR_ACTIONS}")
    samples: list[dict] = []
    for perspective in PERSPECTIVES:
        mirror_dir = gt_root / perspective / "test" / "mirror_test"
        if not mirror_dir.is_dir():
            continue
        action_json = mirror_dir / f"-{action}.json"
        if not action_json.exists():
            continue
        pngs = sorted(
            p for p in mirror_dir.iterdir()
            if p.suffix.lower() == ".png" and p.stem.startswith("data-")
        )
        for png in pngs:
            samples.append({
                "perspective": perspective,
                "test_type": "mirror_test",
                "gt_name": png.stem,        # e.g. "data-25"
                "video": None,
                "action": action_json,
                "frame_png_src": png,
            })
    return samples
