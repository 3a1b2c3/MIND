---
name: mind-benchmark
description: "Benchmark an interactive world model (Waypoint, ABot, matrix, dreamx, sana, ...) on the MIND benchmark at C:\\workspace\\world\\MIND. Use when driving a model over MIND, scoring memory-consistency/action-control metrics, adding a new model driver, running the mirror test, or comparing models. Covers the drive->score->compare flow and its Windows/PowerShell gotchas."
---

# MIND world-model benchmarking

MIND (`C:\workspace\world\MIND`) scores how well an interactive world model keeps **memory consistency** and **action control**. GT is at `C:\workspace\world\MIND-Data` (`{1st_data,3rd_data}/test/{action_space_test,mem_test,mirror_test}`). Driven videos go to `C:\workspace\world\MIND-tests\<model>\...`. Scores are `result_<model>_<timestamp>.json` in the MIND root.

## The pipeline (3 stages)
1. **Drive** a model on MIND actions -> videos in `MIND-tests\<model>`.
2. **Score** vs GT -> `result_<model>_*.json`.
3. **Compare** via the scores table.

```
.\drive_<model>.bat --limit 5        :: smoke first
.\drive_<model>.bat                  :: full (both perspectives)
.\drive_<model>.bat --mirror-test    :: mirror clips (separately)
.\run_mind.bat --% <model> "" 1 both :: score all 5 metrics ("" = default)
.\scores.bat <model>                 :: view newest result table
```

## Metrics
`lcm`↑ (consistency, PSNR-like), `visual`↑ (quality), `dino`↓ (drift), `avg_mse`↓, `gsc`↓ (mirror consistency), `action` (control). Default = all 5. `gsc` needs mirror clips; `action` needs ViPE (JIT-compiles a CUDA ext first use; `run_mind.bat` sets up vcvars + CUDA_HOME).

## Gotchas (these bite every time)
- **PowerShell comma trap:** `run_mind.bat waypoint lcm,visual,dino ...` fails ("4th arg ... got: dino") because cmd splits commas. Fix: `run_mind.bat --% <model> "lcm,visual,dino" 1 both` (`--%` AND quotes), or just pass `""` to get the all-5 default.
- **Resume adds samples, NOT metrics:** if a model was scored with fewer metrics, `run_mind` **skips** those samples forever. To add `action`/`gsc`, `del result_<model>_*.json` then re-score fresh.
- **`.bat` in PowerShell needs `.\`**; `.bat` files must be CRLF.
- **CPU torch:** a MIND/driver venv can silently get `torch+cpu` -> `process.py` sees 0 GPUs -> `mp.Pool(0)` crash. Reinstall: `uv pip install --python <venv>\Scripts\python.exe --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130`.
- **Free the GPU** before scoring/driving (shared 32 GB); check `nvidia-smi` first.

## Adding a new model driver
Model runs in ITS OWN venv/project. Write `src\drive_<model>.py` + `drive_<model>.bat` (runs via that model's python). Pattern:
- `gather()` walks `{persp}/test/{action_space_test,mem_test}/<name>/{video.mp4,action.json}`; output `MIND-tests\<model>\{persp}\{test_type}\<name>\video.mp4`; skip-if-exists.
- **Convert MIND actions** (`ws/ad/ud/lr` tri-state: 0=neutral,1/2=dirs) to the model's control format. Examples: Waypoint (world_engine) -> Windows VK codes (W=0x57,S=0x53,A=0x41,D=0x44) + mouse for lr/ud. ABot -> `{keys:{W,A,S,D,I,J,K,L}}` (WASD move, IJKL look; lr->J/L, ud->I/K).
- **Mirror test:** `from utils.mirror_test_utils import gather_mirror_samples, MIRROR_DEFAULT_ACTION`; seed each `data-NN.png` (a PNG, not a video) + replay the `-w.json` go-then-return; output under `mirror_test`.
- **LOAD-ONCE for speed:** never subprocess-per-sample (reloads the model each time). Either loop in-process (Waypoint keeps the engine loaded), or build a manifest + one call to a batch mode (ABot's `inference.py --mind-batch`, which loops the lru_cached pipeline). Quantize for more speed (ABot `--quant fp8-per-tensor` on Blackwell).

## Existing drivers
`drive_waypoint.bat` (world_engine, scope-overworld venv), `drive_abot.bat` (ABot venv, `--mind-batch` load-once + FP8), plus `drive_matrix3/dreamx/sana/lingbot/deepverse.bat` (each needs its backend installed). `drive_all_with_mirror.bat` runs the default set. `scores.bat` shows the newest result; `scores.bat` (no arg) = all models.
