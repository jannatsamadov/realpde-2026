# RealPDE 2026 — Track 1 (Sim2Real)

Our entry for the **NeurIPS 2026 RealPDE Competition, Track 1: Sim-to-Real
transfer learning**.

- Competition site: https://realpdecompetition.github.io/
- Track 1 on Codabench: https://www.codabench.org/competitions/17363/
- Track 2 (LTTTA) on Codabench: https://www.codabench.org/competitions/17385/
- Data release: https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data
- Benchmark the competition builds on: https://github.com/AI4Science-WestlakeU/RealPDEBench

## What the competition asks

Forecast the real, measured flow around a **NACA4418 airfoil**: given 20 frames
of a velocity field, predict the next 20.

```
input   (N, 20, 32, 64, 3)   ->   output  (N, 20, 32, 64, 3)
```

Each frame is a 32x64 grid; each cell holds `[u, v, p]` — the two velocity
components and pressure. On the real data pressure is never measured, so it is
zero and is not scored. `dt = 0.02 s`, so a window covers 0.4 s of flow, and the
grid spacing is 1.711 mm over a field of view of 21.9 x 10.8 cm.

The real measurements come from time-resolved **Particle Image Velocimetry** in a
water tunnel; the simulated split is matched 3D CFD. Reynolds numbers span
3750–26700 and angles of attack 0°, 5°, 10°, 15°, 20°.

**Why "Sim2Real":** simulation is abundant and clean (100 trajectories, all three
channels, almost no missing data), real measurement is scarce and noisy (81
usable trajectories, two channels, ~10% of cells lost to the laser shadow and to
PIV dropouts). The task is to learn from the first and deliver on the second.

Submissions are ranked by a single `final_score` combining five published
subscores — relative L2, turbulent kinetic energy, mean velocity profile error,
runtime, and a Safe Prediction Score over uncertainty intervals. The combination
rule itself is not published.

## Repository layout

```
src/realpde/
  data.py           windowing, Reynolds-disjoint splits, normalization
  models.py         U-Net forecaster and an advection-augmented variant
  losses.py         MSE plus terms targeting the scored quantities
  advection.py      semi-Lagrangian transport, derived physics channels
  sps.py            analytically optimal interval widths for the safety score
  local_score.py    imports the organizers' scoring.py so local numbers match
scripts/            data prep, training, evaluation, visualization, packaging
submissions/        submission template and packaging
NeurIPS 2026 RealPDE Competition/izah.md    working notes (Azerbaijani)
```

`izah.md` is the project's memory: data format, metric formulas, deadlines,
measurements, and the ideas that did not work, with reasons.

## Getting the data

The 16 GB release is not in this repository. Download `train_real.tar.gz` and
`train_sim.tar.gz` from the Hugging Face link above, unpack them into
`NeurIPS 2026 RealPDE Competition/`, then build the cache:

```bash
python scripts/prepare_data.py        # .h5 -> 32x64 float32, ~3.3 GB
python scripts/test_data.py           # shape, split and leakage checks
```

Note: `train_real/7575_0.h5` duplicates `6300_0` and is excluded — the
organizers confirmed this on 2026-08-12.

## Usage

```bash
python scripts/train.py --model advective --epochs 30      # train
python scripts/model_report.py --checkpoint <ckpt>         # one figure per model
python scripts/eval_baselines.py                           # the floor to beat
python scripts/build_submission.py --checkpoint <ckpt> --tag v1
```

`build_submission.py` verifies an archive the way the platform will — unpacks it,
imports it with this project off `sys.path`, runs `predict()` in a clean
subprocess, and scores the result with the organizers' scorer — because the
Development Phase allows only one submission per day.

Training holds GPU utilization near a target and pauses above a temperature
ceiling (`--gpu-duty`, `--max-temp`); the machine this was written on has a
failed GPU fan.

## Environment

The evaluation container is `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`,
offline, with nothing installed at evaluation time. The local venv matches the
versions that matter — torch 2.2.2+cu121, numpy 1.26 — so code that runs here
runs there. `h5py`, `scipy` and `einops` are absent from the container and must
never be imported from `submission.py`.
