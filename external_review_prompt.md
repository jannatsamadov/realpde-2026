# One-shot prompt for an independent design opinion

Paste everything below the line into a fresh session of another model.

**What is deliberately NOT in it:** our architecture, our training recipe, our
interval strategy, every result we have measured, and our leaderboard scores.
Including any of those turns an independent design into a copy of ours and
destroys the point. Everything that IS in it is either published by the
organizers or directly verifiable from the released data.

The competition is not named on purpose — naming it invites retrieval of the
official baselines and of other people's write-ups instead of a fresh design.

---

You are designing a machine-learning system for a scientific forecasting
benchmark. You do not have access to the data. Work from the specification
below, which is complete and authoritative, plus your own knowledge of fluid
dynamics and ML. Commit to one concrete design; do not give me a survey of
options.

## The task

Given 20 consecutive frames of a two-dimensional velocity field, predict the
next 20 frames.

- Tensor shape in and out: `(N, 20, 32, 64, 3)`, float32, channels ordered
  `[u, v, p]`.
- `u, v` are the two in-plane velocity components in m/s. `p` is pressure.
- Frame interval `dt = 0.02 s`, so each window spans 0.4 s and the forecast
  horizon is another 0.4 s.
- Uniform grid, `dx = dy = 3.422 mm`, i.e. a field of view of about
  21.9 cm x 10.8 cm.

## The physical setting

Flow past a stationary airfoil in a water tunnel. Chord length is approximately
7.2 cm. The airfoil is at a fixed location in the frame in every sample and
never moves.

Two regime parameters vary across the dataset: Reynolds number and angle of
attack.

**The measurement is a planar slice of a three-dimensional flow, and it is
2D2C**: only two of the three velocity components are recorded, on one plane.
The out-of-plane component is not available anywhere in the data.

## The two data splits

**Simulated split.** 100 trajectories, 1000 frames each. 20 distinct Reynolds
numbers from 3750 to 27975, crossed with 5 angles of attack (0, 5, 10, 15, 20
degrees). All three channels `u, v, p` are present and the fields are clean.
The simulation is non-dimensionalised.

**Real split.** 81 trajectories, 282 to 868 frames each, 69085 frames in total.
18 distinct Reynolds numbers from 3750 to 26700, same 5 angles of attack (the
grid is not complete; some combinations are missing). Measured by Particle Image
Velocimetry, so the fields carry measurement noise. Only `u, v` are measured;
`p` is not measured and is stored as zeros. Values are in m/s.

Cells outside the illuminated field of view, and cells inside the airfoil body,
are stored as exact zeros in the real split.

**Scoring is on real, held-out trajectories at Reynolds numbers not present in
training.**

## What the model is given at inference

An input window, and nothing else. **Reynolds number and angle of attack are NOT
provided at scoring time** — the metadata field is empty on scored calls.
Anything the model needs to know about the regime must be inferred from the
input window itself.

## The metrics

Five subscores, each in [0, 100]. Higher is better.

Three accuracy subscores, computed on the measured channels `u, v` only
(`p` is never scored), per window and then averaged:

```
score = 100 / (1 + 0.5 * error)
```

- **rel_l2**: `||pred - target|| / ||target||` over the u, v field.
- **tke**: relative L2 of the turbulent kinetic energy field
  `0.5 * (var_t(u) + var_t(v))`, where the variance is over the 20 frames of
  the window. Note this is a relative L2 of the TKE *map*, not of a scalar.
- **mvpe**: relative L2 of the time-averaged `u, v` at a fixed grid of 36 probe
  points in the wake behind the airfoil.

One speed subscore, where `t_neural` is the mean per-sample inference wall time
in seconds:

```
time_score = 100 / (1 + sqrt(t_neural / 0.72896))
```

The constant 0.72896 s is the runtime of the numerical solver this benchmark is
positioned against.

One calibration subscore, **SPS**. The model may optionally return a lower and
an upper bound per element, the same shape as the prediction. Per element:

```
nil     = (upper - lower) / sigma_global
element = (1 - pm) * exp(-nil)     if lower <= target <= upper, else 0
```

with `sigma_global = 0.0563870259` frozen for the season. `pm = e / (0.5 + e)`
squashes each of the three accuracy errors `e`, and the three resulting branches
are combined with weights DM 0.5 / TKE 0.3 / MVPE 0.2, then multiplied by 100.
Elements outside the field of view or inside the airfoil leave the average
entirely, numerator and denominator both.

Bounds are all-or-nothing across a run: supply them on every call or on none. If
you supply none, the scorer substitutes a default band of +-5% of |prediction|.

The five subscores are combined into a single `final_score` by a rule the
organizers have **not published**.

## Hard constraints

- Inference runs offline in a container with PyTorch 2.2.2, CUDA 12.1,
  Python 3.10, numpy. **scipy, h5py, pandas, sklearn, einops and OpenCV are NOT
  installed** and nothing can be installed at evaluation time. Pure-Python
  dependencies must be vendored into the submission archive.
- The extracted submission archive, weights included, must stay under 256 MB.
- No network access at evaluation time.
- Whole-submission wall clock limit: 5 minutes.
- **External data is banned.** No extra simulations, no external datasets, no
  pretrained weights from any other source. Shortlisted methods are re-trained
  from scratch by the organizers, who hold only the released training data, so
  any recipe that cannot be reproduced from these two splits alone is
  disqualified. Augmentation of the released data is explicitly allowed.

## Training budget

One laptop RTX 2060, 6 GB VRAM, and its cooling is degraded so sustained load is
capped at roughly 70% utilisation. Assume a few hours of training per experiment,
not days, and no multi-GPU.

## What I want back

1. **One architecture**, specified concretely enough to implement: inputs,
   parameter count, how the 20 output frames are produced, and why that choice
   over the obvious alternatives.
2. **The training recipe**: loss, how you use the two splits, augmentation,
   schedule.
3. **Your strategy for the interval bounds**, including how you would choose
   their width. Show the reasoning, not just a rule.
4. **The first three measurements you would run on the data before writing any
   model code**, and what each one would change about your design depending on
   how it comes out.
5. **Predicted subscores** with an honest uncertainty range, and which single
   subscore you would attack first for the largest gain in the combined score.
6. **The two biggest risks** in your design and the cheapest experiment that
   would falsify each.

Where the specification leaves something ambiguous, state your assumption
explicitly and proceed rather than asking me. Be concrete and quantitative.
Flag anything you are guessing at.
