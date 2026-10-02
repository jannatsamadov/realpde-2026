"""Where does Track 1's measured time actually go?

time_score is computed from the mean time per WINDOW of the whole predict()
call, and predict() does far more than the forward pass. On the leaderboard
t1_solverloop measured 16.26 ms per window, while the same model's forward pass
at the chunk size predict() uses costs 0.47 ms per window on this machine --
so about 97% of the score's input is something other than the network.

This times each phase of a built submission separately: the one-off setup
(loading the checkpoint, building the model, moving it to the device, and the
CUDA context that comes with it), the input conversion, the forward passes, and
the interval bounds. A phase that dominates is worth optimising at exactly the
same rate as the model itself, and costs no accuracy at all.

Usage:
    python scripts/profile_t1_predict.py --dir submissions/t1_solverloop
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_submission(d: Path):
    """Import the submission the way the platform does: from its own directory."""
    sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location("t1_submission", d / "submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--windows", type=int, default=273)
    args = ap.parse_args()

    d = Path(args.dir)
    if not d.is_absolute():
        d = ROOT / d

    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    from realpde.data import DEFAULT_VAL_RE, WindowDataset, split_cases

    _, val_cases = split_cases("real", DEFAULT_VAL_RE)
    ds = WindowDataset("real", cases=val_cases, stride=40, normalize=False)
    n = min(args.windows, len(ds))
    x = np.stack([ds[i]["input"].numpy() for i in range(n)]).astype(np.float32)
    # The platform hands over three channels, the third unmeasured.
    x = np.concatenate([x, np.zeros(x.shape[:-1] + (1,), np.float32)], axis=-1)
    print(f"{d.name}: {n} windows, input {x.shape}\n")

    mod = load_submission(d)

    t0 = time.perf_counter()
    out = mod.predict(x, {})
    t_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = mod.predict(x, {})
    t_second = time.perf_counter() - t0

    setup = t_first - t_second
    print(f"{'first predict() call (includes setup)':<44}{t_first * 1000:9.1f} ms")
    print(f"{'second call (model already cached)':<44}{t_second * 1000:9.1f} ms")
    print(f"{'-> one-off setup: load, build, to(device), CUDA':<44}{setup * 1000:9.1f} ms")

    # Phases inside a call, timed individually where the module exposes them.
    import torch
    model, dev = mod._load_model()
    xt = torch.from_numpy((x[..., :2] - mod.MEAN[:2]) / mod.STD[:2]).float()

    def timed(fn, iters=3):
        fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t) / iters

    def forward():
        with torch.no_grad():
            outs = []
            for i in range(0, n, 64):
                outs.append(model(xt[i:i + 64].to(dev)).float().cpu())
            torch.cat(outs)

    t_fwd = timed(forward)
    # With bounds on, predict returns {"prediction", "lower", "upper"}.
    pred = np.asarray(out["prediction"] if isinstance(out, dict) else out,
                      dtype=np.float32)
    t_bounds = timed(lambda: mod._bounds(pred)) if hasattr(mod, "_bounds") else 0.0

    print(f"\n{'forward passes (chunked, as predict does)':<44}{t_fwd * 1000:9.1f} ms")
    print(f"{'interval bounds':<44}{t_bounds * 1000:9.1f} ms")
    print(f"{'the rest of a call (convert, scale, pad)':<44}"
          f"{(t_second - t_fwd - t_bounds) * 1000:9.1f} ms")

    print(f"\nper window, as time_score sees it:")
    print(f"{'  whole first call':<44}{t_first / n * 1000:9.2f} ms")
    print(f"{'  of which forward':<44}{t_fwd / n * 1000:9.2f} ms")
    print(f"{'  of which bounds':<44}{t_bounds / n * 1000:9.2f} ms")
    print(f"{'  of which setup':<44}{setup / n * 1000:9.2f} ms")
    f = lambda t: 100.0 / (1.0 + (t / 0.72896) ** 0.5)
    print(f"\ntime_score at this rate {f(t_first / n):.2f}; "
          f"without setup {f((t_first - setup) / n):.2f}; "
          f"forward only {f(t_fwd / n):.2f}")


if __name__ == "__main__":
    main()
