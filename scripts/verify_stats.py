"""Reproduce the official sigma_global = 0.0563870 from train_real.

The Evaluation page defines it as "the mean of the u, v channel standard
deviations on the official train_real split", frozen for the season. Matching it
exactly confirms our preprocessing agrees with the organizers' on three
ambiguous points at once:

  * whether the masked cells (u == v == 0: airfoil body + outside the PIV field
    of view) are included in the statistic,
  * whether it is computed on the raw 64x128 grid or the 32x64 evaluation grid,
  * whether the excluded file train_real/7575_0.h5 is in or out.

So we compute every combination and report which one lands on the target.

Usage:
    python scripts/verify_stats.py --split-dir "<...>/train_real/train_real"
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

TARGET = 0.0563870
EXCLUDED = {"7575_0.h5"}  # reuses 6300_0 data; organizers say do not use


class Running:
    """Streaming mean/std accumulator (float64, sum of squares)."""

    def __init__(self):
        self.n = 0
        self.s = 0.0
        self.s2 = 0.0

    def add(self, arr: np.ndarray) -> None:
        a = arr.astype(np.float64, copy=False).ravel()
        self.n += a.size
        self.s += a.sum()
        self.s2 += np.dot(a, a)

    @property
    def std(self) -> float:
        if self.n == 0:
            return float("nan")
        mean = self.s / self.n
        return float(np.sqrt(max(self.s2 / self.n - mean * mean, 0.0)))

    @property
    def mean(self) -> float:
        return self.s / self.n if self.n else float("nan")


def subsample(a: np.ndarray) -> np.ndarray:
    """64x128 -> 32x64, the 2x spatial subsampling used for evaluation."""
    return a[..., ::2, ::2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "data" / "processed" / "stats_real.json")
    args = ap.parse_args()

    files = sorted(args.split_dir.glob("*.h5"))
    print(f"{len(files)} files in {args.split_dir}")

    # variant key -> {channel -> Running}
    variants = {
        f"{grid}_{mask}_{excl}": {"u": Running(), "v": Running()}
        for grid in ("full64x128", "sub32x64")
        for mask in ("withzeros", "nozeros")
        for excl in ("keep7575", "drop7575")
    }
    # Per-file std, needed for the "mean over trajectories" reading of the spec.
    per_file = {}

    for i, path in enumerate(files, 1):
        with h5py.File(path, "r") as f:
            u = np.asarray(f["u"], dtype=np.float64)
            v = np.asarray(f["v"], dtype=np.float64)
        is_excluded = path.name in EXCLUDED
        # A cell is masked where BOTH components are exactly zero for all time.
        mask_valid = ~((u == 0) & (v == 0))

        per_file[path.name] = {
            "u_std": float(u.std()), "v_std": float(v.std()),
            "u_mean": float(u.mean()), "v_mean": float(v.mean()),
            "T": int(u.shape[0]), "excluded": is_excluded,
        }

        for grid, (uu, vv, mm) in (
            ("full64x128", (u, v, mask_valid)),
            ("sub32x64", (subsample(u), subsample(v), subsample(mask_valid))),
        ):
            for mask in ("withzeros", "nozeros"):
                us, vs = (uu, vv) if mask == "withzeros" else (uu[mm], vv[mm])
                for excl in ("keep7575", "drop7575"):
                    if excl == "drop7575" and is_excluded:
                        continue
                    key = f"{grid}_{mask}_{excl}"
                    variants[key]["u"].add(us)
                    variants[key]["v"].add(vs)

        if i % 10 == 0 or i == len(files):
            print(f"  {i}/{len(files)}")

    print(f"\n{'variant':<40} {'std_u':>9} {'std_v':>9} {'mean(std)':>11} {'|diff|':>10}")
    print("-" * 82)
    results = {}
    best = None
    for key, chans in sorted(variants.items()):
        su, sv = chans["u"].std, chans["v"].std
        agg = 0.5 * (su + sv)
        diff = abs(agg - TARGET)
        results[key] = {"std_u": su, "std_v": sv, "mean_std": agg, "abs_diff": diff}
        flag = ""
        if best is None or diff < best[1]:
            best = (key, diff)
        print(f"{key:<40} {su:9.6f} {sv:9.6f} {agg:11.7f} {diff:10.7f}{flag}")

    # Alternative reading: average each trajectory's std, then average channels.
    for excl in ("keep7575", "drop7575"):
        sel = [s for n, s in per_file.items() if not (excl == "drop7575" and s["excluded"])]
        agg = 0.5 * (np.mean([s["u_std"] for s in sel]) + np.mean([s["v_std"] for s in sel]))
        diff = abs(agg - TARGET)
        key = f"per_trajectory_then_mean_{excl}"
        results[key] = {"mean_std": float(agg), "abs_diff": float(diff)}
        print(f"{key:<40} {'':>9} {'':>9} {agg:11.7f} {diff:10.7f}")
        if diff < best[1]:
            best = (key, diff)

    print("-" * 82)
    print(f"TARGET (official sigma_global){'':<10} {'':>9} {'':>9} {TARGET:11.7f}")
    print(f"\nclosest variant: {best[0]}  (|diff| = {best[1]:.7f})")
    if best[1] < 1e-5:
        print("MATCH -> preprocessing convention confirmed.")
    else:
        print("no exact match; treat sigma_global as an opaque constant (it is frozen anyway).")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"target": TARGET, "variants": results, "per_file": per_file}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
