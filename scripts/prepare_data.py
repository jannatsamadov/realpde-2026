"""Build a RAM-friendly cache of the competition data at evaluation resolution.

Reading 16 GB of HDF5 on every epoch is wasteful when the data at the resolution
we actually train on is small: subsampled to 32x64 and stored as float32, the
real split is ~1.1 GB and the simulation split ~2.5 GB. Both fit in memory, so we
convert once and memory-map thereafter.

Per split this writes:
    <split>.npy    float32, all trajectories concatenated along time
    <split>.json   per-trajectory offsets and metadata (re, aoa, length)

Channels follow each split's own content: real is [u, v] (pressure is unmeasured
and identically zero, so storing it would waste a third of the file), simulation
is [u, v, p]. The submission format's third channel is re-added at the end of the
pipeline, not here.

train_real/7575_0.h5 is excluded: the organizers confirmed it reuses 6300_0.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --splits real
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "NeurIPS 2026 RealPDE Competition"
OUT_DIR = ROOT / "data" / "processed"

# Applies to the REAL split only: train_real/7575_0.h5 reuses 6300_0's measurements.
# The simulated 7575_0 is a normal, usable trajectory.
EXCLUDED_REAL = {"7575_0.h5"}
SUB = 2  # 64x128 -> 32x64, the evaluation resolution


def excluded_for(split: str) -> set:
    return EXCLUDED_REAL if split == "real" else set()


def split_files(split: str):
    d = DATA_ROOT / f"train_{split}" / f"train_{split}"
    files = sorted(d.glob("*.h5"), key=lambda p: tuple(int(s) for s in p.stem.split("_")))
    skip = excluded_for(split)
    return [p for p in files if p.name not in skip]


def build(split: str) -> None:
    files = split_files(split)
    channels = ["u", "v", "p"] if split == "sim" else ["u", "v"]
    print(f"\n{split}: {len(files)} files, channels {channels}")

    # First pass: shapes only, so the output array can be allocated exactly.
    lengths = []
    for p in files:
        with h5py.File(p, "r") as f:
            lengths.append(f["u"].shape[0])
    total_T = int(sum(lengths))
    h, w = 64 // SUB, 128 // SUB
    nbytes = total_T * h * w * len(channels) * 4
    print(f"  total frames {total_T}, output {total_T}x{h}x{w}x{len(channels)} "
          f"= {nbytes / 1024**3:.2f} GB")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{split}.npy"
    arr = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float32, shape=(total_T, h, w, len(channels))
    )

    index, cursor = [], 0
    for p, T in zip(files, lengths):
        with h5py.File(p, "r") as f:
            for ci, ch in enumerate(channels):
                arr[cursor:cursor + T, :, :, ci] = np.asarray(
                    f[ch][:, ::SUB, ::SUB], dtype=np.float32
                )
            re_true = int(f["re"][()])
            aoa = int(f["aoa"][()])
        re_nom, aoa_nom = (int(s) for s in p.stem.split("_"))
        # The real split's in-file `re` differs from the filename (10125 -> 10142);
        # keep both, because pairing sim to real goes through the filename while
        # conditioning should use the measured value.
        index.append({
            "case": p.stem, "offset": cursor, "length": int(T),
            "re": re_true, "re_nominal": re_nom, "aoa": aoa, "aoa_nominal": aoa_nom,
        })
        cursor += T
        print(f"  {p.stem:<12} T={T:<5} re={re_true:<6} aoa={aoa:<3} -> offset {cursor - T}")

    arr.flush()
    del arr

    meta = {
        "split": split, "channels": channels, "subsample": SUB,
        "height": h, "width": w, "total_frames": total_T,
        "dt": 0.02, "dx": 0.001711, "excluded": sorted(excluded_for(split)),
        "trajectories": index,
    }
    (OUT_DIR / f"{split}.json").write_text(json.dumps(meta, indent=2))
    print(f"  wrote {out_path} and {OUT_DIR / f'{split}.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["real", "sim"], choices=["real", "sim"])
    args = ap.parse_args()
    for split in args.splits:
        build(split)


if __name__ == "__main__":
    main()
