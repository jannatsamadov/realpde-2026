"""Inspect the RealPDE HDF5 release: layout, shapes, dtypes, value ranges.

The two splits use different internal layouts (the HF README warns about this),
so this walks each file's tree rather than assuming key names.

Usage:
    python scripts/inspect_data.py --root "D:/Projects/Competitions/NeurIPS 2026 RealPDE Competition"
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


def walk(h5obj, prefix="", depth=0, max_depth=4):
    """Yield (path, object) for every node in the file."""
    if depth > max_depth:
        return
    for key, item in h5obj.items():
        path = f"{prefix}/{key}"
        yield path, item
        if isinstance(item, h5py.Group):
            yield from walk(item, path, depth + 1, max_depth)


def describe_file(path: Path, sample_values: bool = True) -> None:
    print(f"\n{'=' * 78}\n{path.name}   ({path.stat().st_size / 1024**2:,.1f} MB)\n{'=' * 78}")
    with h5py.File(path, "r") as f:
        if f.attrs:
            print("  file attrs:", {k: f.attrs[k] for k in f.attrs})
        for node_path, item in walk(f):
            if isinstance(item, h5py.Group):
                attrs = f" attrs={dict(item.attrs)}" if item.attrs else ""
                print(f"  [group]   {node_path}{attrs}")
                continue

            info = f"  [dataset] {node_path:<34} shape={str(item.shape):<22} dtype={item.dtype}"
            if item.attrs:
                info += f" attrs={dict(item.attrs)}"
            print(info)

            if not sample_values:
                continue
            # Scalars and tiny arrays: print outright.
            if item.size <= 8:
                print(f"            value = {item[()]}")
            elif sample_values and np.issubdtype(item.dtype, np.number):
                # Sample a few frames rather than reading the whole array.
                if item.ndim >= 3:
                    idx = sorted({0, item.shape[0] // 2, item.shape[0] - 1})
                    chunk = np.stack([np.asarray(item[i]) for i in idx])
                else:
                    chunk = np.asarray(item[: min(len(item), 10000)])
                finite = np.isfinite(chunk)
                nan_pct = 100.0 * (~finite).mean()
                if finite.any():
                    vals = chunk[finite]
                    print(
                        f"            min={vals.min():+.5f} max={vals.max():+.5f} "
                        f"mean={vals.mean():+.5f} std={vals.std():.5f} "
                        f"zeros={100.0 * (vals == 0).mean():.1f}% nonfinite={nan_pct:.2f}%"
                    )


def summarize_split(split_dir: Path) -> None:
    files = sorted(split_dir.glob("*.h5"))
    if not files:
        return
    print(f"\n\n{'#' * 78}\n# {split_dir}  —  {len(files)} files\n{'#' * 78}")

    # Filenames encode the operating point: {Re}_{AoA}.h5
    cases = []
    for p in files:
        stem = p.stem.split("_")
        if len(stem) == 2 and all(s.lstrip("-").isdigit() for s in stem):
            cases.append((int(stem[0]), int(stem[1])))
    if cases:
        res = sorted({c[0] for c in cases})
        aoas = sorted({c[1] for c in cases})
        print(f"  Reynolds numbers ({len(res)}): {res}")
        print(f"  Angles of attack ({len(aoas)}): {aoas}")
        missing = [(r, a) for r in res for a in aoas if (r, a) not in cases]
        print(f"  missing combinations: {missing if missing else 'none (full grid)'}")

    # Full detail on the first file only; shape-only for the rest.
    describe_file(files[0])

    print("\n  --- trajectory lengths across split ---")
    lengths = {}
    for p in files:
        with h5py.File(p, "r") as f:
            shp = None
            for node_path, item in walk(f):
                if isinstance(item, h5py.Dataset) and item.ndim == 3:
                    shp = item.shape
                    break
            lengths[p.name] = shp
    ts = [s[0] for s in lengths.values() if s]
    grids = {s[1:] for s in lengths.values() if s}
    print(f"  T: min={min(ts)} max={max(ts)} mean={np.mean(ts):.0f}   grids={grids}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    args = ap.parse_args()

    # Tarballs unpack with a nested folder of the same name.
    for split in ("train_real", "train_sim"):
        for candidate in (args.root / split / split, args.root / split):
            if candidate.is_dir() and any(candidate.glob("*.h5")):
                summarize_split(candidate)
                break


if __name__ == "__main__":
    main()
