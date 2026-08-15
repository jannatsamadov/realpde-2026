"""Assemble and verify a Track 1 submission archive.

Produces submissions/<tag>.zip containing:

    submission.py   the predict() entry point
    model_def.py    the network definition, vendored so no project imports leak
    model.pth       weights, stripped of optimizer state

and then verifies it the way the platform will, before a scarce daily submission
is spent on a packaging mistake:

  1. unzip into a clean directory,
  2. import submission.py from there with the project's own src/ NOT on the path,
     so an accidental `from realpde...` import fails here rather than on Codabench,
  3. reject any import of a module the evaluation image does not have,
  4. run predict() on a real validation batch, check shape, dtype and finiteness,
  5. score the output with the organizers' scoring.py,
  6. check the extracted size against the 256 MB cap.

Usage:
    python scripts/build_submission.py --checkpoint checkpoints/tke0p5_best.pt --tag unet_v1
"""

import argparse
import ast
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from realpde.data import build_datasets, denormalize
from realpde.local_score import format_scores, score_arrays

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "submissions" / "template" / "submission.py"
MODEL_SRC = ROOT / "src" / "realpde" / "models.py"
# models.py imports advection helpers, so it travels with them. Vendored flat:
# inside the archive there is no package for a relative import to resolve against.
VENDORED = {"model_def.py": MODEL_SRC, "advection.py": ROOT / "src" / "realpde" / "advection.py"}
SIZE_CAP_MB = 256

# Present in pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime, per the Submission page.
ALLOWED_TOP_LEVEL = {
    "torch", "torchvision", "torchaudio", "numpy", "PIL", "yaml", "requests",
    "tqdm", "sympy", "networkx",
    # standard library modules this submission actually uses
    "os", "sys", "math", "json", "time", "typing", "pathlib", "__future__",
    # vendored alongside submission.py
    "model_def", "advection",
}
FORBIDDEN = {"h5py", "scipy", "pandas", "matplotlib", "sklearn", "einops", "cv2"}


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def build(checkpoint: Path, tag: str, return_bounds: bool) -> Path:
    out_dir = ROOT / "submissions" / tag
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    state = torch.load(checkpoint, map_location="cpu")
    train_args = state.get("args", {})
    payload = {
        "model": state["model"],
        "config": {
            # The architecture travels with the weights: submission.py builds
            # whatever this says rather than assuming one class.
            "arch": train_args.get("model", "unet"),
            "t_in": 20, "t_out": 20, "channels": 2,
            "base": train_args.get("base", 64),
        },
    }
    torch.save(payload, out_dir / "model.pth")

    for dest, src in VENDORED.items():
        shutil.copy(src, out_dir / dest)
    text = TEMPLATE.read_text(encoding="utf-8")
    if return_bounds:
        sigma = ROOT / "checkpoints" / "sps_sigma.npz"
        if not sigma.exists():
            raise SystemExit(
                f"--bounds needs {sigma}; run scripts/calibrate_sps.py first")
        shutil.copy(sigma, out_dir / "sps_sigma.npz")
        text = text.replace("RETURN_BOUNDS = False", "RETURN_BOUNDS = True")
    (out_dir / "submission.py").write_text(text, encoding="utf-8")

    zip_path = ROOT / "submissions" / f"{tag}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out_dir.iterdir()):
            z.write(f, f.name)
    return zip_path


def check_imports(sub_dir: Path) -> None:
    print("\n[2/5] import audit")
    for name in ("submission.py", *VENDORED):
        mods = imported_modules(sub_dir / name)
        bad = mods & FORBIDDEN
        unknown = mods - ALLOWED_TOP_LEVEL - FORBIDDEN
        print(f"  {name}: {sorted(mods)}")
        if bad:
            raise SystemExit(f"  FAIL: {name} imports {sorted(bad)}, absent from the image")
        if unknown:
            print(f"  WARNING: {name} imports {sorted(unknown)} — confirm these exist in the image")
    print("  OK")


def run_isolated(sub_dir: Path, npz_in: Path, npz_out: Path) -> float:
    """Import and call predict() in a subprocess whose path excludes the project."""
    script = f'''
import sys, time, json
import numpy as np
sys.path.insert(0, r"{sub_dir}")
import submission

data = np.load(r"{npz_in}")["input"]
t0 = time.perf_counter()
out = submission.predict(data, {{}})
elapsed = time.perf_counter() - t0

if isinstance(out, dict):
    np.savez(r"{npz_out}", elapsed=np.array([elapsed]), **{{k: np.asarray(v) for k, v in out.items()}})
else:
    np.savez(r"{npz_out}", prediction=np.asarray(out), elapsed=np.array([elapsed]))
print("PREDICT_OK", elapsed)
'''
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          cwd=str(sub_dir))
    if proc.returncode != 0 or "PREDICT_OK" not in proc.stdout:
        print(proc.stdout)
        print(proc.stderr)
        raise SystemExit("  FAIL: predict() did not run in a clean process")
    return float(proc.stdout.strip().split()[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bounds", action="store_true", help="return lower/upper for SPS")
    ap.add_argument("--max-windows", type=int, default=64)
    args = ap.parse_args()

    print(f"[1/5] building from {args.checkpoint}")
    zip_path = build(args.checkpoint, args.tag, args.bounds)
    sub_dir = ROOT / "submissions" / args.tag
    print(f"  wrote {zip_path}")

    check_imports(sub_dir)

    print("\n[3/5] size")
    extracted = sum(f.stat().st_size for f in sub_dir.iterdir()) / 1024**2
    print(f"  archive {zip_path.stat().st_size / 1024**2:.2f} MB, "
          f"extracted {extracted:.2f} MB (cap {SIZE_CAP_MB} MB)")
    if extracted >= SIZE_CAP_MB:
        raise SystemExit("  FAIL: over the size cap")
    print("  OK")

    print("\n[4/5] clean-process predict()")
    _, val = build_datasets("real")
    n = min(args.max_windows, len(val))
    x = denormalize(torch.stack([val[i]["input"] for i in range(n)]), 2).numpy()
    y = denormalize(torch.stack([val[i]["target"] for i in range(n)]), 2).numpy()
    # The platform hands predict() the full 3-channel format.
    x3 = np.zeros(x.shape[:-1] + (3,), dtype=np.float32)
    x3[..., :2] = x

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        np.savez(tmp / "in.npz", input=x3)
        elapsed = run_isolated(sub_dir, tmp / "in.npz", tmp / "out.npz")
        res = dict(np.load(tmp / "out.npz"))

    pred = res["prediction"]
    print(f"  returned {pred.shape} {pred.dtype}   {elapsed:.3f}s for {n} windows "
          f"({1000 * elapsed / n:.2f} ms/window)")
    assert pred.shape == (n, 20, 32, 64, 3), f"  FAIL: wrong shape {pred.shape}"
    assert np.all(np.isfinite(pred)), "  FAIL: non-finite values in the prediction"
    has_bounds = "lower" in res and "upper" in res
    print(f"  bounds returned: {has_bounds}")
    if has_bounds:
        assert res["lower"].shape == pred.shape and res["upper"].shape == pred.shape
        assert np.all(res["lower"] <= res["upper"]), "  FAIL: reversed bounds"
    print("  OK")

    print("\n[5/5] scoring with the official scoring.py")
    y3 = np.zeros(y.shape[:-1] + (3,), dtype=np.float32)
    y3[..., :2] = y
    scores = score_arrays(
        pred, y3, mean_t_neural_s=elapsed / n,
        lower=res.get("lower"), upper=res.get("upper"),
    )
    print(format_scores(scores))

    (ROOT / "submissions" / f"{args.tag}_report.json").write_text(
        json.dumps({"checkpoint": str(args.checkpoint), "zip": str(zip_path),
                    "extracted_mb": extracted, "scores": scores}, indent=2))
    print(f"\nready to upload: {zip_path}")


if __name__ == "__main__":
    main()
