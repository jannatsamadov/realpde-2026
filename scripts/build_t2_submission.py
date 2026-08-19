"""Assemble and verify a Track 2 (LTTTA) submission archive.

Produces submissions/<tag>.zip containing:

    submission.py    get_ttt_model / reset_ttt_state / ttt_step
    model_def.py     the network, vendored so no project import can leak
    advection.py     the semi-Lagrangian prior it depends on
    model.pth        weights, stripped of optimizer state
    sps_sigma.npz    the per-element residual map the intervals start from

and verifies it the way the platform will, before a rationed daily submission is
spent on a packaging mistake:

  1. unzip into a clean directory,
  2. run the kit's own local_eval.py against that directory, in a subprocess
     whose working directory and sys.path exclude this project, so an accidental
     `from realpde...` fails here rather than on Codabench,
  3. reject any import of a module the evaluation image does not have,
  4. check the interface contract the kit enforces: shapes, adapt_loss, and the
     all-or-nothing bounds rule,
  5. check the extracted size against the 256 MB cap,
  6. score the real validation stream with scripts/t2_eval.py and report the
     wall clock against the 10-minute execution limit.

Usage:
    python scripts/build_t2_submission.py --tag t2_v1
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
KIT = (ROOT / "NeurIPS 2026 RealPDE Competition"
       / "realpde_t2_starting_kit_v6" / "realpde_t2_starting_kit_v6")
TEMPLATE_DIR = ROOT / "submissions" / "t2_template"
SIZE_CAP_MB = 256
EXEC_LIMIT_S = 600

VENDORED = {
    "model_def.py": ROOT / "src" / "realpde" / "models.py",
    "advection.py": ROOT / "src" / "realpde" / "advection.py",
}

# Present in the Track 2 image (PyTorch 2.2.2, CUDA 12.1, Python 3.10).
ALLOWED_TOP_LEVEL = {
    "torch", "torchvision", "torchaudio", "numpy", "PIL", "yaml", "requests",
    "tqdm", "sympy", "networkx",
    "os", "sys", "math", "json", "time", "copy", "typing", "pathlib", "__future__",
    "model_def", "advection", "ttt_model",
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


def build(checkpoint: Path, sigma: Path, tag: str,
          sigma_online: str | None = None) -> tuple[Path, Path]:
    out_dir = ROOT / "submissions" / tag
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # Variants are applied to the BUILT copy, never to the template. Editing the
    # template in place is how the Track 1 one silently kept adv_v3's losing
    # settings, so every later build inherited them.
    text = (TEMPLATE_DIR / "submission.py").read_text(encoding="utf-8")
    if sigma_online is not None:
        needle = 'SIGMA_ONLINE = "scalar"'
        if needle not in text:
            raise SystemExit(f"cannot apply --sigma-online: {needle!r} not in template")
        text = text.replace(needle, f'SIGMA_ONLINE = "{sigma_online}"')
        print(f"  variant: SIGMA_ONLINE = {sigma_online!r}")
    (out_dir / "submission.py").write_text(text, encoding="utf-8")
    # Re-vendored from src on every build. A stale copy here once made a timing
    # measurement report an optimisation that was not in the archive.
    for dest, src in VENDORED.items():
        shutil.copy(src, out_dir / dest)

    state = torch.load(checkpoint, map_location="cpu")
    train_args = state.get("args", {})
    torch.save({
        "model": state["model"],
        "config": {
            "arch": train_args.get("model", "advective"),
            "t_in": 20, "t_out": 20, "channels": 2,
            "base": train_args.get("base", 64),
        },
    }, out_dir / "model.pth")
    shutil.copy(sigma, out_dir / "sps_sigma.npz")

    zip_path = ROOT / "submissions" / f"{tag}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out_dir.iterdir()):
            if f.is_file():
                z.write(f, f.name)
    return zip_path, out_dir


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
            print(f"  WARNING: {name} imports {sorted(unknown)} — confirm these exist")
    print("  OK")


def run_kit_eval(extracted: Path) -> None:
    """The organizers' own local_eval.py, from outside the project tree."""
    print("\n[4/5] kit local_eval.py on the extracted archive")
    # Inherit the environment (torch needs PATH for its CUDA DLLs on Windows) but
    # drop PYTHONPATH, and run from the extracted directory: if submission.py
    # ever reaches back into the project's src/, it has to fail here.
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, str(KIT / "local_eval.py"), "--submission", str(extracted)],
        capture_output=True, text=True, cwd=str(extracted), env=env,
    )
    out = proc.stdout + proc.stderr
    for line in out.splitlines():
        print(f"  {line}")
    if proc.returncode != 0 or "OK: submission ran end-to-end" not in out:
        raise SystemExit("  FAIL: the kit's own evaluator rejected this archive")
    print("  OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "checkpoints" / "adv_finetuned_best.pt")
    ap.add_argument("--sigma", type=Path,
                    default=ROOT / "checkpoints" / "sps_sigma.npz")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sigma-online", choices=["off", "scalar"], default=None,
                    help="override the template's online interval calibration for "
                         "this build only. 'off' is the controlled comparison: it "
                         "changes nothing else, so the leaderboard difference is "
                         "the calibration's own contribution.")
    ap.add_argument("--skip-score", action="store_true")
    args = ap.parse_args()

    print(f"[1/5] building from {args.checkpoint.name} + {args.sigma.name}")
    zip_path, sub_dir = build(args.checkpoint, args.sigma, args.tag,
                              args.sigma_online)
    print(f"  wrote {zip_path}")

    check_imports(sub_dir)

    print("\n[3/5] size")
    extracted_mb = sum(f.stat().st_size for f in sub_dir.iterdir()
                       if f.is_file()) / 1024**2
    print(f"  archive {zip_path.stat().st_size / 1024**2:.2f} MB, "
          f"extracted {extracted_mb:.2f} MB (cap {SIZE_CAP_MB} MB)")
    if extracted_mb >= SIZE_CAP_MB:
        raise SystemExit("  FAIL: over the size cap")
    print("  OK")

    with tempfile.TemporaryDirectory() as tmp:
        clean = Path(tmp) / "extracted"
        clean.mkdir()
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(clean)
        run_kit_eval(clean)

    if args.skip_score:
        return

    print("\n[5/5] scoring the real validation stream")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "t2_eval.py"),
         "--submission", str(sub_dir),
         "--out", str(ROOT / "submissions" / f"{args.tag}_eval.json")],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    print(proc.stdout[-2500:] or proc.stderr[-2500:])
    if proc.returncode != 0:
        raise SystemExit("  FAIL: scoring run failed")

    scored = json.loads((ROOT / "submissions" / f"{args.tag}_eval.json").read_text())
    s = next(iter(scored.values()))
    wall = s.get("_total_wall_s", 0.0)
    print(f"  stream wall clock {wall:.1f} s over 546 steps "
          f"({1000 * s['_mean_t_neural_s']:.1f} ms/step); the platform's limit is "
          f"{EXEC_LIMIT_S} s for its own stream length")

    (ROOT / "submissions" / f"{args.tag}_report.json").write_text(json.dumps(
        {"checkpoint": str(args.checkpoint), "sigma": str(args.sigma),
         "zip": str(zip_path), "extracted_mb": extracted_mb, "scores": s}, indent=2))
    print(f"\nready to upload: {zip_path}")


if __name__ == "__main__":
    main()
