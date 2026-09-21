"""Regression test for the model zoo: refactors must not change what a checkpoint computes.

Four checks, each one a way this has gone wrong or nearly gone wrong:

1. REPRODUCTION. Every trained checkpoint is rebuilt from its architecture name
   -- exactly as the submission templates do -- and re-evaluated on the split it
   was validated on. The accuracy subscores must match the numbers recorded in
   its history file when it was trained. A refactor that shifts them has
   changed the model, whatever the diff looks like.

2. ZERO-INIT IDENTITY. A refining model with a zero-initialised head must equal
   its non-refining twin exactly: one_step(adv[k-1]) is adv[k]. This is what
   makes the refinement a clean experiment, and a broken prior shows up here.

3. SPLIT CONTRACT. forward_from_features(features(x)) must equal forward(x) or
   raise. The Track 2 adapter treats "did not raise" as "splittable"; a model
   that silently returns only its first pass would be deployed as a different
   function from the one it was trained as. The first vendoring test missed
   exactly this, because at a zero-initialised head the two coincide -- so this
   check runs on trained weights.

4. VENDORING. models.py and advection.py load flat, with no realpde package on
   the path, which is how they travel inside a submission archive.

Usage:
    python scripts/test_models.py
"""

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from realpde.data import (DEFAULT_VAL_RE, EXTRAPOLATION_VAL_RE, build_datasets)
from realpde.models import _ARCHS, build_model

CKPT = ROOT / "checkpoints"
TOL = 2e-3          # score points; GPU convolutions are not bit-deterministic
FAILURES = []


def check(ok: bool, msg: str) -> None:
    print(("  PASS  " if ok else "  FAIL  ") + msg)
    if not ok:
        FAILURES.append(msg)


def load_train_module():
    spec = importlib.util.spec_from_file_location("train", ROOT / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reproduction(device) -> None:
    print("\n1. trained checkpoints reproduce their recorded scores")
    train = load_train_module()
    cases = [("xf_advective", 25), ("xf_scaleinv", 25), ("xf_solverloop", 25),
             ("adv_finetuned", 30)]
    loaders = {}
    for tag, epoch in cases:
        ck_path, hist_path = CKPT / f"{tag}_best.pt", CKPT / f"{tag}_history.json"
        if not ck_path.exists() or not hist_path.exists():
            print(f"  skip  {tag}: missing")
            continue
        ck = torch.load(ck_path, map_location=device)
        a = ck["args"]
        model = build_model(a["model"], base=a["base"], channels=2).to(device).eval()
        model.load_state_dict(ck["model"])

        val_key = a.get("val_re") or "default"
        if val_key not in loaders:
            val_re = EXTRAPOLATION_VAL_RE if val_key == "extrap" else DEFAULT_VAL_RE
            _, val_ds = build_datasets("real", val_re=val_re)
            loaders[val_key] = DataLoader(val_ds, batch_size=16, shuffle=False,
                                          num_workers=0)
        got = train.evaluate(model, loaders[val_key], device)
        rec = next(r for r in json.loads(hist_path.read_text()) if r["epoch"] == epoch)
        for k in ("rel_l2_score", "tke_score", "mvpe_score", "sps_score"):
            d = abs(got[k] - rec[k])
            check(d < TOL, f"{tag:<14} [{a['model']}] {k:<13} recorded "
                           f"{rec[k]:.4f}  now {got[k]:.4f}  (diff {d:.1e})")


def zero_init_identity() -> None:
    print("\n2. a refining model at a zero-initialised head equals its twin")
    torch.manual_seed(0)
    x = torch.randn(2, 20, 32, 64, 2) * 0.6
    x[:, :, 12:18, 20:28, :] = -1.6       # a masked patch, as real data has
    for plain, loop in (("advective", "solverloop"), ("scaleinv", "scaleinv_loop")):
        a = build_model(plain, base=16, channels=2).eval()
        b = build_model(loop, base=16, channels=2).eval()
        b.load_state_dict(a.state_dict())
        with torch.no_grad():
            d = (a(x) - b(x)).abs().max().item()
        check(d < 1e-5, f"{loop:<14} == {plain:<10} at init   max diff {d:.1e}")


def split_contract(device) -> None:
    print("\n3. forward_from_features(features(x)) equals forward(x) or raises")
    torch.manual_seed(1)
    x = (torch.randn(2, 20, 32, 64, 2) * 0.6).to(device)
    for name in _ARCHS:
        m = build_model(name, base=16, channels=2).to(device).eval()
        # Trained-looking weights: at a zero head a refining model's two paths
        # coincide and the check would pass for the wrong reason.
        torch.nn.init.normal_(m.head.weight, std=0.05)
        torch.nn.init.normal_(m.head.bias, std=0.05)
        with torch.no_grad():
            full = m(x)
            try:
                split = m.forward_from_features(*m.features(x))
            except RuntimeError:
                check(True, f"{name:<14} raises -> adapter runs forward() whole")
                continue
            d = (full - split).abs().max().item()
        check(d < 1e-5, f"{name:<14} split equals forward   max diff {d:.1e}")


def vendoring() -> None:
    print("\n4. vendored flat, with no realpde package importable")
    tmp = Path(tempfile.mkdtemp())
    try:
        shutil.copy(ROOT / "src" / "realpde" / "models.py", tmp / "model_def.py")
        shutil.copy(ROOT / "src" / "realpde" / "advection.py", tmp / "advection.py")
        code = (
            "import sys, torch; sys.path.insert(0, r'%s')\n"
            "from model_def import build_model, _ARCHS\n"
            "x = torch.randn(1, 20, 32, 64, 2)\n"
            "for n in _ARCHS:\n"
            "    y = build_model(n, base=16, channels=2).eval()(x)\n"
            "    assert y.shape == (1, 20, 32, 64, 2), n\n"
            "print('ok', len(_ARCHS))\n" % tmp)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, cwd=tmp)
        check(r.returncode == 0 and r.stdout.startswith("ok"),
              f"all {len(_ARCHS)} architectures build and run from the flat copy"
              + ("" if r.returncode == 0 else f"\n{r.stderr[-600:]}"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    zero_init_identity()
    split_contract(device)
    vendoring()
    reproduction(device)
    print(f"\n{'ALL PASSED' if not FAILURES else f'{len(FAILURES)} FAILED'}")
    raise SystemExit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
