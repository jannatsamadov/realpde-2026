"""Trace a trained checkpoint stage by stage: what the built model actually does.

Not a description of the design -- a measurement of the object. The checkpoint is
rebuilt from its architecture name exactly as the submission templates rebuild
it, real validation windows are pushed through it, and every stage is called
through the model's own methods. For each stage this records the tensor shape,
the parameters it owns, and -- where the stage produces a forecast -- that
forecast's error against the truth.

The final output is recomputed through the stages and checked against a plain
model(x) call, so the trace cannot silently describe a different computation
from the one that is scored.

Usage:
    python scripts/trace_model.py --checkpoint sl_default_best.pt
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from realpde.data import (DEFAULT_VAL_RE, EXTRAPOLATION_VAL_RE, WindowDataset,
                          denormalize, split_cases)
from realpde.models import build_model, fold_time


def n_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="write the trace as JSON")
    args = ap.parse_args()

    dev = torch.device(args.device)
    ck = torch.load(ROOT / "checkpoints" / args.checkpoint, map_location=dev)
    a = ck["args"]
    model = build_model(a["model"], base=a["base"], channels=2).to(dev).eval()
    model.load_state_dict(ck["model"])

    val_re = EXTRAPOLATION_VAL_RE if a.get("val_re") == "extrap" else DEFAULT_VAL_RE
    _, val_cases = split_cases("real", val_re)
    ds = WindowDataset("real", cases=val_cases, stride=40)
    x = torch.stack([ds[i]["input"] for i in range(len(ds))])[..., :2].to(dev)
    y = torch.stack([ds[i]["target"] for i in range(len(ds))])[..., :2].to(dev)
    truth = denormalize(y.cpu()).numpy()

    def err(pred_norm) -> dict:
        """RMS error in m/s, and the scorer's relative L2, of a normalized forecast."""
        p = denormalize(pred_norm.detach().cpu()).numpy()
        rms = float(np.sqrt(((p - truth) ** 2).mean()))
        rel = float(np.mean([np.linalg.norm(p[i] - truth[i]) / np.linalg.norm(truth[i])
                             for i in range(len(p))]))
        return {"rms_ms": rms, "rel_l2": rel}

    stages = []

    def add(name, **kw):
        stages.append({"stage": name, **kw})
        shape = kw.get("shape")
        e = kw.get("error")
        print(f"{name:<34} {str(shape):<24} "
              f"{kw.get('params', ''):>9} "
              + (f" rms {e['rms_ms']:.5f} m/s  rel_l2 {e['rel_l2']:.4f}" if e else ""))

    print(f"checkpoint {args.checkpoint}   arch {a['model']}   "
          f"refine {getattr(model, 'refine', 0)}   total params {n_params(model):,}")
    print(f"{len(x)} validation windows, val Re {val_re}\n")
    print(f"{'stage':<34} {'shape':<24} {'params':>9}  error vs truth")

    with torch.no_grad():
        add("input window", shape=list(x.shape[1:]))
        add("persistence (copy last frame)",
            shape=list(y.shape[1:]),
            error=err(x[:, -1:].expand_as(y)))

        # 1. The physics prior: semi-Lagrangian self-advection of the last frame.
        adv = model._advected(x)
        add("advection prior (20 steps)", shape=list(adv.shape[1:]), params=0,
            error=err(adv))

        # 2. Input channels, and what they are made of.
        feats, adv_f = model.features(x)
        assert torch.equal(adv, adv_f)
        c_x = fold_time(x).shape[1]
        c_adv = fold_time(adv).shape[1]
        breakdown = {"input frames": c_x, "advected frames": c_adv,
                     "speed, vorticity, divergence": 3, "x, y coordinates": 2}
        if feats.shape[1] > c_x + c_adv + 5:
            breakdown["log velocity scale"] = feats.shape[1] - (c_x + c_adv + 5)
        assert sum(breakdown.values()) == feats.shape[1] == model.enc1.net[0].in_channels
        add("network input channels", shape=list(feats.shape[1:]), params=0,
            channels=breakdown)

        # 3. The U-Net, level by level, with its own parameter counts.
        e1 = model.enc1(feats);                      add("enc1", shape=list(e1.shape[1:]), params=n_params(model.enc1))
        e2 = model.enc2(F.avg_pool2d(e1, 2));        add("enc2", shape=list(e2.shape[1:]), params=n_params(model.enc2))
        e3 = model.enc3(F.avg_pool2d(e2, 2));        add("enc3", shape=list(e3.shape[1:]), params=n_params(model.enc3))
        bo = model.bottleneck(F.avg_pool2d(e3, 2));  add("bottleneck", shape=list(bo.shape[1:]), params=n_params(model.bottleneck))
        if hasattr(model, "cond"):
            add("scale conditioning (FiLM)", shape=list(bo.shape[1:]), params=n_params(model.cond))
        d3 = model.dec3(torch.cat([F.interpolate(bo, scale_factor=2, mode="nearest"), e3], 1)); add("dec3 (+ skip enc3)", shape=list(d3.shape[1:]), params=n_params(model.dec3))
        d2 = model.dec2(torch.cat([F.interpolate(d3, scale_factor=2, mode="nearest"), e2], 1)); add("dec2 (+ skip enc2)", shape=list(d2.shape[1:]), params=n_params(model.dec2))
        d1 = model.dec1(torch.cat([F.interpolate(d2, scale_factor=2, mode="nearest"), e1], 1)); add("dec1 (+ skip enc1)", shape=list(d1.shape[1:]), params=n_params(model.dec1))
        add("head (1x1 conv) -> residual", shape=[model.head.out_channels, *d1.shape[2:]], params=n_params(model.head))

        # 4. First pass = prior + residual, through the model's own method.
        pass1 = model._single_pass(feats, adv)
        add("PASS 1 = prior + residual", shape=list(pass1.shape[1:]), params=0, error=err(pass1))

        final = pass1
        for k in range(getattr(model, "refine", 0)):
            adv2 = model._prior_from_forecast(x, final)
            add(f"refined prior {k + 1} (1-step each)", shape=list(adv2.shape[1:]), params=0, error=err(adv2))
            feats2 = model._repack(x, adv2, feats)
            final = model._single_pass(feats2, adv2)
            add(f"PASS {k + 2} = refined prior + residual", shape=list(final.shape[1:]),
                params=0, error=err(final))

        # The trace must describe the scored computation, not a lookalike.
        scored = model(x)
        same = float((scored - final).abs().max())
        print(f"\ntrace final == model(x): max diff {same:.2e}")
        assert same < 1e-5, "the traced stages do not reproduce model(x)"

    out = {"checkpoint": args.checkpoint, "arch": a["model"],
           "refine": getattr(model, "refine", 0), "params": n_params(model),
           "substeps": getattr(model, "substeps", None),
           "windows": len(x), "val_re": list(val_re), "stages": stages}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
