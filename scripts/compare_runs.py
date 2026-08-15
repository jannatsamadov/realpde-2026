"""Line up every trained run's best evaluation side by side.

Reads the history files written by train.py, so it costs nothing and can be run
while another job holds the GPU. Rows are ordered by the selection score, which
is the mean of rel_l2, tke and mvpe — final_score's own combination rule is not
published, so this is a proxy, and time_score is shown but not selected on.

Usage:
    python scripts/compare_runs.py
    python scripts/compare_runs.py --baseline advective
"""

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"

# Measured on the same validation split, for context.
REFERENCE = {
    "persistence (no model)": dict(rel_l2_score=93.50, tke_score=66.67,
                                   mvpe_score=93.39, sps_score=17.01,
                                   time_score=92.35, _selection=84.520),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="advective",
                    help="tag whose scores the delta column is measured against")
    args = ap.parse_args()

    rows = []
    for hist in sorted(CKPT.glob("*_history.json")):
        tag = hist.stem.replace("_history", "")
        entries = json.loads(hist.read_text())
        if not entries:
            continue
        best = max(entries, key=lambda r: r["_selection"])
        ckpt = CKPT / f"{tag}_best.pt"
        meta = {}
        if ckpt.exists():
            state = torch.load(ckpt, map_location="cpu")
            a = state.get("args", {})
            meta = {"arch": a.get("model", "unet"), "base": a.get("base", 64),
                    "split": a.get("split", "real"),
                    "init": Path(a["init_from"]).stem if a.get("init_from") else "-"}
        rows.append((tag, meta, best))

    # Runs trained on the simulated split are scored on simulated validation
    # windows, so their numbers are not comparable with the rest.
    real = [r for r in rows if r[1].get("split", "real") == "real"]
    sim = [r for r in rows if r[1].get("split") == "sim"]
    real.sort(key=lambda r: -r[2]["_selection"])

    def line(run, arch, base, init, ep, s, delta=""):
        return (f"{run:<20} {arch:<10} {str(base):>4}  {init:<18} {str(ep):>3}"
                f"{s['rel_l2_score']:>9.2f}{s['tke_score']:>8.2f}{s['mvpe_score']:>8.2f}"
                f"{s['sps_score']:>7.2f}{s['time_score']:>8.2f}{s['_selection']:>9.3f}"
                f"{delta:>9}")

    hdr = (f"{'run':<20} {'arch':<10} {'base':>4}  {'init from':<18} {'ep':>3}"
           f"{'rel_l2':>9}{'tke':>8}{'mvpe':>8}{'sps':>7}{'time':>8}{'sel':>9}{'vs base':>9}")
    print("SCORED ON THE REAL VALIDATION SPLIT")
    print(hdr)
    print("-" * len(hdr))

    base_sel = next((b["_selection"] for t, _, b in real if t == args.baseline), None)
    for name, ref in REFERENCE.items():
        d = f"{ref['_selection'] - base_sel:+.3f}" if base_sel else ""
        print(line(name, "-", "-", "-", "-", ref, d))

    for tag, m, b in real:
        d = f"{b['_selection'] - base_sel:+.3f}" if base_sel else ""
        star = " *" if tag == args.baseline else ""
        print(line(tag + star, m.get("arch", "?"), m.get("base", "?"),
                   m.get("init", "-"), b["epoch"], b, d))

    if sim:
        print("\nSCORED ON THE SIMULATED SPLIT — not comparable with the table above")
        print(hdr)
        print("-" * len(hdr))
        for tag, m, b in sim:
            print(line(tag, m.get("arch", "?"), m.get("base", "?"), "-", b["epoch"], b))

    print(f"\n* baseline for the delta column. Our validation split varies by about")
    print("  3-4 points across its own Reynolds and angle-of-attack groups, so a")
    print("  difference smaller than that is not yet evidence of anything.")


if __name__ == "__main__":
    main()
