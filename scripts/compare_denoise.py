"""Paired comparison: strain-scaled noise augmentation against the plain run.

Both pipelines are identical except for `--noise-aug 0.4` during the simulated
pretraining stage, so the two histories are paired epoch by epoch on the same
validation split with the same seed. That makes the per-epoch difference the
measurement of interest — far more informative than either curve alone, because
the split's own internal spread (3-4 points) dwarfs the effect being tested.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints"
FIG = ROOT / "figures"

PAIRS = [
    ("sim", "sim_pretrain", "sim_denoise", "Sim pretrain (15 epoxa)"),
    ("real", "adv_finetuned", "adv_denoise", "Real finetune (30 epoxa)"),
]
METRICS = [("_selection", "seçim"), ("tke_score", "tke"), ("rel_l2_score", "rel_l2")]


def load(tag: str) -> list[dict]:
    path = CKPT / f"{tag}_history.json"
    if not path.exists():
        raise FileNotFoundError(f"history missing: {path}")
    return json.loads(path.read_text())


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))

    for row, (_, base_tag, test_tag, title) in enumerate(PAIRS):
        base, test = load(base_tag), load(test_tag)
        ep = [r["epoch"] for r in base]
        assert [r["epoch"] for r in test] == ep, "histories are not paired"

        for col, (key, label) in enumerate(METRICS):
            ax = axes[row, col]
            b = np.array([r[key] for r in base])
            t = np.array([r[key] for r in test])

            ax.plot(ep, b, "o-", color="#4a5568", label=f"şumsuz ({base_tag})")
            ax.plot(ep, t, "s-", color="#c05621", label=f"şumlu ({test_tag})")
            ax.set_xlabel("epoxa")
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

            d = t - b
            ax.set_title(f"{label}   Δ orta {d.mean():+.3f}", fontsize=10)

            # The difference is the point, so annotate it where it is readable.
            for x, y, dd in zip(ep, t, d):
                ax.annotate(f"{dd:+.2f}", (x, y), textcoords="offset points",
                            xytext=(0, -14), ha="center", fontsize=7,
                            color="#c05621")

        axes[row, 0].text(-0.28, 0.5, title, transform=axes[row, 0].transAxes,
                          rotation=90, va="center", ha="center", fontsize=11)

    fig.suptitle("Deformasiyaya görə şum augmentasiyası — cütləşdirilmiş müqayisə\n"
                 "(Δ = şumlu − şumsuz; val split-inin daxili yayılması 3-4 xaldır)",
                 fontsize=12)
    fig.tight_layout(rect=(0.02, 0, 1, 0.94))
    out = FIG / "denoise_comparison.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")

    # The verdict in numbers, so it lands in the notes without being re-derived.
    print(f"\n{'mərhələ':<16} {'metrik':<10} {'şumsuz':>9} {'şumlu':>9} {'Δ':>8}")
    print("-" * 56)
    for _, base_tag, test_tag, title in PAIRS:
        base, test = load(base_tag), load(test_tag)
        for key, label in METRICS:
            b = np.array([r[key] for r in base])
            t = np.array([r[key] for r in test])
            print(f"{title.split('(')[0].strip():<16} {label:<10} "
                  f"{b[-1]:>9.3f} {t[-1]:>9.3f} {t[-1] - b[-1]:>+8.3f}")


if __name__ == "__main__":
    main()
