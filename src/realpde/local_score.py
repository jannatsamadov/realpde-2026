"""Score predictions locally with the organizers' own scoring.py.

The starting kit's scoring.py is imported and called directly rather than
reimplemented. A reimplementation would drift from the leaderboard the moment the
organizers change anything, and they have already changed the SPS mapping once
this season. Importing it means our local numbers are the leaderboard's numbers
by construction.

Only `final_score` is missing, because its combination rule is unpublished.

Usage as a library:
    from realpde.local_score import score_arrays
    scores = score_arrays(pred, target, mean_t_neural_s=0.004)
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
KIT = (ROOT / "NeurIPS 2026 RealPDE Competition"
       / "realpde_t1_starting_kit_v9" / "realpde_t1_starting_kit_v9")


def load_scoring_module():
    """Import the kit's scoring.py from its path, without installing it."""
    path = KIT / "scoring.py"
    if not path.exists():
        raise FileNotFoundError(
            f"Track 1 scoring.py not found at {path}. The starting kit must be "
            "unpacked there for local scores to match the leaderboard."
        )
    spec = importlib.util.spec_from_file_location("rpde_scoring", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rpde_scoring"] = module
    spec.loader.exec_module(module)
    return module


SCORING = load_scoring_module()


def to_five_channel(a: np.ndarray) -> np.ndarray:
    """Pad a 2-channel (u, v) array out to the 3 channels the scorer expects."""
    a = np.asarray(a, dtype=np.float32)
    if a.shape[-1] == 3:
        return a
    pad = np.zeros(a.shape[:-1] + (3 - a.shape[-1],), dtype=np.float32)
    return np.concatenate([a, pad], axis=-1)


def score_arrays(
    pred: np.ndarray,
    target: np.ndarray,
    mean_t_neural_s: float,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> dict:
    """Return the five subscores for (N, T, H, W, C) prediction/target arrays.

    `mean_t_neural_s` is the per-sample inference wall time. On the leaderboard
    the ingestion program measures it; locally we must supply it, and it drives
    time_score entirely.
    """
    pred = to_five_channel(pred)
    target = to_five_channel(target)
    SCORING.validate_shapes(pred, target)

    if not np.all(np.isfinite(pred)):
        return dict(SCORING.ZERO_SCORES) | {"error": "non-finite prediction"}

    c = SCORING.measured_channels(target)
    rel_l2 = float(np.mean(SCORING.rel_l2_per_sample(pred, target, c)))
    tke = float(np.mean(SCORING.tke_rel_l2_per_sample(pred, target, c)))
    mvpe = SCORING.mvpe_rel_l2(pred, target)

    if lower is not None:
        lower = to_five_channel(lower)
    if upper is not None:
        upper = to_five_channel(upper)
    sps, coverage = SCORING.aggregate_sps(pred, target, c, lower=lower, upper=upper)

    return {
        "rel_l2_score": SCORING.score_error(rel_l2),
        "tke_score": SCORING.score_error(tke),
        "mvpe_score": SCORING.score_error(mvpe),
        "time_score": SCORING.score_time(mean_t_neural_s),
        "sps_score": SCORING.score_sps(sps),
        # Raw errors are not on the leaderboard but are what we actually optimise.
        "_rel_l2_error": rel_l2,
        "_tke_error": tke,
        "_mvpe_error": mvpe,
        "_sps_coverage": coverage,
        "_measured_channels": c,
        "_mean_t_neural_s": mean_t_neural_s,
    }


def score_via_files(pred, target, mean_t_neural_s, lower=None, upper=None) -> dict:
    """Run the scorer end to end through npz files, exactly as Codabench does.

    Slower than score_arrays, but it exercises the same path the platform uses —
    worth running once before a real submission to catch packaging mistakes.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "ref").mkdir()
        (tmp / "out").mkdir()
        bundle = {"prediction": to_five_channel(pred),
                  "mean_t_neural_s": np.array([mean_t_neural_s], dtype=np.float32)}
        if lower is not None and upper is not None:
            bundle["lower"] = to_five_channel(lower)
            bundle["upper"] = to_five_channel(upper)
        np.savez(tmp / "predictions.npz", **bundle)
        np.savez(tmp / "ref" / "targets.npz", target=to_five_channel(target))

        argv = sys.argv
        sys.argv = ["scoring.py", str(tmp), str(tmp / "out")]
        try:
            SCORING.main()
        finally:
            sys.argv = argv

        import json
        return json.loads((tmp / "out" / "scores.json").read_text())


def format_scores(scores: dict) -> str:
    main = ["rel_l2_score", "tke_score", "mvpe_score", "time_score", "sps_score"]
    lines = ["  " + "-" * 46]
    for k in main:
        if k in scores:
            lines.append(f"  {k:<16} {scores[k]:>8.3f}")
    lines.append("  " + "-" * 46)
    for k, v in scores.items():
        if k.startswith("_"):
            lines.append(f"  {k:<16} {v:>8.5f}" if isinstance(v, float) else f"  {k:<16} {v:>8}")
    if "error" in scores:
        lines.append(f"  ERROR: {scores['error']}")
    return "\n".join(lines)
