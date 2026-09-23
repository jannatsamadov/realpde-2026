"""Does capturing the forward pass in a CUDA graph cut the Track 2 step time?

WHY THIS IS THE LEVER
    On the leaderboard the solver-loop model cost 3.45 time_score in Track 2 and
    only 0.15 in Track 1, for the same 15% of extra arithmetic. Backing the times
    out of the score formula: 13.4 ms -> 24.2 ms per step at batch 1, a factor of
    1.8, against 1.17 measured here. At batch one on a fast GPU the step is bound
    by kernel launches, not by arithmetic, and the refinement pass roughly doubles
    the kernel count. A CUDA graph replays the whole pass as a single launch, so
    it attacks exactly that cost.

    With the refit weights (time 0.098), those 3.45 points are -0.34 final: the
    accuracy gain (+0.40 sps = +0.11) was real and the time loss buried it.

WHAT IS MEASURED
    Eager against graph replay, at batch 1, alternating between the two so this
    card's 300-2100 MHz clock swing cannot rank them. Both architectures: the
    single-pass model behind t2_v1, and the solver-loop model.

    Correctness first: a graph that produces different numbers is not a speedup.
    Replay writes into a fixed output tensor, so each replay is compared against
    eager on the same input before any timing.

Usage:
    python scripts/bench_cuda_graph.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch

from realpde.data import DEFAULT_VAL_RE, WindowDataset, split_cases
from realpde.models import build_model

ARCHS = [("advective", "adv_finetuned_best.pt"), ("solverloop", "sl_default_best.pt")]


def capture(model, static_in):
    """Warm up on a side stream, then capture one forward pass."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        for _ in range(3):
            model(static_in)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g):
        static_out = model(static_in)
    return g, static_out


def bench(fn, iters=40):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    dev = torch.device("cuda")

    _, val_cases = split_cases("real", DEFAULT_VAL_RE)
    ds = WindowDataset("real", cases=val_cases, stride=40)
    windows = [ds[i]["input"][..., :2].unsqueeze(0).to(dev) for i in range(0, 24, 4)]

    for arch, ckpt in ARCHS:
        ck = torch.load(ROOT / "checkpoints" / ckpt, map_location=dev)
        model = build_model(arch, base=ck["args"]["base"], channels=2).to(dev).eval()
        model.load_state_dict(ck["model"])

        # Batch 1 is Track 2's scored step; 64 is Track 1's chunk.
        for batch in (1, 64):
            static_in = torch.zeros(batch, *windows[0].shape[1:], device=dev)
            try:
                g, static_out = capture(model, static_in)
            except Exception as e:                   # capture must never be fatal
                print(f"{arch:<11} b{batch:<3} capture FAILED: "
                      f"{type(e).__name__}: {e}")
                continue

            # Correctness: replay on real windows, compare against eager.
            worst = 0.0
            with torch.no_grad():
                for w in windows:
                    wb = w.expand(batch, -1, -1, -1, -1).contiguous()
                    static_in.copy_(wb)
                    g.replay()
                    worst = max(worst, float((static_out - model(wb)).abs().max()))
            ok = worst < 1e-4

            def eager():
                with torch.no_grad():
                    model(static_in)

            def graphed():
                g.replay()

            te = tg = 0.0
            for _ in range(3):                       # alternate, clock-proof
                te += bench(eager, iters=20 if batch > 1 else 40)
                tg += bench(graphed, iters=20 if batch > 1 else 40)
            te, tg = te / 3 / batch, tg / 3 / batch  # per sample

            print(f"{arch:<11} batch {batch:<3} eager {te:6.2f} ms/sample   "
                  f"graph {tg:6.2f} ms/sample   speedup {te / tg:4.2f}x   "
                  f"max|diff| {worst:.2e} {'OK' if ok else 'MISMATCH'}")

            # What the platform would score if its time scaled by the same ratio.
            ref = ((("t2_v1", 13.4), ("t2_solverloop", 24.2)) if batch == 1
                   else (("adv_v2 T1", 15.84), ("t1_solverloop", 16.26)))
            f = lambda t: 100.0 / (1.0 + (t / 1000.0 / 0.72896) ** 0.5)
            for name, t_plat in ref:
                t_new = t_plat * (tg / te)
                print(f"               {name:<15} time_score {f(t_plat):.2f} -> "
                      f"{f(t_new):.2f}  ({0.098 * (f(t_new) - f(t_plat)):+.2f} final)")


if __name__ == "__main__":
    main()
