"""What subscores would a target final_score require?

There is a structural constraint worth seeing before choosing what to work on.
scoring.py builds SPS as

    sps = 100 * acc * Q,  Q = E[exp(-nil) * inside] <= 1
    acc = 0.5*(1-pm_dm) + 0.3*(1-pm_tke) + 0.2*(1-pm_mvpe),  pm = err/(0.5+err)

Q cannot exceed 1, so **sps <= 100 * acc**: the interval score is capped by the
prediction's own accuracy, and no amount of bound tuning gets past that. At our
current errors acc = 0.690, so sps can never exceed 69 — we scored 36.17, and the
ceiling itself has to be raised by predicting better.

That coupling means tke is worth more than its own subscore: it enters acc with
weight 0.3, so improving it lifts the sps ceiling too.

final_score weights are the two-submission fit from infer_hidden_sigma.py, which
is one consistent solution rather than the published rule. Treat the ranking as
sound and the absolute level as approximate.

Usage:
    python scripts/what_would_it_take.py
"""

W_SPS, W_OTHER = 0.301, 0.194

NOW = dict(rel_l2=94.55, tke=74.63, mvpe=93.49, time=87.15, sps=36.17)
ACHIEVED_Q = 0.524   # what our current bands achieve on the hidden set


def err(score):
    return 2.0 * (100.0 / score - 1.0)


def pm(e):
    return e / (0.5 + e)


def acc_of(rel_l2, tke, mvpe):
    return (0.5 * (1 - pm(err(rel_l2))) + 0.3 * (1 - pm(err(tke)))
            + 0.2 * (1 - pm(err(mvpe))))


def final_of(rel_l2, tke, mvpe, time, sps):
    return W_OTHER * (rel_l2 + tke + mvpe + time) + W_SPS * sps


def row(name, rel_l2, tke, mvpe, time, q):
    a = acc_of(rel_l2, tke, mvpe)
    ceiling = 100 * a
    sps = ceiling * q
    f = final_of(rel_l2, tke, mvpe, time, sps)
    print(f"  {name:<30}{rel_l2:>8.1f}{tke:>7.1f}{mvpe:>7.1f}{time:>7.1f}"
          f"{ceiling:>9.1f}{sps:>7.1f}{f:>9.2f}")
    return f


def main() -> None:
    print("THE SPS CEILING IS SET BY ACCURACY\n")
    a = acc_of(NOW["rel_l2"], NOW["tke"], NOW["mvpe"])
    print(f"  current accuracy factor      {a:.4f}")
    print(f"  therefore sps cannot exceed  {100 * a:.1f}")
    print(f"  we scored                    {NOW['sps']:.2f}  "
          f"(Q = {NOW['sps'] / (100 * a):.3f} of the ceiling)")
    print(f"  perfect bands alone would give final "
          f"{final_of(NOW['rel_l2'], NOW['tke'], NOW['mvpe'], NOW['time'], 100 * a):.2f}")
    print("  -- so bound tuning alone cannot reach 90, whatever we do to it.\n")

    hdr = (f"  {'scenario':<30}{'rel_l2':>8}{'tke':>7}{'mvpe':>7}{'time':>7}"
           f"{'sps cap':>9}{'sps':>7}{'final':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    row("now", NOW["rel_l2"], NOW["tke"], NOW["mvpe"], NOW["time"], ACHIEVED_Q)
    row("+ retuned bands", NOW["rel_l2"], NOW["tke"], NOW["mvpe"], NOW["time"], 0.552)
    row("+ alpha 1.15", NOW["rel_l2"], 75.8, NOW["mvpe"], NOW["time"], 0.552)
    row("time back to 92 (drop advect)", NOW["rel_l2"], 75.8, NOW["mvpe"], 92.0, 0.552)
    print()
    row("tke 80", NOW["rel_l2"], 80.0, NOW["mvpe"], NOW["time"], 0.552)
    row("tke 85", NOW["rel_l2"], 85.0, NOW["mvpe"], NOW["time"], 0.552)
    row("tke 90", NOW["rel_l2"], 90.0, NOW["mvpe"], NOW["time"], 0.552)
    print()
    row("everything +2, tke 85", 96.5, 85.0, 95.5, 92.0, 0.552)
    row("everything +2, tke 90, Q .65", 96.5, 90.0, 95.5, 92.0, 0.65)
    row("strong: 97/92/97/93, Q .70", 97.0, 92.0, 97.0, 93.0, 0.70)
    row("near-perfect accuracy, Q .75", 98.5, 95.0, 98.5, 95.0, 0.75)

    print("\nREADING THIS")
    print("  Reaching 90 needs tke somewhere near 90 — it is currently 74.6, and it")
    print("  is the metric that punishes an MSE-trained model hardest. Every other")
    print("  subscore is already above 87 and has only a few points left in it.")
    print("  tke pays twice: once directly, once by raising the sps ceiling.")
    print("\n  Post-hoc corrections cannot do this. Alpha bought 1.2 points of tke")
    print("  before the spatial term in the metric turned against it. Getting to 90")
    print("  means predicting the fluctuation in the right place, not just at the")
    print("  right amplitude — which is a different model, not a tweak.")


if __name__ == "__main__":
    main()
