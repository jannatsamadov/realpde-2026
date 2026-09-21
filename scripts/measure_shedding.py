"""Does the vortex-shedding period change with Reynolds number, in FRAMES?

WHY THIS DECIDES THE SCALE-CONDITIONING DESIGN
    Non-dimensionalising the velocity (u -> u/U_inf) removes the amplitude
    difference between Reynolds numbers. It does not remove a *temporal* one.
    Navier-Stokes is invariant under u -> u/U, x -> x/c, t -> t*U/c: the last
    factor matters because our frame interval dt is fixed in seconds, so the
    non-dimensional step dt*U_inf/c grows linearly with U_inf, hence with Re.

    Concretely: if the Strouhal number St = f*c/U_inf is constant, then the
    shedding frequency in Hz is proportional to U_inf, and the period measured
    in FRAMES shrinks as 1/Re. A network fed u/U_inf would then still see a
    20-frame window covering a different number of shedding cycles at every Re
    -- the amplitude is normalised but the dynamics are not.

    If instead the period in frames is roughly constant across Re, velocity
    rescaling alone is a complete non-dimensionalisation for our purposes and
    no temporal resampling is needed.

METHOD
    Per trajectory, over its full length: remove the temporal mean per cell,
    take the real FFT of the transverse velocity v along time, sum power over
    all cells, and report the peak frequency (f = 0 excluded). v is used rather
    than u because shedding is a transverse oscillation and v has no large mean
    to leak into the low bins.

    The peak is reported in cycles/frame, which is the quantity the network
    actually experiences, and converted to Hz with the dataset's dt.

Usage:
    python scripts/measure_shedding.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from realpde.data import U_PER_RE, load_split

ROOT = Path(__file__).resolve().parent.parent
DT = 0.02  # s per frame; checked against the recorded 0.325 s = 16.25 frames


def dominant_frequency(v: np.ndarray) -> tuple[float, float]:
    """Peak temporal frequency of a (T, H, W) field, in cycles/frame.

    Returns (frequency, sharpness) where sharpness is the peak's share of the
    total fluctuation power -- a flat spectrum means there is no clean shedding
    and the frequency should not be trusted.
    """
    T = v.shape[0]
    f = v - v.mean(axis=0, keepdims=True)
    spec = np.abs(np.fft.rfft(f, axis=0)) ** 2          # (T//2+1, H, W)
    power = spec.reshape(spec.shape[0], -1).sum(axis=1)  # over all cells
    power[0] = 0.0                                       # drop the mean
    freqs = np.fft.rfftfreq(T, d=1.0)                    # cycles per frame
    k = int(np.argmax(power))
    # Sharpness: the peak bin plus its two neighbours, over the whole spectrum.
    lo, hi = max(k - 1, 0), min(k + 2, len(power))
    sharp = float(power[lo:hi].sum() / max(power.sum(), 1e-30))
    return float(freqs[k]), sharp


def main() -> None:
    arr, meta = load_split("real")
    chans = meta["channels"]
    vi = chans.index("v") if "v" in chans else 1
    print(f"channels {chans}, using index {vi} for v")
    print(f"{len(meta['trajectories'])} real trajectories\n")

    rows = []
    for t in meta["trajectories"]:
        o, L = t["offset"], t["length"]
        v = np.asarray(arr[o:o + L, ..., vi], dtype=np.float64)
        f_frame, sharp = dominant_frequency(v)
        re = float(t["re_nominal"])
        aoa = float(t.get("aoa_nominal", t.get("aoa", np.nan)))
        rows.append((t["case"], re, aoa, f_frame, sharp, L))

    rows.sort(key=lambda r: (r[2], r[1]))

    print(f"{'case':>12} {'Re':>7} {'AoA':>5} {'f[cyc/frame]':>13} "
          f"{'period[fr]':>11} {'f[Hz]':>7} {'St':>6} {'sharp':>6}")
    keep = []
    for case, re, aoa, f_frame, sharp, L in rows:
        period = np.inf if f_frame == 0 else 1.0 / f_frame
        f_hz = f_frame / DT
        u_inf = re * U_PER_RE
        chord = 0.072
        st = f_hz * chord / u_inf if u_inf > 0 else np.nan
        print(f"{case:>12} {re:>7.0f} {aoa:>5.1f} {f_frame:>13.5f} "
              f"{period:>11.2f} {f_hz:>7.3f} {st:>6.3f} {sharp:>6.3f}")
        if sharp > 0.05 and f_frame > 0:
            keep.append((re, aoa, f_frame, st))

    if not keep:
        print("\nno trajectory had a sharp enough peak to trust")
        return

    keep = np.array(keep)
    re, aoa, f_frame, st = keep[:, 0], keep[:, 1], keep[:, 2], keep[:, 3]
    print(f"\n{len(keep)}/{len(rows)} trajectories with a usable peak "
          f"(>5% of power in the peak bin +/- 1)")

    # The question, stated three ways.
    r_f = np.corrcoef(re, f_frame)[0, 1]
    print(f"\ncorr(Re, f_frame)          = {r_f:+.4f}   "
          f"(Strouhal scaling predicts +1)")
    # Log-log slope: f_frame ~ Re**p. Strouhal predicts p = 1, no change p = 0.
    p = np.polyfit(np.log(re), np.log(f_frame), 1)[0]
    print(f"log-log slope d ln f/d ln Re = {p:+.4f}   "
          f"(1 = pure Strouhal, 0 = period fixed in frames)")
    print(f"period in frames: min {1/f_frame.max():.2f}  "
          f"max {1/f_frame.min():.2f}  "
          f"ratio {f_frame.max()/f_frame.min():.2f}x")
    print(f"St = f*c/U_inf: mean {st.mean():.3f} sd {st.std():.3f}"
          f"   (bluff-body classic ~0.2)")

    # How many shedding cycles does a 20-frame window actually contain?
    cyc = 20.0 * f_frame
    print(f"\ncycles per 20-frame window: min {cyc.min():.2f}  "
          f"max {cyc.max():.2f}  mean {cyc.mean():.2f}")

    # Same, split by angle of attack, since AoA changes the effective bluff width.
    print(f"\n{'AoA':>5} {'n':>3} {'mean f[cyc/fr]':>15} {'mean period[fr]':>16} "
          f"{'slope vs Re':>12}")
    for a in np.unique(aoa):
        m = aoa == a
        if m.sum() < 3:
            print(f"{a:>5.1f} {m.sum():>3}  (too few for a slope)")
            continue
        s = np.polyfit(np.log(re[m]), np.log(f_frame[m]), 1)[0]
        print(f"{a:>5.1f} {m.sum():>3} {f_frame[m].mean():>15.5f} "
              f"{1/f_frame[m].mean():>16.2f} {s:>12.3f}")


if __name__ == "__main__":
    main()
