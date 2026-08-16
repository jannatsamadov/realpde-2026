"""Two diagnostics for the physical structure of the wake.

Both come from ideas about the flow's mechanics, and both have standard names and
standard measures. Nothing here changes the model — it is analysis.

1. COHERENT VORTICES VERSUS STRAINED REGIONS  (the Q-criterion)

   A velocity gradient tensor splits into a rotation part and a stretching part:

       S = (grad(u) + grad(u)^T) / 2     strain rate — pulls fluid elements apart
       W = (grad(u) - grad(u)^T) / 2     rotation rate — spins them

       Q = 0.5 * (||W||^2 - ||S||^2)

   Q > 0 means rotation dominates: a coherent vortex whose fluid moves together,
   which survives and is carried downstream as a unit. Q < 0 means strain
   dominates: the structure is being pulled apart, neighbouring fluid moves in
   diverging directions, and organised motion is being handed to smaller scales.

   That is exactly the distinction between "the particles in this eddy move
   together so it stays a vortex" and "their velocities differ sharply so they
   scatter". Vorticity alone cannot make it: a plain shear layer has large
   vorticity but no coherent core, because rotation and strain cancel.

2. HOW THE STRUCTURE SIZE GROWS DOWNSTREAM  (vortex pairing and wake spreading)

   The shear layer rolls up at a scale set by its own thickness, so structures
   start small. Moving downstream they merge in pairs, and each merger roughly
   doubles the streamwise wavelength, while the wake itself widens. Both push the
   dominant scale up with distance from the airfoil.

   Measured here by taking the spanwise energy spectrum in a window around each
   x-station and reporting where its peak sits.

Usage:
    python scripts/vortex_structure.py --case 20325_15
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "NeurIPS 2026 RealPDE Competition" / "train_real" / "train_real"
OUT = ROOT / "figures"
DX = 0.001711          # metres per cell, native 64x128 grid
DT = 0.02


def gradients(u, v, dx=DX):
    """Velocity gradient components. Row index runs opposite to y."""
    dudx = np.gradient(u, dx, axis=-1)
    dvdx = np.gradient(v, dx, axis=-1)
    dudy = np.gradient(u, -dx, axis=-2)
    dvdy = np.gradient(v, -dx, axis=-2)
    return dudx, dudy, dvdx, dvdy


def q_criterion(u, v):
    """Q = 0.5 (||W||^2 - ||S||^2) for the in-plane gradient tensor.

    Positive where rotation beats strain (a coherent vortex), negative where
    strain beats rotation (a structure being torn apart).
    """
    dudx, dudy, dvdx, dvdy = gradients(u, v)
    s12 = 0.5 * (dudy + dvdx)          # off-diagonal strain
    w12 = 0.5 * (dvdx - dudy)          # rotation
    s_norm2 = dudx ** 2 + dvdy ** 2 + 2.0 * s12 ** 2
    w_norm2 = 2.0 * w12 ** 2
    return 0.5 * (w_norm2 - s_norm2)


def vorticity(u, v):
    dudx, dudy, dvdx, dvdy = gradients(u, v)
    return dvdx - dudy


def dominant_wavelength(field, dx=DX):
    """Peak of the 1-D spatial spectrum along the row axis, in metres.

    `field` is (T, H) — a strip of the flow at one x-station over time. Each row
    is transformed, the spectra are averaged over time, and the peak returned.
    """
    f = field - field.mean(axis=-1, keepdims=True)
    spec = np.abs(np.fft.rfft(f, axis=-1)) ** 2
    spec = spec.mean(axis=0)
    freqs = np.fft.rfftfreq(f.shape[-1], d=dx)
    spec[0] = 0.0
    k = freqs[np.argmax(spec)]
    return (1.0 / k if k > 0 else np.nan), freqs, spec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="20325_15")
    ap.add_argument("--frame", type=int, default=400)
    ap.add_argument("--frames", type=int, default=300)
    args = ap.parse_args()

    with h5py.File(DATA / f"{args.case}.h5", "r") as f:
        T = f["u"].shape[0]
        n = min(args.frames, T)
        # The snapshot must sit inside the block that is actually read.
        k = min(args.frame, n - 1)
        u = np.asarray(f["u"][:n], dtype=np.float64)
        v = np.asarray(f["v"][:n], dtype=np.float64)
        x = np.asarray(f["x"]); y = np.asarray(f["y"])
        re, aoa = int(f["re"][()]), int(f["aoa"][()])

    mask = (u == 0) & (v == 0)
    q = q_criterion(u, v)
    w = vorticity(u, v)

    # ---------- Figure 1: rotation versus strain ----------
    fig, axes = plt.subplots(3, 1, figsize=(11, 9.5), dpi=115)
    ext = [x.min(), x.max(), y.min(), y.max()]
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#9a9a9a")

    wk = np.where(mask[k], np.nan, w[k])
    lim = float(np.nanpercentile(np.abs(wk), 99))
    im = axes[0].imshow(wk, extent=ext, origin="upper", cmap=cmap,
                        vmin=-lim, vmax=lim, aspect="equal", interpolation="nearest")
    axes[0].set_title(f"vorticity — rotation AND shear together, cannot separate them",
                      fontsize=10)
    fig.colorbar(im, ax=axes[0], fraction=0.026, pad=0.02)

    qk = np.where(mask[k], np.nan, q[k])
    qlim = float(np.nanpercentile(np.abs(qk), 99))
    im = axes[1].imshow(qk, extent=ext, origin="upper", cmap=cmap,
                        vmin=-qlim, vmax=qlim, aspect="equal", interpolation="nearest")
    axes[1].set_title("Q-criterion — red: rotation wins, a coherent vortex.  "
                      "blue: strain wins, the structure is being pulled apart",
                      fontsize=10)
    fig.colorbar(im, ax=axes[1], fraction=0.026, pad=0.02)

    # Coherent cores only, with the velocity field over them.
    core = np.where((qk > 0.25 * qlim), qk, np.nan)
    hot = plt.get_cmap("autumn_r").copy(); hot.set_bad("#f2f2f2")
    axes[2].imshow(core, extent=ext, origin="upper", cmap=hot,
                   vmin=0, vmax=qlim, aspect="equal", interpolation="nearest")
    st = 4
    ys, xs = np.mgrid[0:u.shape[1]:st, 0:u.shape[2]:st]
    uu = np.where(mask[k][ys, xs], np.nan, u[k][ys, xs])
    vv = np.where(mask[k][ys, xs], np.nan, v[k][ys, xs])
    axes[2].quiver(x[ys, xs], y[ys, xs], uu, vv, color="k",
                   scale=float(np.nanpercentile(np.hypot(u[k], v[k]), 99)) * 25,
                   width=0.0016, alpha=0.6)
    axes[2].set_title("coherent cores only (Q above threshold) with the velocity field",
                      fontsize=10)
    for a in axes:
        a.set_xticks([]); a.set_yticks([])

    frac_rot = float(np.nanmean(qk > 0))
    fig.suptitle(f"Rotation versus strain — {args.case}  (Re={re}, AoA={aoa}°), frame {k}\n"
                 f"{100 * frac_rot:.1f}% of measured cells are rotation-dominated",
                 fontsize=12)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    p1 = OUT / f"vortex_q_{args.case}.png"
    fig.savefig(p1, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {p1}")

    # ---------- Figure 2: structure size against downstream distance ----------
    # Measured through the shedding FREQUENCY, not a spatial spectrum. The
    # transverse extent is only 10.8 cm and the wake fills most of it, so a
    # spanwise spectrum peaks at the largest resolvable scale everywhere and says
    # nothing. Time is sampled at 50 Hz over hundreds of frames, so the temporal
    # spectrum at a fixed point resolves the passing structures cleanly, and
    #
    #     wavelength = local convection speed / shedding frequency
    #
    # turns it into a length. Vortex pairing halves the frequency at each merger,
    # so a frequency that falls downstream is that mechanism showing up directly.
    valid_cols = [i for i in range(u.shape[2]) if not mask[:, :, i].any()]
    cols = [c for c in range(min(valid_cols), u.shape[2], 4)]
    lam, xs_m, freqs_hz = [], [], []
    for c in cols:
        # Follow the row with the strongest fluctuation at this station — the
        # wake centreline, which drifts with angle of attack.
        fluct_col = v[:, :, c] - v[:, :, c].mean(axis=0, keepdims=True)
        row = int(np.argmax((fluct_col ** 2).mean(axis=0)))
        sig = fluct_col[:, row]
        spec = np.abs(np.fft.rfft(sig - sig.mean())) ** 2
        ff = np.fft.rfftfreq(len(sig), d=DT)
        spec[0] = 0.0
        f_peak = ff[int(np.argmax(spec))]
        u_conv = float(u[:, row, c].mean())
        lam.append((u_conv / f_peak * 100.0) if f_peak > 0 else np.nan)
        freqs_hz.append(f_peak)
        xs_m.append(x[0, c] * 100.0)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), dpi=115)
    ax[0].plot(xs_m, lam, "o-", ms=4, lw=1.5)
    ax[0].set_xlabel("distance downstream x [cm]")
    ax[0].set_ylabel("structure wavelength [cm]")
    ax[0].set_title("U / f — structure size against downstream distance", fontsize=10)
    ax[0].grid(alpha=0.3)

    ax[2].plot(xs_m, freqs_hz, "^-", ms=4, lw=1.5, color="#4f81bd")
    ax[2].set_xlabel("distance downstream x [cm]")
    ax[2].set_ylabel("shedding frequency [Hz]")
    ax[2].set_title("frequency — a drop means vortices are merging", fontsize=10)
    ax[2].grid(alpha=0.3)

    # Wake width from the transverse profile of fluctuation energy.
    fluct = v - v.mean(axis=0, keepdims=True)
    energy = (fluct ** 2).mean(axis=0)           # (H, W)
    widths = []
    for c in cols:
        col = energy[:, c]
        thr = 0.5 * col.max()
        idx = np.where(col > thr)[0]
        widths.append((idx.max() - idx.min() + 1) * DX * 100.0 if len(idx) else np.nan)
    ax[1].plot(xs_m, widths, "s-", ms=4, lw=1.5, color="#c0504d")
    ax[1].set_xlabel("distance downstream x [cm]")
    ax[1].set_ylabel("wake width at half peak energy [cm]")
    ax[1].set_title("wake spreading", fontsize=10)
    ax[1].grid(alpha=0.3)

    fig.suptitle(f"Downstream evolution — {args.case}  (Re={re}, AoA={aoa}°), "
                 f"{n} frames", fontsize=12)
    fig.tight_layout()
    p2 = OUT / f"wake_scales_{args.case}.png"
    fig.savefig(p2, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {p2}")

    ok = ~np.isnan(lam)
    if ok.sum() > 3:
        a, b = np.polyfit(np.array(xs_m)[ok], np.array(lam)[ok], 1)
        print(f"\nwavelength grows {a:+.3f} cm per cm downstream "
              f"(corr {np.corrcoef(np.array(xs_m)[ok], np.array(lam)[ok])[0, 1]:+.3f})")
    okw = ~np.isnan(widths)
    if okw.sum() > 3:
        a, b = np.polyfit(np.array(xs_m)[okw], np.array(widths)[okw], 1)
        print(f"wake width grows {a:+.3f} cm per cm downstream "
              f"(corr {np.corrcoef(np.array(xs_m)[okw], np.array(widths)[okw])[0, 1]:+.3f})")
    print(f"rotation-dominated share of measured cells: {100 * frac_rot:.1f}%")


if __name__ == "__main__":
    main()
