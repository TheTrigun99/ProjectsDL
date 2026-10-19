"""Visualisation. The only module that imports matplotlib: nothing here is
called by generation, so that the workers stay lightweight.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .colorimetry import (LUM, XYZ_D50_TO_SRGB, XYZ_D65_TO_SRGB, apply_matrix,
                          cat_matrix, luminance, srgb_encode)


def auto_expose(lin, target=0.18):
    """Puts the geometric mean of the luminance on middle grey."""
    key = float(np.exp(np.log(np.clip(luminance(lin), 1e-6, None)).mean()))
    return lin * (target / max(key, 1e-8))


def to_display(lin_srgb, ev=0.0):
    return srgb_encode(lin_srgb * 2.0 ** ev)


def xyz_to_display(xyz, illum=None, white_balance=True):
    """As-shot XYZ -> encoded sRGB, with or without chromatic adaptation.

    `illum` is required when white_balance=True; the default of None then
    raises inside cat_matrix.
    """
    if white_balance:
        lin = apply_matrix(xyz, XYZ_D50_TO_SRGB @ cat_matrix(illum))
    else:
        lin = apply_matrix(xyz, XYZ_D65_TO_SRGB)
    return np.clip(auto_expose(np.clip(lin, 0, None)), 0, 1)


def show_row(images, titles, ev=0.0, size=3.2):
    fig, axes = plt.subplots(1, len(images), figsize=(size * len(images), size))
    for ax, img, title in zip(np.atleast_1d(axes), images, titles):
        ax.imshow(to_display(np.asarray(img, np.float32), ev))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Local dominance: who carries the structure, in displayed space
# --------------------------------------------------------------------------
def pente_srgb(y):
    """Derivative of the sRGB encoding: on-screen contrast of a small linear step."""
    y = np.clip(y, 0.0031308, 1.0)
    return (1.055 / 2.4) * y ** (1 / 2.4 - 1)


def moyenne_par_patch(e, P):
    h, w = e.shape[0] // P, e.shape[1] // P
    return e[:h * P, :w * P].reshape(h, P, w, P).mean(axis=(1, 3))


def carte_dominance(sim, P=32, k=2.0, seuil=0.01, cles=("t", "r", "m")):
    """RMS gradient amplitude of each layer, as it appears in the DISPLAYED
    mixture (what we see is decided by the slope of the sRGB curve at the level
    of m, not by the linear amplitude).

    gT, gR: RMS of the gradient in screen levels (0.01 = 2.5/255 per pixel).
    fR, fT: share of the cells where the reflection / the transmission
            dominates by a factor k while exceeding the threshold.
    fplat : t with no structure.
    """
    t, r, m = (np.asarray(sim[c], np.float32) @ LUM for c in cles)
    poids = pente_srgb(m) ** 2             # the same weight for both layers

    def rms(x):
        gy, gx = np.gradient(x)
        return np.sqrt(moyenne_par_patch((gx ** 2 + gy ** 2) * poids, P))

    gT, gR = rms(t), rms(r)
    dom_R = (gR > k * gT) & (gR > seuil)
    dom_T = (gT > k * gR) & (gT > seuil)
    plat = gT < seuil
    return dict(gT=gT, gR=gR, dom_R=dom_R, dom_T=dom_T, plat=plat,
                fR=float(dom_R.mean()), fT=float(dom_T.mean()),
                fplat=float(plat.mean()))


def affiche_dominance(sim, c=None, cles=("t", "r", "m"), **kw):
    """Mixture, attenuated transmission, reflection, then a log2(gR / gT) map."""
    c = c or carte_dominance(sim, cles=cles, **kw)
    ecran = lambda x: np.clip(np.asarray(x, np.float32), 0, 1) ** (1 / 2.2)
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.6), constrained_layout=True)
    for a, cle, titre in zip(ax, (cles[2], cles[0], cles[1]),
                             ("melange", "transmission attenuee", "reflet")):
        a.imshow(ecran(sim[cle]))
        a.set_title(titre)
        a.axis("off")
    rapport = np.log2((c["gR"] + 1e-6) / (c["gT"] + 1e-6))
    im = ax[3].imshow(rapport, cmap="coolwarm", vmin=-3, vmax=3,
                      interpolation="nearest")
    ys, xs = np.nonzero(c["plat"])         # t has no structure in this cell
    ax[3].plot(xs, ys, ".", color="black", markersize=3)
    ax[3].set_title(f"fR {c['fR']:.2f}   fT {c['fT']:.2f}   fplat {c['fplat']:.2f}")
    ax[3].axis("off")
    fig.colorbar(im, ax=ax[3], shrink=0.8,
                 label="log2(gR / gT) : rouge = reflet domine")
    return fig


# --------------------------------------------------------------------------
# Diagnostics for a run (they read usage.npz / scene_params.parquet)
# --------------------------------------------------------------------------
ACCENT, INK, MUTED = "#3b6ea5", "#1f2328", "#8b949e"


def plot_usage(use_t, use_r):
    """How many times each image of the pool was drawn. The scale is log:
    bin 0 (images never drawn) dwarfs everything else."""
    def count_hist(ax, x, title):
        x = np.asarray(x)
        ax.hist(x, bins=np.arange(int(x.max()) + 2) - 0.5, color=ACCENT,
                edgecolor="white", linewidth=0.6)
        ax.set_yscale("log")
        ax.axvline(x.mean(), color=INK, ls="--", lw=1.2)
        ax.text(x.mean(), ax.get_ylim()[1], f"  mean {x.mean():.2f}",
                ha="left", va="top", fontsize=8, color=INK)
        ax.set_title(f"{title} - {int((x == 0).sum())}/{x.size} jamais tirees, "
                     f"max {int(x.max())}", fontsize=10, color=INK)
        ax.set_xlabel("nb de tirages")
        ax.set_ylabel("nb d'images (log)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=MUTED, alpha=.25, lw=.6)
        ax.set_axisbelow(True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    count_hist(axes[0], use_t, "transmission t")
    count_hist(axes[1], use_r, "reflet r")
    fig.tight_layout()
    return fig


def plot_scene_params(params, ncols=3):
    """Distribution of each pane parameter that was actually kept.

    params: a DataFrame from scene_params.parquet, or a dict of lists.
    """
    if hasattr(params, "to_dict"):
        params = {c: params[c].to_numpy() for c in params.columns}
    cols = {k: np.asarray(v) for k, v in params.items() if len(v) > 0}
    nrows = -(-len(cols) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.asarray(axes).reshape(-1)
    for ax, (name, arr) in zip(axes, cols.items()):
        if np.issubdtype(arr.dtype, np.number) or arr.dtype == bool:
            flat = arr.astype(float).ravel()
            print(f"{name:15s} n={len(flat):6d} mean={flat.mean():.4g} "
                  f"std={flat.std():.4g} min={flat.min():.4g} "
                  f"max={flat.max():.4g} median={np.median(flat):.4g}")
            ax.hist(flat, bins=30, color="#4C72B0", edgecolor="black", alpha=0.85)
        else:
            cats, counts = np.unique(arr, return_counts=True)
            print(f"{name:15s} n={len(arr):6d}  "
                  + ", ".join(f"{c}={n}" for c, n in zip(cats, counts)))
            ax.bar(cats.astype(str), counts, color="#DD8452", edgecolor="black")
            ax.tick_params(axis="x", rotation=45)
        ax.set_title(name)
    for ax in axes[len(cols):]:
        ax.axis("off")
    plt.tight_layout()
    return fig
