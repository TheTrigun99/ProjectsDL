"""Culling of the simulated examples: measurements, then a verdict.

Three grids of criteria, selected by --cull:

  "std"    -> measure()         : SSIM(m, t) inside a window, non-flat
                                  reflection, correct exposure, structured
                                  transmission.
  "a_tdom" -> measure_a_tdom()  : reasons over 32x32 cells about LOCAL
                                  DOMINANCE (the reflection must dominate part
                                  of the image without crushing the
                                  transmission).
  "kee"    -> measure_kee()     : Kee et al.'s permissive grid, SSIM window
                                  0.40-0.94, std of the SSIM map < 0.50,
                                  mean of m <= 0.70 (mission 7).

Both return a dict of measurements with the key "verdict": "ok", or the name of
the first criterion that fails. That dict is what goes into trials.parquet,
which is why it is worth leaving in the quantities that did not decide the
verdict.

The verdict strings are kept in French ("voile", "exposition", "t_plat",
"pas_de_partenaire"...) because they are the literal values written into
trials.parquet; renaming them would break comparison with the runs already on
disk.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import _paths  # noqa: F401  (inserts main_files into sys.path)
from acr_tone_fast import finish  # noqa: E402

from .colorimetry import LUM, luminance  # noqa: E402
from .config import CULL_SIZE, JPEG_QUALITY  # noqa: E402

# --------------------------------------------------------------------------
# PSNR along the display chain
#
# Linear PSNR says nothing about what the network will see, so we measure after
# the rendering curve, the downscale and a JPEG round trip -- that is, in the
# same space as the real benchmarks. The two helpers below redo
# dataset_sim.square_resize(..., rnd=None) and dataset_sim.jpeg_roundtrip; they
# are copied over so that generation does not depend on torch.
# --------------------------------------------------------------------------
_SAMPLING = {n: getattr(cv2, f"IMWRITE_JPEG_SAMPLING_FACTOR_{n}")
             for n in ("444", "440", "422", "420")}


def center_square_resize(img, size):
    """Square crop centred on the short side, then downscale to size x size."""
    h, w = img.shape[:2]
    c = min(h, w)
    y, x = (h - c) // 2, (w - c) // 2
    crop = img[y:y + c, x:x + c]
    if c == size:
        return crop
    interp = cv2.INTER_AREA if c > size else cv2.INTER_LINEAR
    return cv2.resize(crop, (size, size), interpolation=interp)


def jpeg_roundtrip(rgb_u8, quality=JPEG_QUALITY, sampling="444"):
    """Encode/decode in memory. cv2 expects BGR, hence the channel flip.

    On an encode failure it returns the INPUT unchanged, so a broken encoder
    silently yields a perfect score (capped at 120 dB by the 1e-12 floor in
    psnr_raw_to_jpeg) instead of raising.
    """
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(rgb_u8[..., ::-1]), [
        cv2.IMWRITE_JPEG_QUALITY, int(quality),
        cv2.IMWRITE_JPEG_SAMPLING_FACTOR, _SAMPLING[sampling]])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1] if ok else rgb_u8


def psnr(a, b):
    """PSNR on [0, 1]"""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(1 / max(mse, 1e-10))


def psnr_raw_to_jpeg(M, T, size=CULL_SIZE, quality=JPEG_QUALITY):
    """PSNR of m against true_t after tone mapping, downscale and JPEG."""
    f = lambda b: jpeg_roundtrip(center_square_resize(finish(b, ev=0.0, tone=True),
                                                      size), quality,
                                 "444").astype(np.float32) / 255.0
    M, T = f(M), f(T)
    return 10 * np.log10(1.0 / max(float(((M - T) ** 2).mean()), 1e-12))


# --------------------------------------------------------------------------
# SSIM
# --------------------------------------------------------------------------
def ssim_fast(m, t, data_range=1.0):
    """skimage's defaults (uniform 7x7 window + sample covariance), 4.5x
    faster. The channels are weighted by the mean of m, and the 3 border pixels
    are dropped the way skimage does.
    Returns (mean, standard deviation) of the SSIM map.

    `data_range` is accepted for signature compatibility but IGNORED: C1 and C2
    are hardcoded for a range of 1.0. Do not pass anything else.
    """
    C1, C2 = 1e-4, 9e-4
    cn = 49 / 48                                   # skimage: NP / (NP - 1), NP = 7*7
    f = lambda im: cv2.boxFilter(im, -1, (7, 7), normalize=True,
                                 borderType=cv2.BORDER_REFLECT)
    ux, uy = f(m), f(t)
    vx = cn * (f(m * m) - ux * ux)
    vy = cn * (f(t * t) - uy * uy)
    vxy = cn * (f(m * t) - ux * uy)
    S = ((2 * ux * uy + C1) * (2 * vxy + C2)) / ((ux * ux + uy * uy + C1) * (vx + vy + C2))

    w = np.array([m[..., k].mean() for k in range(3)], np.float64)
    w /= max(w.sum(), 1e-8)
    S = (S[3:-3, 3:-3] * w).sum(axis=-1)
    return float(S.mean()), float(S.std())


# --------------------------------------------------------------------------
# Per-cell descriptors
# --------------------------------------------------------------------------
_S, _P, _SIG = CULL_SIZE, 32, 8.0


def lum(x):
    return np.ascontiguousarray(x.astype(np.float32) @ LUM)


def hf(Y, sig=_SIG):
    """High frequencies: what a Gaussian of sigma `sig` does not capture."""
    return Y - cv2.GaussianBlur(Y, (0, 0), sig)


def cases(X, f, size=None, p=_P):
    """Applies a function per p x p cell. `size` must be a multiple of p."""
    n = (size or X.shape[0]) // p
    return f(X.reshape(n, p, n, p).transpose(0, 2, 1, 3).reshape(n, n, p * p),
             axis=-1)


def _rs(x, size=None):
    return cv2.resize(np.ascontiguousarray(x.astype(np.float32)),
                      (size or _S,) * 2, interpolation=cv2.INTER_AREA)


def descripteurs(m, tp, r, size=None):
    """Every detail of the local dominance, for exploring criteria.

    m, tp (attenuated transmission), r (reflection): linear sRGB HxWx3, with
    m ~ tp + r. Not used by generation: this is the tool that was used to
    calibrate the thresholds of measure_a_tdom.
    """
    size = size or _S
    Ym, Yt, Yr = lum(_rs(m, size)), lum(_rs(tp, size)), lum(_rs(r, size))
    Ht, Hr = hf(Yt), hf(Yr)
    st, sr = cases(Ht, np.std, size), cases(Hr, np.std, size)
    mu = cases(Ym, np.mean, size)
    ct, cr = st / (mu + 1e-3), sr / (mu + 1e-3)         # Weber contrast
    rho = sr / (st + 1e-6)
    lrho = np.log2(np.clip(rho, 1 / 64, 64))
    part_r = cases(Yr, np.mean, size) / (mu + 1e-6)
    actif = (ct + cr) > 0.02                            # cell carries structure
    return dict(
        ratio=float(Hr.std() / (Ht.std() + 1e-8)),
        A_rdom=float((rho > 1).mean()),                 # reflection dominant
        A_rdom2=float((rho > 2).mean()),
        A_rdom4=float((rho > 4).mean()),                # transmission masked
        A_tdom=float((rho < 1).mean()),
        A_tdom2=float((rho < 0.5).mean()),              # transmission dominant
        A_tvis=float((ct > 0.03).mean()),               # transmission visible
        A_tvis_rel=float(((ct > 0.03) & (rho < 2)).mean()),
        rho_med=float(np.median(rho)), lrho_mean=float(lrho.mean()),
        lrho_p10=float(np.percentile(lrho, 10)),
        lrho_p90=float(np.percentile(lrho, 90)),
        lrho_std=float(lrho.std()),                     # spatial heterogeneity
        part_r=float(Yr.mean() / (Ym.mean() + 1e-6)),
        A_rbrillant=float((part_r > 0.5).mean()),
        ct_med=float(np.median(ct)), cr_med=float(np.median(cr)),
        t_std=float(Yt.std()), t_hf=float(Ht.std()), r_hf=float(Hr.std()),
        A_actif=float(actif.mean()),
    )


# --------------------------------------------------------------------------
# Culling grids
# --------------------------------------------------------------------------
def _layers(exa, size=CULL_SIZE):
    rs = lambda x: cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
    mc, tc, rc = rs(exa["m"]), rs(exa["t"]), rs(exa["r"])
    return mc, tc, rc, luminance(mc), luminance(tc), luminance(rc)


def measure(exa, size=CULL_SIZE):
    """Global grid. is_useful(ex) <=> measure(ex)["verdict"] == "ok".

    Needs exa["params"]["defocus_px"], which both simulate_example and
    compose_weak provide.
    """
    mc, tc, rc, Ym, Yt, Yr = _layers(exa, size)
    ssim_m_t, std_m_t = ssim_fast(mc, tc)

    Yr_hf, Yt_hf = hf(Yr), hf(Yt)
    ratio = float(Yr_hf.std() / (Yt_hf.std() + 1e-8))

    d = {"ssim": ssim_m_t, "ssim_std": std_m_t, "Ym": float(Ym.mean()),
         "m_mean": float(mc.mean()),
         "t_std": float(Yt.std()), "r_std": float(Yr.std()),
          "ratio": ratio,}

    if not 0.60 < d["ssim"] < 0.95:          d["verdict"] = "ssim"
    elif not 0.05 <= d["ssim_std"] <= 0.35:  d["verdict"] = "ssim_std"
    elif d["r_std"] < 0.005:                 d["verdict"] = "voile"
    elif ratio > 3:                          d["verdict"] = "reflet_dominant"
    elif not 0.04 < d["Ym"] < 0.30:          d["verdict"] = "exposition"
    elif d["m_mean"] >= 0.7:                 d["verdict"] = "m_mean"
    elif d["t_std"] < 0.015:                 d["verdict"] = "t_plat"
    else:                                    d["verdict"] = "ok"
    return d


def measure_a_tdom(exa, size=CULL_SIZE):
    """Local-dominance grid (datasets net_atdom*).

    Looks for strong reflections, but not impossible ones: the transmission
    must dominate 40-65 % of the cells (A_tdom), the reflection must dominate
    at least 20 % of them by a factor of 2 (A_rdom2), and the 90th percentile
    of the local reflection share must stay at or below 0.8 (p90_part_loc), so
    that the worst cells are not completely drowned.
    """
    mc, tc, rc, Ym, Yt, Yr = _layers(exa, size)
    ssim_m_t, std_m_t = ssim_fast(mc, tc)

    st, sr = cases(hf(Yt), np.std, size), cases(hf(Yr), np.std, size)
    rho = sr / (st + 1e-6)
    mt, mr = cases(Yt, np.mean, size), cases(Yr, np.mean, size)
    part_loc = mr / (mr + mt)
    p90_part_loc = float(np.percentile(part_loc, 90))
    ratio_lum = float(Yr.mean() / Yt.mean())

    d = dict(A_rdom2=float((rho > 2).mean()),
             A_rdom4=float((rho > 4).mean()),
             A_tdom=float((rho < 1).mean()),
             A_tdom2=float((rho < 0.5).mean()),
             part_r=float(Yr.mean() / (Ym.mean() + 1e-6)),
             # measured here anyway: they are useful for diagnosis
             ssim=ssim_m_t, ssim_std=std_m_t, Ym=float(Ym.mean()),
             m_mean=float(mc.mean()), t_std=float(Yt.std()),
             r_std=float(Yr.std()), p90_part_loc=p90_part_loc,
             ratio_lum=ratio_lum,
             sat_frac=float((mc.max(axis=-1) > 0.99).mean()))

    if not 0.4 < d["A_tdom"] < 0.65:          d["verdict"] = "A_tdom"
    elif ratio_lum > 1:                       d["verdict"] = "ratio_lum"
    elif p90_part_loc > 0.8:                  d["verdict"] = "p90_part_loc"
    elif d["part_r"] > 0.50:                  d["verdict"] = "part_r"
    elif d["A_rdom2"] < 0.20:                 d["verdict"] = "A_rdom2"
    elif ssim_m_t < 0.5:                      d["verdict"] = "ssim"
    elif d["r_std"] < 0.07:                   d["verdict"] = "voile_uniforme"
    elif not 0.05 < std_m_t < 0.35:           d["verdict"] = "ssim_std"
    elif not 0.04 < d["Ym"] < 0.3:            d["verdict"] = "exposition"
    elif d["m_mean"] > 0.7:                   d["verdict"] = "mean"
    elif d["t_std"] < 0.020:                  d["verdict"] = "t_std"
    else:                                     d["verdict"] = "ok"
    return d


def measure_kee(exa, size=CULL_SIZE):
    """Kee et al.'s permissive grid (mission 7, culling ablation).

    Same measurements as measure(), only three criteria: 0.40 < SSIM(m, t) < 0.94,
    std of the SSIM map < 0.50, mean of m <= 0.70 (read as a rejection, like the
    m_mean criterion of measure()).
    """
    mc, tc, rc, Ym, Yt, Yr = _layers(exa, size)
    ssim_m_t, std_m_t = ssim_fast(mc, tc)
    ratio = float(hf(Yr).std() / (hf(Yt).std() + 1e-8))

    d = {"ssim": ssim_m_t, "ssim_std": std_m_t, "Ym": float(Ym.mean()),
         "m_mean": float(mc.mean()),
         "t_std": float(Yt.std()), "r_std": float(Yr.std()),
         "ratio": ratio}

    if not 0.40 < d["ssim"] < 0.94:          d["verdict"] = "ssim"
    elif not d["ssim_std"] >0.05:           d["verdict"] = "ssim_std"
    elif d["m_mean"] > 0.70:                 d["verdict"] = "m_mean"
    else:                                    d["verdict"] = "ok"
    return d


CULLERS = {"std": measure, "a_tdom": measure_a_tdom, "kee": measure_kee}


def is_useful(exa, cull="std"):
    return CULLERS[cull](exa)["verdict"] == "ok"
