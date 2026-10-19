"""Composition of one example: Func. 1 and Func. S1 of the paper.

    m = t * (1 - R) + R * ghost(defocus(r))
    then a re-exposure shared by m, t and r so the mean pixel aims at TAU.

Two modes:
  simulate_example  : the reflection is taken as it comes from the pool. Its
                      strength is whatever Fresnel and the relative luminances
                      give it.
  simulate_weak     : the partner is chosen by its luminance so that the
                      mixture falls inside an imposed PSNR band.

Every output is linear sRGB clipped to [0, 1], and `true_t` is the
transmission WITHOUT the attenuation of the glass: that is the network's target.

White balance (MISSION 7, 10/10), `wb`:
  "t"   : m, t, r are balanced to the as-shot white of the transmission's source
          (generator v1.1, bit for bit).
  "mix" : idealized auto white balance of the MIXTURE, as a camera would do it
          (Kee et al. use the C5 white balancer, supp. Func. S2): the two as-shot
          whites are mixed in proportion of the light each layer brings to m,
          w = mean Y(r) / mean Y(m) in XYZ, W = (1 - w) W_t + w W_r (Y = 1). The
          SAME white balances m, t, r and true_t, so input and target stay
          consistent; the transmission is no longer perfectly neutral, the
          reflection's cast is reduced. `wb_dmired` = |mired(W) - mired(W_t)|,
          which the generator uses to reject extreme shifts (like Kee).
"""
from __future__ import annotations

import numpy as np

from .colorimetry import XYZ_D50_TO_SRGB, apply_matrix, cat_matrix
from .config import (BIAS_PRESETS, CULL_SIZE, DEFAULT_BIAS, PATCH, SAT_TOL,
                     TAU, WeakConfig)
from .culling import measure, psnr_raw_to_jpeg
from .pool import make_t_crop, random_crop, make_r_crop
from .scene import (defocus_continuous, deterministic_scene2, double_reflection,
                    sample_scene2, sample_scene3)

WB_KEYS = ("wb_w", "wb_x", "wb_y", "wb_dmired")


def _xy(W):
    W = np.asarray(W, np.float64)
    return float(W[0] / W.sum()), float(W[1] / W.sum())


def mired(W):
    """1e6 / CCT, CCT from McCamy's cubic in xy (fine for usual camera whites)."""
    x, y = _xy(W)
    n = (x - 0.3320) / (0.1858 - y)
    return 1e6 / (449.0 * n ** 3 + 3525.0 * n ** 2 + 6823.3 * n + 5520.33)


def white_point(illum_t, illum_r, m, r, wb="mix", wb_lock=None):
    """White the capture is balanced to (XYZ) and its log parameters.

    m, r: the mixture and the reflection layer, in XYZ, before exposure.
    wb_lock: the white imposed (XYZ array, or (x, y)) -- the members of a group share
    their anchor's white.
    """
    if wb == "t":                    # members of a group share t, hence this white
        W, w = illum_t, 0.0
    elif wb_lock is not None:
        if len(wb_lock) == 3:
            W = np.asarray(wb_lock, np.float64)
        else:
            x, y = wb_lock
            W = np.array([x / y, 1.0, (1.0 - x - y) / y])
        w = float("nan")
    elif wb == "mix":
        Wt = np.asarray(illum_t, np.float64) / float(illum_t[1])
        Wr = np.asarray(illum_r, np.float64) / float(illum_r[1])
        y_m = float(np.mean(m[..., 1], dtype=np.float64))
        w = float(np.mean(r[..., 1], dtype=np.float64)) / max(y_m, 1e-12)
        w = min(max(w, 0.0), 1.0)
        W = (1.0 - w) * Wt + w * Wr
    else:
        raise ValueError(f"wb = {wb!r} (attendu : 'mix' ou 't')")
    x, y = _xy(W)
    return W, {"wb_w": w, "wb_x": x, "wb_y": y,
               "wb_dmired": abs(mired(W) - mired(illum_t))}


def compute_exposure(m, t, r, sat_t, sat_r, M, e_lock=None, tau=TAU,
                     sat_tol=SAT_TOL):
    """Func. S1: conversion to sRGB and choice of the shared exposure gain.

    With no saturated pixel in either source, we aim at tau for the mean pixel
    (tau = srgb_decode(0.4)). As soon as one of the two sources is already
    clipped, aiming at a mean would only clip further, so the gain becomes
    1 / min(t_max, r_max).

    e_lock imposes the gain: that is what lets several reflections share the
    very same `true_t`, bit for bit, inside a group.
    """
    m_s, t_s, r_s = apply_matrix(m, M), apply_matrix(t, M), apply_matrix(r, M)
    if e_lock is not None:
        e = e_lock
    elif sat_t <= sat_tol and sat_r <= sat_tol:
        e = tau / max(m_s.mean(), 1e-8)
    else:
        e = 1.0 / min(float(t_s.max()), float(r_s.max()))
    return m_s * e, t_s * e, r_s * e, e


def apply_exposure(t, M, e):
    return apply_matrix(t, M) * e


def _scene_params(scene, expo):
    """What we keep from the draw so it can be analysed after the fact."""
    return {"f_px": scene["f_px"], "theta0": scene["theta0"],
            "ghost_px": scene["ghost_px"].astype(np.float32),
            "defocus_px": scene["defocus_px"],
            "double_pane": scene["double_pane"],
            "exposure": expo, "R_map": scene["R_map"],"fov": float(scene.get("fov", np.nan)), "N": float(scene.get("N", np.nan)),
            "d_refl": float(scene.get("d_refl", np.nan)), "proche": bool(scene.get("proche", False))}


def simulate_example(src_t, src_r, rng, patch=None, tau=None, det_params=None,
                     deterministic=False, t_pre=None, r_pre=None, e_lock=None,
                     scene=None, bias_dict=None, polar=True, wb="mix",
                     wb_lock=None, bande=None):
    """Func. 1. src_t, src_r = pool entries. Returns m, t, r, true_t, params,
    or None when src_r cannot give a FOV inside the band (pool.make_r_crop).

    t_pre: (crop, saturated_fraction) already drawn, so a transmission can be
    shared between several examples. r_pre: (crop, saturated_fraction, fov);
    careful, it must ALREADY have gone through the random mirror: it is not
    applied here. The FOV of the camera is that of the content of r.
    """
    patch = patch or PATCH
    tau = tau or TAU
    bias_dict = bias_dict if bias_dict is not None else BIAS_PRESETS[DEFAULT_BIAS]

    if t_pre is None:
        t, frac_t = random_crop(src_t["xyz"], patch, src_t["saturation"], rng)
    else:
        t, frac_t = t_pre
    if r_pre is None:
        if scene is not None and "fov" in scene:      # imposed pane: r is cropped to ITS FOV
            bande = (scene["fov"], scene["fov"])
        rc = make_r_crop(src_r, rng, patch, bande)
        if rc is None:
            return None
        r, frac_r, fov = rc
        if rng.random() < 0.5:
            r = r[:, ::-1].copy()
    else:
        r, frac_r, fov = r_pre

    # de-exposure: the pixels become proportional to radiance again
    r = r / src_r["e"]
    t = t / src_t["e"]
    true_t = t.copy()

    if scene is None:
        if deterministic:
            assert det_params, "deterministic=True exige det_params"
            scene = deterministic_scene2(rng, patch, det_params, polar=polar)
        else:
            scene = sample_scene3(rng, patch, fov, bias_dict, polar=polar)

    r = defocus_continuous(r, scene["defocus_px"])
    r = double_reflection(r, scene)              # ghost from both faces
    r = r * scene["R_map"]                       # per-pixel Fresnel
    t = t * (1.0 - scene["R_map"])               # the pane attenuates t as well
    m = t + r                                    # additive mixture

    # chromatic adaptation: one white for m, t, r, true_t (see `wb`)
    W, wbp = white_point(src_t["illum"], src_r["illum"], m, r, wb, wb_lock)
    M = XYZ_D50_TO_SRGB @ cat_matrix(W)
    m, t, r, expo = compute_exposure(m, t, r, frac_t, frac_r, M, e_lock, tau=tau)
    m, t, r, true_t = (np.clip(x, 0.0, 1.0)
                       for x in (m, t, r, apply_exposure(true_t, M, expo)))
    return {"m": m, "t": t, "r": r, "true_t": true_t,
            "params": _scene_params(scene, expo) | wbp, "wb_W": W}


# historical name, kept for the notebooks already written
simulate_example_enc = simulate_example


def prep_t(src_t, rng, patch=None, bias_dict=None, polar=True):
    """Transmission + pane, without the reflection.

    Lets us cull on the attenuation alone before paying for the defocus, which
    is the expensive step.
    """
    patch = patch or PATCH
    bias_dict = bias_dict if bias_dict is not None else BIAS_PRESETS[DEFAULT_BIAS]
    t, frac_t = make_t_crop(src_t, rng, patch)
    M = XYZ_D50_TO_SRGB @ cat_matrix(src_t["illum"])
    t = t / src_t["e"]
    true_t = t.copy()
    scene = sample_scene2(rng, patch, biais=True, biais_params=bias_dict,
                          polar=polar)
    return dict(t_raw=t, true_t=true_t, M=M, scene=scene, frac_t=frac_t,
                t=t * (1.0 - scene["R_map"]))


def compose_weak(pre, alpha, e_lock=None, tau=None, wb_lock=None):
    """m = t(1-R) + alpha * r_R, then Func. S1. The equivalent of
    simulate_example when the layers are already prepared.

    With pre["illum_t"] / pre["illum_r"] the white is computed on THIS mixture
    (it depends on alpha) with pre["wb"]; without them, pre["M"] is used as is
    (older callers)."""
    m = pre["t"] + alpha * pre["r"]
    if "illum_t" in pre:
        W, wbp = white_point(pre["illum_t"], pre["illum_r"], m, alpha * pre["r"],
                             pre.get("wb", "mix"), wb_lock)
        M = XYZ_D50_TO_SRGB @ cat_matrix(W)
    else:
        M, wbp, W = pre["M"], {}, None
    m, t, r, expo = compute_exposure(m, pre["t"], alpha * pre["r"],
                                     pre["frac_t"], pre["frac_r"], M,
                                     e_lock, tau=tau or TAU)
    m, t, r, true_t = (np.clip(x, 0.0, 1.0)
                       for x in (m, t, r,
                                 apply_exposure(pre["true_t"], M, expo)))
    return {"m": m, "t": t, "r": r, "true_t": true_t,
            "params": _scene_params(pre["scene"], expo) | wbp, "wb_W": W}


def simulate_weak(combo, rng, rnd, pool, weak=None, patch=None, bias_dict=None,
                  cull=measure, polar=True, wb="mix", wb_max_mired=None):
    """Weak-reflection example, aimed inside weak.band (JPEG PSNR of m vs true_t).

    Returns (ex, info, ia, ib, t_pre). ex is None when the draw is rejected,
    and info is then a dict {"verdict": reason, "mode": "weak"} -- just as on
    success, so that every trial lands in the same parquet.

    The gain alpha is fixed at weak.alpha (1.0 by default): it is the CHOICE OF
    PARTNER by its luminance that puts the mixture inside the band, not a gain
    tuned by bisection. A gain would have made the reflection weak by also
    making it unrealistic (a real reflection is not attenuated by an arbitrary
    factor).
    """
    weak = weak or WeakConfig()
    patch = patch or PATCH
    lt, lr = combo
    rej = lambda why, ia, ib: (None, {"verdict": why, "mode": "weak"}, ia, ib, None)

    # (1) transmission + pane
    ia = pool.sample_in(lt, rnd)
    # drawn but unused except as the `target_db` log column: the acceptance
    # test below compares against weak.band itself. Kept because removing it
    # would shift the RNG stream of every dataset generated so far.
    tgt = rnd.uniform(*weak.band)
    t_raw, frac_t = make_t_crop(pool[ia], rng, patch)
    M = XYZ_D50_TO_SRGB @ cat_matrix(pool[ia]["illum"])
    t = t_raw / pool[ia]["e"]
    true_t = t.copy()


    # (2) partner chosen by luminance; its crop gives the FOV, THEN the pane
    ib = pool.pick_partner_weak(ia, rnd, lr, logl=weak.logl)
    if ib is None or pool[ib]["name"] == pool[ia]["name"]:
        return rej("pas_de_partenaire", ia, -1)
    src_r = pool[ib]
    rc = make_r_crop(src_r, rng, patch)
    if rc is None:
        return rej("r_hors_bande", ia, ib)
    r, frac_r, fov = rc
    if rng.random() < 0.5:
        r = r[:, ::-1].copy()
    scene = sample_scene3(rng, patch, fov,
                          bias_dict if bias_dict is not None else BIAS_PRESETS[DEFAULT_BIAS],
                          polar=polar)
    t_att = t * (1.0 - scene["R_map"])
    r = r / src_r["e"]
    r = defocus_continuous(r, scene["defocus_px"])
    r = double_reflection(r, scene)
    r = r * scene["R_map"]

    pre = dict(t=t_att, r=r, true_t=true_t, M=M, scene=scene,
               frac_t=frac_t, frac_r=frac_r, illum_t=pool[ia]["illum"],
               illum_r=src_r["illum"], wb=wb)

    # (3) is the band reached without touching alpha?
    ex = compose_weak(pre, weak.alpha)
    wbp = {k: ex["params"][k] for k in WB_KEYS if k in ex["params"]}
    if wb_max_mired is not None and wbp.get("wb_dmired", 0.0) > wb_max_mired:
        return None, {"verdict": "balance_blancs", "mode": "weak", **wbp}, ia, ib, None
    got = psnr_raw_to_jpeg(ex["m"], ex["true_t"])
    if not weak.band[0] <= got < weak.band[1]:
        return None, {"verdict": "hors_bande", "mode": "weak", "jpeg_psnr": got, **wbp}, ia, ib, None

    # (4) structural culling, as for a normal example
    d = cull(ex)
    d.update(mode="weak", alpha=float(weak.alpha), target_db=tgt,
             jpeg_psnr=got, jpeg_psnr_size=CULL_SIZE,
             logratio=float(pool.log_l[ib] - pool.log_l[ia]), **wbp)
    if d["verdict"] != "ok":
        return None, d, ia, ib, None
    return ex, d, ia, ib, (t_raw, frac_t)
