"""Weak reflections reached with a bounded gain alpha (dataset_weak_27_5dB).

MISSION 7 (10/10): port of mission_claude/gen_weak2.py onto the package's physics (simulator v2: Fresnel per
polarization, white balance of the mixture). Same sampling, same bisection, same structural culling; one
deliberate change: the Laval EXR views are no longer in the reflection pool (Laval is evaluation only).

Method (gen_weak2, authorized by Damien on 23/09; alpha accepted again on 10/10):
  1. transmission among the PHOTOS (FiveK + RAISE), at most `max_use` uses; pane drawn; psnr0 = PSNR of the
     example WITHOUT reflection (attenuation only), rejected if < lo_db + marge_min (one image pass, before any
     reflection processing);
  2. reflection partner chosen by luminance, log10(L_r / L_t) in [-0.70, 0.10], IBL preferred with probability
     `ibl_frac` (pair selection only, nothing changed in the simulation);
  3. reflection processed (defocus, double reflection, Fresnel), then alpha in [0.25, 4] (+/- 2 stops) found by
     bisection so that the JPEG PSNR of m against true_t hits a target drawn in [lo_db, hi_db];
  4. margin: PSNR without reflection minus PSNR with it >= marge_min; then the notebook's structural culling
     (r_std, ratio + defocus, Ym, m_mean, t_std, saturation).

    python -m generating_sim_data generate-weak2 --out DIR --n 30000 --seed 7717 --lo-db 27.5 --hi-db 30
"""
from __future__ import annotations

import fcntl
import json
import os
import random
import re
import signal
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from . import _paths  # noqa: F401
from acr_tone_fast import finish  # noqa: E402
from npz_octets import ecrit_octets  # noqa: E402

from .colorimetry import XYZ_D50_TO_SRGB, apply_matrix, cat_matrix  # noqa: E402
from .config import BIAS_PRESETS, CULL_SIZE, PATCH, TAU  # noqa: E402
from .culling import jpeg_roundtrip, measure  # noqa: E402
from .pool import make_t_crop  # noqa: E402
from .scene import defocus_continuous, double_reflection, sample_scene2  # noqa: E402
from .simulate import WB_KEYS, _scene_params, apply_exposure, compute_exposure, white_point  # noqa: E402

ALPHA_LO, ALPHA_HI = 0.25, 4.0      # +/- 2 stops
LOG_LO, LOG_HI = -0.70, 0.10        # useful band of log10(L_r / L_t), measured (gen_weak2)


def jpeg_psnr(ex, size=384, key="true_t"):
    """gen_weak.jpeg_psnr: PSNR of m against `key` after resize (linear, INTER_AREA) + finish + JPEG q95 4:4:4."""
    rs = lambda x: cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
    f = lambda a: jpeg_roundtrip(np.ascontiguousarray(finish(a, ev=0.0, tone=True)), 95, "444"
                                 ).astype(np.float32) / 255.
    A, B = f(rs(ex["m"])), f(rs(ex[key]))
    return 10 * np.log10(1. / max(float(((A - B) ** 2).mean()), 1e-12))


def prep_t(src_t, rng, bias_dict, polar=True):
    """Transmission layer + pane, no reflection yet (lets us cull on psnr0 before the costly defocus)."""
    t, frac_t = make_t_crop(src_t, rng, PATCH)
    M = XYZ_D50_TO_SRGB @ cat_matrix(src_t["illum"])
    t = t / src_t["e"]
    true_t = t.copy()
    scene = sample_scene2(rng, PATCH, biais=True, biais_params=bias_dict, polar=polar)
    return dict(t_raw=t, true_t=true_t, M=M, scene=scene, frac_t=frac_t, t=t * (1.0 - scene["R_map"]),
                illum_t=src_t["illum"])


def compose(pre, alpha, e_lock=None, wb="mix", wb_lock=None):
    """m = t + alpha r, white balance of the mixture (v2), re-exposure (Func. S1), clipping."""
    m = pre["t"] + alpha * pre["r"]
    W, wbp = white_point(pre["illum_t"], pre["illum_r"], m, alpha * pre["r"], wb, wb_lock)
    M = XYZ_D50_TO_SRGB @ cat_matrix(W)
    m, t, r, expo = compute_exposure(m, pre["t"], alpha * pre["r"], pre["frac_t"], pre["frac_r"], M,
                                     e_lock=e_lock)
    clip = lambda x: np.clip(x, 0.0, 1.0)
    true_t = clip(apply_exposure(pre["true_t"], M, expo))
    params = _scene_params(pre["scene"], expo) | wbp | {"alpha": float(alpha)}
    return {"m": clip(m), "t": clip(t), "r": clip(r), "true_t": true_t, "params": params, "wb_W": W}


def psnr0_of(pt, wb="mix"):
    """PSNR of the example without reflection, exposure locked to the one a null reflection would give."""
    e = TAU / max(float(apply_matrix(pt["t"], pt["M"]).mean()), 1e-8)
    pre = dict(pt, r=np.zeros_like(pt["t"]), frac_r=pt["frac_t"], illum_r=pt["illum_t"])
    return jpeg_psnr(compose(pre, 0.0, e_lock=e, wb=wb), 256), e


def add_reflection(pt, src_r, rng):
    r, frac_r = make_t_crop(src_r, rng, PATCH)
    if rng.random() < 0.5:
        r = r[:, ::-1].copy()
    r = r / src_r["e"]
    r = defocus_continuous(r, pt["scene"]["defocus_px"])
    r = double_reflection(r, pt["scene"])
    r = r * pt["scene"]["R_map"]
    return dict(pt, r=r, frac_r=frac_r, illum_r=src_r["illum"])


def solve_alpha(pre, target, wb="mix", iters=7, size=256):
    """Bisection at 256 px on [0.25, 4]; the final PSNR is recomputed at 384 (gen_weak2)."""
    lo, hi, ex = ALPHA_LO, ALPHA_HI, None
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        ex = compose(pre, mid, wb=wb)
        got = jpeg_psnr(ex, size)
        if got < target:
            hi = mid
        else:
            lo = mid
    ex["alpha"] = mid
    ex["psnr_got"] = jpeg_psnr(ex, 384)
    return ex


def sat_frac(ex, size=CULL_SIZE):
    mc = cv2.resize(ex["m"], (size, size), interpolation=cv2.INTER_AREA)
    return float((mc.max(axis=-1) > 0.99).mean())


def culling_weak2(d, defocus_px):
    """The notebook's structural culling used by gen_weak2 (no SSIM criterion)."""
    if d["r_std"] < 0.0015:                          return "voile"
    if d["ratio"] > 2 and defocus_px < 1.0:          return "reflet_dominant"
    if not 0.04 < d["Ym"] < 0.20:                    return "exposition"
    if d["m_mean"] >= 0.7:                           return "m_mean"
    if d["t_std"] < 0.015:                           return "t_plat"
    if d["sat_frac"] > 0.10:                         return "satures"
    return "ok"


def run_weak2(out, n, seed, workers=16, lo_db=27.5, hi_db=30.0, ibl_frac=0.4, max_use=6, marge_min=2.5,
              bias="defocus_high", fresnel="polar", wb="mix", wb_max_mired=60.0, group_offset=0, pool=None):
    import pandas as pd

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    verrou_dossier = open(out / ".weak2_verrou", "w")            # one batch at a time in a folder
    try:
        fcntl.flock(verrou_dossier, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"{out} : un autre lot generate-weak2 tourne dans ce dossier")
    s = f"_{group_offset:05d}" if group_offset else ""
    assert not (out / f"meta{s}.json").exists(), f"meta{s}.json existe : lot deja fait (choisir --group-offset)"
    deja = {int(m.group(1)) for f in os.listdir(out) if (m := re.match(r"ex_(\d+)_00\.npz$", f))}
    chevauche = sorted(deja & set(range(group_offset, group_offset + n)))
    assert not chevauche, f"ex_{chevauche[0]:05d}_00.npz existe deja : choisir --group-offset >= {max(deja) + 1}"
    if pool is None:
        from .pool import build_pool
        pool = build_pool(("fivek", "raise", "ibl"), with_exr=False)
    E = pool.entries
    photos = [i for i, e in enumerate(E) if e["loc"] != "ibl"]          # transmissions: photos only
    n_photo = len(photos)
    assert photos == list(range(n_photo)), "le pool doit commencer par les photos (ordre fivek, raise, ibl)"
    logL = np.asarray(pool.log_l, np.float64)                           # all entries (photos + IBL)
    order = np.argsort(logL)
    sortedL = logL[order]
    is_ibl = order >= n_photo
    bias_dict = BIAS_PRESETS[bias]
    polar = fresnel == "polar"

    lock = threading.Lock()
    st = {"saved": 0, "tried": 0, "stop": False}
    signal.signal(signal.SIGTERM, lambda *_: st.__setitem__("stop", True))   # workers leave, logs are written
    use_t, use_r, rows, rej = Counter(), Counter(), [], Counter()
    t0 = time.time()

    def pick_partner(ia, rnd):
        a = np.searchsorted(sortedL, logL[ia] + LOG_LO)
        b = np.searchsorted(sortedL, logL[ia] + LOG_HI)
        if b <= a:
            return None
        want_ibl = rnd.random() < ibl_frac
        for _ in range(12):
            k = rnd.randrange(a, b)
            ib = int(order[k])
            if ib == ia or use_r[ib] >= max_use * 3:
                continue
            if is_ibl[k] == want_ibl:
                return ib
        return int(order[rnd.randrange(a, b)])

    def ecrit_manifeste():
        if rows:
            fn = out / f"trials_groups{s}.parquet"
            pd.DataFrame(rows).to_parquet(str(fn) + ".tmp")
            os.replace(str(fn) + ".tmp", fn)

    def worker(w):
        cv2.setNumThreads(1)
        rng = np.random.default_rng([seed, w])
        rnd = random.Random(seed * 1_000_003 + w)
        n_loc = 0
        while True:
            with lock:
                if st["saved"] >= n or st["stop"]:
                    return
                st["tried"] += 1
                ia = None
                for _ in range(60):
                    c = rnd.randrange(n_photo)
                    if use_t[c] < max_use:
                        ia = c
                        break
                if ia is None:
                    return
            try:
                pt = prep_t(E[ia], rng, bias_dict, polar)
                p0, _ = psnr0_of(pt, wb)
                if p0 < lo_db + marge_min:
                    with lock:
                        rej["attenuation_domine"] += 1
                    continue
                with lock:
                    ib = pick_partner(ia, rnd)
                    if ib is None:
                        rej["pas_de_partenaire"] += 1
                        continue
                if E[ia]["name"] == E[ib]["name"]:
                    with lock:
                        rej["meme_image"] += 1
                    continue
                pre = add_reflection(pt, E[ib], rng)
                tgt = rnd.uniform(lo_db, hi_db)
                ex = solve_alpha(pre, tgt, wb)
                if abs(ex["psnr_got"] - tgt) > 0.5:
                    with lock:
                        rej["alpha_borne"] += 1
                    continue
                p = ex["params"]
                if wb == "mix" and p["wb_dmired"] > wb_max_mired:
                    with lock:
                        rej["balance_blancs"] += 1
                    continue
                # same white as ex: the margin isolates the reflection from the attenuation
                p0v = jpeg_psnr(compose(pre, 0.0, e_lock=p["exposure"], wb=wb, wb_lock=ex["wb_W"]), 384)
                if p0v - ex["psnr_got"] < marge_min:
                    with lock:
                        rej["marge_faible"] += 1
                    continue
                d = measure(ex)
                d["sat_frac"] = sat_frac(ex)
                v = culling_weak2(d, pt["scene"]["defocus_px"])
                if v != "ok":
                    with lock:
                        rej[v] += 1
                    continue
            except Exception as e:
                with lock:
                    rej[f"exc:{type(e).__name__}"] += 1
                continue

            # loc95 (gen_weak2): share of |m - true_t| carried by its 5 % largest pixels
            _d = np.abs(ex["m"].astype(np.float32) - ex["true_t"].astype(np.float32)).mean(-1)
            _s = max(float(_d.sum()), 1e-9)
            d.update(loc95=float(_d[_d >= np.quantile(_d, .95)].sum() / _s),
                     jpeg_psnr_384=ex["psnr_got"], alpha=ex["alpha"], target_db=tgt, psnr_attenuation_only=p0v,
                     marge_reflet_db=p0v - ex["psnr_got"], logratio=float(logL[ib] - logL[ia]), verdict="ok",
                     mode="weak2", **{k: p[k] for k in WB_KEYS})
            # npz written under a temporary name OUTSIDE the lock, renamed once its index is reserved
            n_loc += 1
            tmp = out / f"tmp_w{w:02d}_n{n_loc}.npz.tmp"
            scal = {f"p_{x}": np.asarray(v_) for x, v_ in p.items() if x not in ("R_map", "ghost_px")}
            try:
                with open(tmp, "wb") as fh:
                    ecrit_octets({"m": ex["m"].astype(np.float16), "true_t": ex["true_t"].astype(np.float16),
                                  "R_map": np.asarray(p["R_map"]).astype(np.float16),
                                  "ghost_px": np.asarray(p["ghost_px"]).astype(np.float16), **scal,
                                  "member": 0, "t_name": E[ia]["name"], "r_name": E[ib]["name"]}, fh)
            except Exception as e:
                with lock:
                    rej[f"exc_ecriture:{type(e).__name__}"] += 1
                if tmp.exists():
                    tmp.unlink()
                continue
            with lock:
                if st["saved"] >= n or st["stop"]:
                    tmp.unlink()
                    return
                k = group_offset + st["saved"]
                dest = out / f"ex_{k:05d}_00.npz"
                if dest.exists():                                 # never overwrite a file
                    rej["collision"] += 1
                    tmp.unlink()
                    continue
                os.replace(tmp, dest)
                st["saved"] += 1
                use_t[ia] += 1
                use_r[ib] += 1
                rows.append({**d, "group": k, "K": 1, "t_name": E[ia]["name"], "r_name": E[ib]["name"],
                             "t_loc": E[ia]["loc"], "r_loc": E[ib]["loc"],
                             **{f"p_{x}": float(p[x]) for x in ("defocus_px", "theta0", "f_px", "exposure")},
                             "p_double_pane": bool(p["double_pane"]),
                             "p_ghost_px": float(np.asarray(p["ghost_px"]).mean())})
                if st["saved"] % 500 == 0:
                    ecrit_manifeste()
                    el = time.time() - t0
                    print(f"[weak2] {st['saved']}/{n}  {st['tried']} tirages  {el / 60:5.1f} min  "
                          f"{st['saved'] / el * 3600:6.0f}/h  {dict(rej.most_common(6))}", flush=True)

    tp = ThreadPoolExecutor(workers)
    futs = [tp.submit(worker, w) for w in range(workers)]
    for f in futs:
        f.result()
    tp.shutdown(wait=True)
    el = time.time() - t0
    ecrit_manifeste()
    meta = dict(generateur="generating_sim_data.weak2 (v2)", n=n, seed=seed, workers=workers, lo_db=lo_db,
                hi_db=hi_db, ibl_frac=ibl_frac, max_use=max_use, marge_min=marge_min, alpha=[ALPHA_LO, ALPHA_HI],
                logl=[LOG_LO, LOG_HI], bias=bias, bias_dict=bias_dict, fresnel=fresnel, wb=wb,
                wb_max_mired=wb_max_mired, group_offset=group_offset, saved=st["saved"], tried=st["tried"],
                interrompu=bool(st["stop"]),
                rejets=dict(rej), elapsed_s=round(el, 1), exr=False)
    (out / f"meta{s}.json").write_text(json.dumps(meta, indent=2))
    print(f"[weak2] fini : {st['saved']} exemples, {st['tried']} tirages, {el / 60:.1f} min ; rejets {dict(rej)}",
          flush=True)
    return 0 if st["saved"] == n else 1
