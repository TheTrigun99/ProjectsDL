"""Sharp "hard but readable" reflections (dataset_30knet_mid) and "same scene" reflections (dataset_meme_scene).

MISSION 7 (10/10): port of mission4/gen_mid.py + mission4/crit4.py + mission2/simgen.py onto the package's
physics (simulator v2: Fresnel per polarization, white balance of the mixture). The SAMPLING is mission 4's,
unchanged; only the physics backend (mission 4 ran a frozen copy of data_clean.ipynb) is replaced.

One attempt (seeds np.random.default_rng([seed, offset, w, n, 0 or 1]), w = worker, n = counter):
  1. cell (bucket, regime) drawn in proportion of the remaining quotas;
  2. transmission among the bucket's kinds (weights = cap - usage, x preference);
  3. reflection among the bucket's kinds, LOG_L[r] - LOG_L[t] inside the cell's window, never the same panorama
     (same-scene buckets: ANOTHER, non-adjacent view of the SAME panorama, or the other half of the same photo);
  4. draw = crops + mirror + pane (all the physical randomness, `tirer`);
  5. early statistics at alpha = 1, before exposure -> feasible interval of alpha for the targeted regime
     (`choisir_alpha_lu`: accepted with probability log-width / log 12, alpha log-uniform inside = the law of the
     full path alpha ~ log-uniform [0.5, 6] without composing the infeasible draws), or rejection BEFORE composing;
  6. composition (`composer`), exact criteria on ex["t"], ex["r"] (mission 3 §9 rule + revised culling §10 +
     regime), white-balance shift check (v2);
  7. accepted if the reached cell still has quota (atomic check + increment); npz written (tmp then rename).

alpha multiplies the reflection's radiance (a free gain in [0.5, 6], as in mission 4; Damien accepted it).
The early statistics use the transmission's white (as mission 4 did); with wb="mix" the white of the composed
example depends slightly on alpha, which the x1.5 margin on the alpha interval absorbs; the exact criteria are
recomputed after composition anyway.

    python -m generating_sim_data generate-mid --out DIR --n 20000 --seed 4400 --offset 0 [--meme-scene]
"""
from __future__ import annotations

import fcntl
import glob
import json
import os
import re
import signal
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from . import _paths  # noqa: F401
from npz_octets import ecrit_octets  # noqa: E402

from .colorimetry import LUM, XYZ_D50_TO_SRGB, apply_matrix, cat_matrix  # noqa: E402
from .config import (BANDE_FOV, BIAS_PRESETS, DEFOCUS_MAX_PX, K_MAP, N_FERME, N_OUVERT,  # noqa: E402
                     P_OUVERT, P_PROCHE, PATCH, S_PROCHE, SENSOR_W_c, TAU)  # noqa: E402
from .culling import measure  # noqa: E402
from .pool import make_r_crop, random_crop  # noqa  # noqa: E402
from .scene import defocus_continuous, double_reflection, sample_scene3  # noqa: E402
from .simulate import WB_KEYS, _scene_params, compute_exposure, white_point  # noqa: E402

# --------------------------------------------------------------------------- mission 4 sampling (gen_mid.py)
# (kinds of t, kinds of r, share); kinds: outdoor (FiveK), indoor (FiveK), raise (RAISE outdoor), ibl_in (indoor IBL;
# the "outdoor" IBL are nights exposed as days: excluded)
BUCKETS = {
    "b1_raise_x_out":   (("raise",), ("outdoor", "raise"), 0.15),
    "b2_raise_x_raise": (("raise",), ("raise",), 0.15),
    "b3_raise_x_in":    (("raise",), ("indoor", "ibl_in"), 0.20),
    "b4_in_x_raise":    (("indoor", "ibl_in"), ("raise",), 0.20),
    "b5_in_x_ibl":      (("indoor",), ("ibl_in",), 0.15),
    "b6_ibl_x_in":      (("ibl_in", "indoor"), ("indoor",), 0.15),
}
PREF_T = {"b4_in_x_raise": {"indoor": 3.0, "ibl_in": 1.0}}
#   ms_ibl   : r = ANOTHER view of the SAME IBL panorama as t (non-adjacent: 6 views 60 deg apart, 90 deg FOV);
#   ms_photo : t and r in the two HALVES (left / right) of the same RAW photo.
BUCKETS_MS = {"ms_ibl": (("ibl_in",), ("ibl_in",), 0.5), "ms_photo": (("outdoor", "indoor", "raise"), None, 0.5)}
PARTS_REGIME = {"mid": 0.80, "haut": 0.20}
FENETRE_LARGE = (-9.0, 9.0)
FENETRES_DEFAUT = Path(__file__).resolve().parent / "fenetres_mid.json"   # = mission4/fenetres.json

# --------------------------------------------------------------------------- mission 4 criteria (crit4.py)
S, P, SIG = 256, 32, 8.0
NC = (S // P) ** 2
EPS_RHO = 1e-6
SEUILS = dict(A_tdom_min=0.40, ratio_lum_max=1.0, p90_max=0.80)
MID = (26, 41)
HAUT = (42, 64)
A_RDOM2_MID = 0.25
ALPHA_MIN, ALPHA_MAX = 0.5, 6.0
MARGE_ALPHA = 1.5


def lum(x):
    return np.ascontiguousarray(x.astype(np.float32) @ LUM)


def hf(Y, sig=SIG):
    return Y - cv2.GaussianBlur(Y, (0, 0), sig)


def cases(X, f):
    n = S // P
    return f(X.reshape(n, P, n, P).transpose(0, 2, 1, 3).reshape(n, n, P * P), axis=-1)


def rs(x):
    return cv2.resize(np.ascontiguousarray(x.astype(np.float32)), (S, S), interpolation=cv2.INTER_AREA)


def stats(tp, r):
    Yt, Yr = lum(rs(tp)), lum(rs(r))
    return dict(st=cases(hf(Yt), np.std), sr=cases(hf(Yr), np.std),
                mt=cases(Yt, np.mean), mr=cases(Yr, np.mean),
                Lt=float(Yt.mean()), Lr=float(Yr.mean()))


def _p90(mr, mt):
    return float(np.percentile(mr / np.maximum(mr + mt, 1e-12), 90))


def finaux(tp, r):
    s = stats(tp, r)
    rho = s["sr"] / (s["st"] + EPS_RHO)
    return dict(A_tdom=float((rho < 1).mean()), A_rdom2=float((rho > 2).mean()), k_tdom=int((rho < 1).sum()),
                ratio_lum=s["Lr"] / max(s["Lt"], 1e-8), p90_part_loc=_p90(s["mr"], s["mt"]))


def culling(d):
    if not d["ssim"] > 0.5:                  return "ssim"
    if not 0.05 <= d["ssim_std"] <= 0.35:    return "ssim_std"
    if d["r_std"] < 0.005:                   return "voile"
    if not 0.04 < d["Ym"] < 0.30:            return "exposition"
    if not d["m_mean"] < 0.7:                return "m_mean"
    if d["t_std"] < 0.015:                   return "t_plat"
    return "ok"


def regime(f):
    if MID[0] <= f["k_tdom"] <= MID[1] and f["A_rdom2"] >= A_RDOM2_MID:
        return "mid"
    if f["k_tdom"] >= HAUT[0]:
        return "haut"
    return None


def verdict(f, d):
    if f["A_tdom"] < SEUILS["A_tdom_min"]:   return "A_tdom<0.4"
    if not f["ratio_lum"] <= SEUILS["ratio_lum_max"]: return "ratio_lum>1"
    if not f["p90_part_loc"] <= SEUILS["p90_max"]: return "p90>0.8"
    c = culling(d)
    if c != "ok":                            return c
    if regime(f) is None:                    return "hors_regime"
    return "ok"


def precoce(tir):
    """Statistics at alpha = 1, BEFORE exposure and clipping, with the transmission's white (tir["M"])."""
    sc, R, M = tir["scene"], tir["scene"]["R_map"], tir["M"]
    r = double_reflection(defocus_continuous(tir["r"], sc["defocus_px"]), sc)
    s = stats(apply_matrix(tir["t"] * (1.0 - R), M), apply_matrix(r * R, M))
    s["rho1"] = s["sr"] / (s["st"] + EPS_RHO)
    return s


def alpha_max_p90(s, hi):
    f = lambda a: _p90(a * s["mr"], s["mt"])
    if f(hi) <= SEUILS["p90_max"]:
        return hi
    lo = 1e-3
    if f(lo) > SEUILS["p90_max"]:
        return 0.0
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if f(mid) <= SEUILS["p90_max"] else (lo, mid)
    return lo


def choisir_alpha_lu(s, cible, rng):
    v = np.sort(s["rho1"].ravel())
    rl1 = s["Lr"] / max(s["Lt"], 1e-8)
    a_hi = alpha_max_p90(s, min(ALPHA_MAX, 1.0 / max(rl1, 1e-8)))
    a_lo = ALPHA_MIN
    inv = lambda x: 1.0 / x if x > 0 else np.inf
    if cible == "mid":
        a_lo = max(a_lo, 2.0 / max(v[NC - 16], 1e-12) * (1 + 1e-6))
        lo, hi = max(a_lo, inv(v[MID[1]])), min(a_hi, inv(v[MID[0] - 1]))
    else:
        lo, hi = a_lo, min(a_hi, inv(v[HAUT[0] - 1]))
    info = dict(rl1=rl1, a_lo=lo, a_hi=hi, k1=int((v < 1).sum()))
    lo, hi = max(ALPHA_MIN, lo / MARGE_ALPHA), min(ALPHA_MAX, hi * MARGE_ALPHA)
    info.update(a_lo=lo, a_hi=hi)
    if not hi > lo:
        return None, "alpha_vide", info
    frac = np.log(hi / lo) / np.log(ALPHA_MAX / ALPHA_MIN)
    info["frac"] = float(frac)
    if rng.random() >= frac:
        return None, "non_tire", info
    alpha = float(lo * (hi / lo) ** rng.random())
    return alpha, int((v * alpha < 1).sum()), info


# --------------------------------------------------------------------------- physics (simgen.py, package v2)
def tirer(src_t, src_r, rng, patch=PATCH, bias_dict=None, polar=True):
    """Crops + mirror + pane: all the physical randomness. Same draw order as simulate_example."""
    t, frac_t = random_crop(src_t["xyz"], patch, src_t["saturation"], rng)
    rc = make_r_crop(src_r, rng, patch)
    if rc is None:                                   # source (or half) unable to give a FOV inside the band
        return None
    r, frac_r, fov = rc
    if rng.random() < 0.5:
        r = r[:, ::-1].copy()
    scene = sample_scene3(rng, patch, fov,
                          bias_dict if bias_dict is not None else BIAS_PRESETS["defocus_low"], polar=polar)
    M = XYZ_D50_TO_SRGB @ cat_matrix(src_t["illum"])
    return dict(t=t / src_t["e"], r=r / src_r["e"], frac_t=frac_t, frac_r=frac_r, M=M, scene=scene,
                t_name=src_t["name"], r_name=src_r["name"], illum_t=src_t["illum"], illum_r=src_r["illum"])


def composer(tir, alpha=1.0, wb="mix"):
    """m = t (1 - R) + R ghost(defocus(alpha r)), white balance of the mixture, Func. S1 exposure, clipping."""
    sc = tir["scene"]
    R = sc["R_map"]
    t = tir["t"]
    true_t = t.copy()
    r = tir["r"] * alpha
    r = defocus_continuous(r, sc["defocus_px"])
    r = double_reflection(r, sc)
    r = r * R
    t = t * (1.0 - R)
    m = t + r
    W, wbp = white_point(tir["illum_t"], tir["illum_r"], m, r, wb)
    M = XYZ_D50_TO_SRGB @ cat_matrix(W)
    m, t, r, expo = compute_exposure(m, t, r, tir["frac_t"], tir["frac_r"], M, None, tau=TAU)
    true_t = apply_matrix(true_t, M) * expo
    m, t, r, true_t = (np.clip(x, 0.0, 1.0).astype(np.float32) for x in (m, t, r, true_t))
    params = _scene_params(sc, expo) | wbp | {"alpha": float(alpha), "R_mean": float(R.mean())}
    return {"m": m, "t": t, "r": r, "true_t": true_t, "params": params}


# --------------------------------------------------------------------------- sources
def kind(src):
    """simgen.kind: outdoor / indoor (FiveK), raise (RAISE outdoor), ibl_in / ibl_out, exr (Laval)."""
    if src.get("loc") == "ibl":
        cats = src.get("cats") or {}
        tags = cats.get("tags", []) if isinstance(cats, dict) else []
        if "laval" in tags or "exr" in tags:
            return "exr"
        categories = cats.get("categories", []) if isinstance(cats, dict) else []
        return "ibl_out" if "outdoor" in categories else "ibl_in"
    if src.get("raise_outdoor"):
        return "raise"
    return src["loc"]


def moitie(src, cote):
    """Source restricted to its left (g) or right (d) half: memmap views, no copy."""
    W = src["xyz"].shape[1] // 2
    sl = np.s_[:, :W] if cote == "g" else np.s_[:, W:]
    return {**src, "xyz": src["xyz"][sl], "saturation": src["saturation"][sl]}


def table_sources(pool):
    P = pool.entries
    kinds = np.array([kind(s) for s in P])
    pano = np.array([s.get("src") or s["name"] for s in P])
    noms = np.array([s["name"] for s in P])
    garde = np.isin(kinds, ["outdoor", "indoor", "raise", "ibl_in"])     # never exr (Laval = evaluation)
    idx = np.flatnonzero(garde)
    vue = np.array([int(m.group(1)) if (m := re.search(r"_v(\d+)$", n)) else -1 for n in noms])
    return dict(i=idx, kind=kinds[idx], pano=pano[idx], nom=noms[idx],
                logl=np.asarray(pool.log_l, np.float64)[idx], vue=vue[idx])


def quotas(n, parts_b, parts_g):
    cles = [(b, g) for b in parts_b for g in parts_g]
    brut = np.array([n * parts_b[b] * parts_g[g] for b, g in cles])
    q = np.floor(brut).astype(int)
    for i in np.argsort(-(brut - q))[: n - q.sum()]:
        q[i] += 1
    return cles, q


# --------------------------------------------------------------------------- driver
def run_mid(out, n, seed, offset=0, k0=0, workers=16, cap=100000, meme_scene=False, fenetres=None,
            bias="defocus_low", fresnel="polar", wb="mix", wb_max_mired=60.0, duree_max=1e9, pool=None):
    import multiprocessing as mp
    import pandas as pd

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    verrou_dossier = open(out / ".mid_verrou", "w")              # one batch at a time in a folder
    try:
        fcntl.flock(verrou_dossier, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"{out} : un autre lot generate-mid tourne dans ce dossier")
    assert not glob.glob(str(out / f"manifeste_s{seed}_o{offset}_w*.parquet")), "lot (seed, offset) deja present"
    assert not (out / f"meta_mid_s{seed}_o{offset}.json").exists(), "lot (seed, offset) deja commence"
    deja = {int(m.group(1)) for f in os.listdir(out) if (m := re.match(r"ex_(\d+)_00\.npz$", f))}
    chevauche = sorted(deja & set(range(k0, k0 + n)))
    assert not chevauche, f"ex_{chevauche[0]:05d}_00.npz existe deja dans [k0, k0 + n) : choisir k0 >= {max(deja) + 1}"
    if pool is None:
        from .pool import build_pool
        pool = build_pool(("fivek", "raise", "ibl"), with_exr=False)
    T = table_sources(pool)
    defs_b = BUCKETS_MS if meme_scene else BUCKETS
    parts_b = {b: v[2] for b, v in defs_b.items()}
    cles, q = quotas(n, parts_b, PARTS_REGIME)
    fen = {f"{b}|{g_}": FENETRE_LARGE for b, g_ in cles}
    if not meme_scene:
        lu = {k: tuple(v) for k, v in json.load(open(fenetres or FENETRES_DEFAUT)).items()}
        assert set(lu) == set(fen), (sorted(lu), sorted(fen))
        fen.update(lu)
    bias_dict = BIAS_PRESETS[bias]
    polar = fresnel == "polar"

    ns = len(T["i"])
    usages = mp.Array("i", ns, lock=False)
    quota = mp.Array("i", [int(x) for x in q], lock=False)
    compteur = mp.Value("i", 0, lock=False)
    verrou = mp.Lock()
    t_debut = time.time()
    by_kind = {k: np.flatnonzero(T["kind"] == k) for k in np.unique(T["kind"])}
    meta = dict(n=n, seed=seed, offset=offset, k0=k0, workers=workers, cap=cap, meme_scene=meme_scene,
                fenetres=fen, bias=bias, bias_dict=bias_dict, fresnel=fresnel, wb=wb, wb_max_mired=wb_max_mired,
                cles=[list(c) for c in cles], quotas=[int(x) for x in q], n_sources=ns,
                sources_par_sorte={k: int(len(v)) for k, v in by_kind.items()},
                debut=time.strftime("%Y-%m-%d %H:%M:%S"), generateur="generating_sim_data.mid (v3 : FOV du recadrage de r, sample_scene3)",
                camera_v3=dict(bande_fov=BANDE_FOV, n_ferme=N_FERME, n_ouvert=N_OUVERT, p_ouvert=P_OUVERT,
                               p_proche=P_PROCHE, s_proche=S_PROCHE, k_map=K_MAP, sensor_w=SENSOR_W_c))
    (out / f"meta_mid_s{seed}_o{offset}.json").write_text(json.dumps(meta, indent=1))
    print(f"[mid] {ns} sources {meta['sources_par_sorte']} ; quotas {dict(zip(map(str, cles), q.tolist()))}",
          flush=True)

    def tirer_source(rng, sortes, pref, U, exclus_pano=None, base_logl=None, fenetre=None, meme_pano=None,
                     exclus_nom=None, vue_t=None):
        idx = np.concatenate([by_kind[s] for s in sortes if s in by_kind])
        w = (cap - U[idx]).astype(np.float64)
        if pref:
            w *= np.array([pref.get(k, 1.0) for k in T["kind"][idx]])
        ok = w > 0
        if exclus_pano is not None:
            ok &= T["pano"][idx] != exclus_pano
        if meme_pano is not None:
            ok &= (T["pano"][idx] == meme_pano) & (T["nom"][idx] != exclus_nom)
        if vue_t is not None:
            dv = np.abs(T["vue"][idx] - vue_t)
            ok &= np.minimum(dv, 6 - dv) >= 2
        if fenetre is not None:
            dl = T["logl"][idx] - base_logl
            ok &= (dl >= fenetre[0]) & (dl <= fenetre[1])
        if not ok.any():
            return None
        w = np.where(ok, w, 0.0)
        return int(idx[rng.choice(len(idx), p=w / w.sum())])

    def worker(w):
        cv2.setNumThreads(1)
        U = np.frombuffer(usages, dtype=np.int32)
        Q = np.frombuffer(quota, dtype=np.int32)
        rows, raisons = [], Counter()
        n_ = 0
        t_flush, n_flush = time.time(), 0

        def flush():
            if rows:
                fn = out / f"manifeste_s{seed}_o{offset}_w{w:02d}.parquet"
                pd.DataFrame(rows).to_parquet(str(fn) + ".tmp.parquet")
                os.replace(str(fn) + ".tmp.parquet", fn)
            (out / f"raisons_s{seed}_o{offset}_w{w:02d}.json").write_text(json.dumps(dict(raisons)))

        signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
        parent = os.getppid()

        def une(n_, rem):
            rs_ = np.random.default_rng([seed, offset, w, n_, 0])
            c = int(rs_.choice(len(cles), p=rem / rem.sum()))
            b, cible = cles[c]
            st_k, sr_k, _ = defs_b[b]
            it = tirer_source(rs_, st_k, PREF_T.get(b), U)
            if it is None:
                raisons["pas_de_t"] += 1
                return
            moitie_t = moitie_r = ""
            if b == "ms_photo":
                ir = it
            elif b == "ms_ibl":
                ir = tirer_source(rs_, sr_k, None, U, meme_pano=T["pano"][it], exclus_nom=T["nom"][it],
                                  base_logl=T["logl"][it], fenetre=fen[f"{b}|{cible}"], vue_t=T["vue"][it])
            else:
                ir = tirer_source(rs_, sr_k, None, U, exclus_pano=T["pano"][it], base_logl=T["logl"][it],
                                  fenetre=fen[f"{b}|{cible}"])
            if ir is None:
                raisons["pas_de_r"] += 1
                return
            st, sr = pool[int(T["i"][it])], pool[int(T["i"][ir])]
            if b == "ms_photo":
                moitie_t, moitie_r = ("g", "d") if rs_.random() < 0.5 else ("d", "g")
                st, sr = moitie(st, moitie_t), moitie(sr, moitie_r)
            tir = tirer(st, sr, np.random.default_rng([seed, offset, w, n_, 1]), bias_dict=bias_dict, polar=polar)
            if tir is None:
                raisons["r_hors_bande"] += 1
                return
            s = precoce(tir)
            alpha, k_cible, info = choisir_alpha_lu(s, cible, rs_)
            if alpha is None:
                raisons["precoce:" + k_cible] += 1
                return
            ex = composer(tir, alpha=alpha, wb=wb)
            p = ex["params"]
            if wb == "mix" and p["wb_dmired"] > wb_max_mired:
                raisons["balance_blancs"] += 1
                return
            f = finaux(ex["t"], ex["r"])
            d = measure(ex)
            v = verdict(f, d)
            reg = regime(f) if v == "ok" else None
            if v != "ok":
                raisons[v] += 1
                return
            cf = cles.index((b, reg))
            if Q[cf] <= 0:
                raisons["plein"] += 1
                return
            row = dict(seed=seed, offset=offset, w=w, n=n_, bucket=b, cible=cible, regime=reg,
                       moitie_t=moitie_t, moitie_r=moitie_r, k_cible=k_cible, alpha=alpha,
                       t_idx=int(T["i"][it]), r_idx=int(T["i"][ir]), t_name=str(T["nom"][it]),
                       r_name=str(T["nom"][ir]), t_kind=str(T["kind"][it]), r_kind=str(T["kind"][ir]),
                       dlogL=float(T["logl"][ir] - T["logl"][it]), defocus=float(p["defocus_px"]),
                       ghost=float(np.asarray(p["ghost_px"]).mean()), R_mean=float(p["R_mean"]),
                       exposure=float(p["exposure"]), double_pane=bool(p["double_pane"]), f_px=float(p["f_px"]),
                       theta0=float(p["theta0"]), **{k: float(p[k]) for k in WB_KEYS}, **f,
                       **{x: float(d[x]) for x in ("ssim", "ssim_std", "Ym", "m_mean", "t_std", "r_std",
                                                    "ratio")},
                       k1=info.get("k1", -1), rl1=info.get("rl1", np.nan))
            tmp = out / f"tmp_w{w:02d}_n{n_}.npz.tmp"
            scal = {f"p_{k}": np.asarray(v_) for k, v_ in p.items() if k not in ("R_map", "ghost_px")}
            with open(tmp, "wb") as fh:
                ecrit_octets({"m": ex["m"].astype(np.float16), "true_t": ex["true_t"].astype(np.float16),
                              "R_map": np.asarray(p["R_map"]).astype(np.float16),
                              "ghost_px": np.asarray(p["ghost_px"]).astype(np.float16), **scal,
                              "t_name": row["t_name"], "r_name": row["r_name"], "bucket": b, "regime": reg,
                              "member": 0}, fh)
            with verrou:
                plein = Q[cf] <= 0 or U[it] >= cap or U[ir] >= cap
                collision = False
                if not plein:
                    k = k0 + compteur.value
                    dest = out / f"ex_{k:05d}_00.npz"
                    collision = dest.exists()                  # never overwrite a file
                    if not collision:
                        os.replace(tmp, dest)
                        Q[cf] -= 1
                        U[it] += 1
                        U[ir] += 1
                        compteur.value += 1
            if plein or collision:
                os.remove(tmp)
                raisons["collision" if collision else "plein"] += 1
                return
            raisons["ok"] += 1
            row["k"] = k
            rows.append(row)

        try:
            while time.time() - t_debut < duree_max:
                if os.getppid() != parent:                   # parent gone: stop (no orphan worker)
                    break
                rem = Q.astype(np.float64).clip(0)
                if rem.sum() <= 0:
                    break
                n_ += 1
                try:
                    une(n_, rem)
                except Exception as e:                       # an attempt in error does not kill the worker
                    raisons["exception:" + type(e).__name__] += 1
                if time.time() - t_flush > 60 or len(rows) - n_flush >= 20:
                    flush()
                    t_flush, n_flush = time.time(), len(rows)
        finally:
            flush()

    procs = [mp.get_context("fork").Process(target=worker, args=(w,)) for w in range(workers)]
    for p_ in procs:
        p_.start()

    def arreter(*_):                                     # SIGTERM to the parent stops the workers too
        for q_ in procs:
            if q_.is_alive():
                q_.terminate()
    signal.signal(signal.SIGTERM, arreter)
    t_aff = time.time()
    while any(p_.is_alive() for p_ in procs):
        time.sleep(5)
        if time.time() - t_aff > 300:
            Q = np.frombuffer(quota, dtype=np.int32)
            dt = (time.time() - t_debut) / 3600
            print(f"[mid] {time.strftime('%H:%M')} {compteur.value} acceptes en {dt:.2f} h "
                  f"({compteur.value / max(dt, 1e-9):.0f}/h) ; quota restant {int(Q.clip(0).sum())}", flush=True)
            t_aff = time.time()
    codes = [p_.exitcode for p_ in procs]
    npz = glob.glob(str(out / "ex_*.npz"))
    tmpr = glob.glob(str(out / "tmp_*"))
    raisons = Counter()
    for f in glob.glob(str(out / f"raisons_s{seed}_o{offset}_w*.json")):
        raisons.update(json.load(open(f)))
    meta.update(fin=time.strftime("%Y-%m-%d %H:%M:%S"), elapsed_s=round(time.time() - t_debut, 1),
                acceptes=int(compteur.value), raisons=dict(raisons), codes=[int(c) for c in codes if c is not None])
    (out / f"meta_mid_s{seed}_o{offset}.json").write_text(json.dumps(meta, indent=1))
    print(f"[mid] fini : {compteur.value} acceptes en {(time.time() - t_debut) / 60:.1f} min ; codes {Counter(codes)} ; "
          f"{len(npz)} ex_*.npz, {len(tmpr)} temporaires ; raisons {dict(raisons.most_common(8))}", flush=True)
    return 0 if not any(codes) and compteur.value == n else 1
