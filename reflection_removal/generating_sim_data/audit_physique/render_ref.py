"""Rendu de la couche de reflet : code vs reference, avec variantes d'attribution."""
import sys, numpy as np, cv2
sys.path.insert(0, "/home/damien/ProjectsDL/reflection_removal")
import ref_physics as RP
from generating_sim_data.scene import (_geometry, defocus_continuous,
                                       double_reflection)
from generating_sim_data.config import GLASS_IOR as NG, SENSOR_W, BIAS_PRESETS
from generating_sim_data.scene import tilt

TAB = {d: RP.kernel_table(NG, d) for d in (False, True)}

def draw_scalars(rng, B):
    """Reproduit EXACTEMENT l'ordre des tirages de sample_scene2."""
    T = lambda lo, hi, k, tw="lo": tilt(rng, lo, hi, B[k] if B else 1.0, tw)
    fov = float(T(40, 90, "fov"))
    phi = float(rng.uniform(0, 2*np.pi))
    half = np.arctan(np.tan(np.radians(fov)/2)*(abs(np.cos(phi))+abs(np.sin(phi))))
    th0 = float(T(0, np.radians(83.0)-half, "theta0", "hi"))
    N = float(T(1.4, 8.0, "N"))
    d_focus = 10.0**float(T(-0.3, 0.9, "d_focus"))
    d_hi = max(0.4, min(3.0, d_focus))
    d_glass = float(np.exp(T(np.log(0.3), np.log(d_hi), "d_glass")))
    d_refl = d_glass + 10.0**float(T(-1.0, 1.7, "d_refl"))
    h = float(T(0.003, 0.020, "h_glass", "hi"))
    gap = float(T(0.006, 0.020, "gap", "hi"))
    dbl = rng.random() < (B["p_double"] if B else 0.5)
    return dict(fov=fov, phi=phi, theta0=th0, N=N, d_focus=d_focus,
                d_glass=d_glass, d_refl=d_refl, h_glass=h, gap=gap,
                double_pane=bool(dbl))

def maps(size, p):
    """Cartes geometriques de reference : cos_i, t_hat, offset unitaire, copies."""
    f = (size/2)/np.tan(np.radians(p["fov"])/2)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64) - (size-1)/2
    th0, phi = p["theta0"], p["phi"]
    nv = np.array([np.sin(th0)*np.cos(phi), np.sin(th0)*np.sin(phi), np.cos(th0)])
    v = np.stack([xs, ys, np.full_like(xs, f)], -1)
    pr = (v*nv).sum(-1, keepdims=True)
    t = v - pr*nv
    th = t/np.maximum(np.linalg.norm(t, axis=-1, keepdims=True), 1e-30)
    cos_i = np.clip(np.abs(pr[...,0])/np.linalg.norm(v, axis=-1), 1e-6, 1.0)
    # offset d'echantillonnage par metre de marche (parallaxe complete)
    Z = p["d_refl"]
    U = (f/Z)*np.stack([th[...,0] - xs*th[...,2]/f, th[...,1] - ys*th[...,2]/f], -1)
    tan_i = np.sqrt(np.maximum(1-cos_i**2, 0))/cos_i
    tan_t = np.tan(np.arcsin(np.clip(np.sin(np.arccos(cos_i))/NG, -1, 1)))
    cos_g, keys, W = TAB[p["double_pane"]]
    Wi = np.stack([np.interp(cos_i, cos_g, W[:, j]) for j in range(len(keys))], -1)
    walks = np.stack([kg*p["h_glass"]*tan_t + ka*p["gap"]*tan_i for kg, ka in keys], -1)
    return dict(f=f, cos_i=cos_i, U=U, keys=keys, W=Wi, walks=walks,
                xs=xs, ys=ys, size=size)

def warp(img, dx, dy, M):
    return cv2.remap(img, (M["xs"]+(M['size']-1)/2+dx).astype(np.float32),
                     (M["ys"]+(M['size']-1)/2+dy).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def ref_layer(r, p, M, geom="exact", kernel="exact", order="defocus_last",
              defocus_px=None):
    """Couche de reflet de reference, variantes isolables.

    geom   : "exact" parallaxe complete | "code" |D|*cos_i sur (tx,ty)
    kernel : "exact" toutes les copies | "code" 2 copies normalisees
    order  : "defocus_last" (physique) | "defocus_first" (le code)
    """
    W, walks, keys = M["W"], M["walks"], M["keys"]
    if kernel == "code":
        Rs, Rp = RP.fresnel_R(1.0, NG, np.arccos(M["cos_i"]))
        k = 4 if p["double_pane"] else 2
        tot = (lambda R: 4*R/(1+3*R)) if p["double_pane"] else (lambda R: 2*R/(1+R))
        fs, fp = tot(Rs), tot(Rp)
        a_s, a_p = (1-Rs)**k, (1-Rp)**k
        c1 = 0.5*(fs/(1+a_s) + fp/(1+a_p))
        c2 = 0.5*(fs*a_s/(1+a_s) + fp*a_p/(1+a_p))
        W = np.stack([c1, c2], -1)
        tan_i = np.sqrt(np.maximum(1-M["cos_i"]**2, 0))/M["cos_i"]
        tan_t = np.tan(np.arcsin(np.clip(np.sin(np.arccos(M["cos_i"]))/NG, -1, 1)))
        path = p["h_glass"]*tan_t + (p["gap"]*tan_i if p["double_pane"] else 0.0)
        walks = np.stack([np.zeros_like(path), 2*path], -1)
        keys = [(0,0), (1,1)]
    sig = (W.max(axis=(0,1)) > 1e-3*W.sum(-1).max())
    src = defocus_continuous(r, defocus_px) if order == "defocus_first" else r
    out = np.zeros_like(r)
    for j in np.where(sig)[0]:
        if geom == "exact":
            dx, dy = walks[...,j]*M["U"][...,0], walks[...,j]*M["U"][...,1]
        else:
            Dxy = np.stack([walks[...,j]*0, walks[...,j]*0], -1)  # place
            nxy = np.linalg.norm(M["U"], axis=-1)  # non utilise
            # formule du code : |D|*cos_i dans la direction (tx,ty)
            raise SystemExit("geom='code' passe par _geometry, voir cmp_code")
        if np.abs(np.stack([dx,dy])).max() < 1e-3 and j > 0 and walks[...,j].max() < 1e-9:
            w = W[...,j:j+1]; out += w*src; continue
        out += W[...,j:j+1]*warp(src, dx, dy, M)
    return defocus_continuous(out, defocus_px) if order == "defocus_last" else out

def code_layer(r, p, size):
    sc = _geometry(size, p["fov"], p["phi"], p["theta0"], p["N"], p["d_focus"],
                   p["d_refl"], p["h_glass"], p["gap"], p["double_pane"])
    x = defocus_continuous(r, sc["defocus_px"])
    x = double_reflection(x, sc)
    return x*sc["R_map"], sc
