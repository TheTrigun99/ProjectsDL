"""D'ou vient l'erreur de geometrie ? Decomposition terme par terme.

Offset d'echantillonnage exact (parallaxe, scene plane a la profondeur Z) :
    off = (f/Z) * (Dx - x*Dz/f , Dy - y*Dz/f)      D = walk * t_hat
Le code ecrit :
    off = (f/Z) * |D| * cos_i * (tx,ty)/|(tx,ty)|
On teste les variantes intermediaires pour isoler les deux approximations.
"""
import sys, numpy as np
sys.path.insert(0, "/home/damien/ProjectsDL/reflection_removal")
import ref_physics as RP
from generating_sim_data.scene import _geometry
from generating_sim_data.config import GLASS_IOR as n

SIZE, DREF = 96, 1.0

def variantes(fov, phi, th0, h, gap, dbl):
    f = (SIZE/2)/np.tan(np.radians(fov)/2)
    ys, xs = np.mgrid[0:SIZE, 0:SIZE].astype(float) - (SIZE-1)/2
    nv = np.array([np.sin(th0)*np.cos(phi), np.sin(th0)*np.sin(phi), np.cos(th0)])
    v = np.stack([xs, ys, np.full_like(xs, f)], -1)
    t = v - (v*nv).sum(-1, keepdims=True)*nv
    th = t/np.linalg.norm(t, axis=-1, keepdims=True)
    tht = np.arcsin(np.sin(th0)/n)   # pas utilise, la marche vient du trace
    walk, _ = RP.trace_walk(v, nv, n, h, gap, dbl)
    L = np.linalg.norm(walk, axis=-1, keepdims=True)
    cosi = np.abs((v*nv).sum(-1, keepdims=True))/np.linalg.norm(v, axis=-1, keepdims=True)
    D = L*th
    Z = DREF
    out = {}
    # 1. exact (trace + projection, non linearise)
    ex, ey, _ = RP.ghost_offset_exact(xs, ys, f, nv, n, h, gap, dbl, DREF)
    out["exact"] = np.stack([ex, ey], -1)
    # 2. parallaxe linearisee, terme radial inclus
    out["lin"] = (f/Z)*np.stack([D[...,0] - xs*D[...,2]/f, D[...,1] - ys*D[...,2]/f], -1)
    # 3. parallaxe sans le terme radial
    out["sans_radial"] = (f/Z)*D[...,:2]
    # 4. code : |D|*cos_i dans la direction (tx,ty)
    nxy = np.linalg.norm(D[...,:2], axis=-1, keepdims=True)
    out["code_formule"] = (f/Z)*L*cosi*D[...,:2]/np.maximum(nxy, 1e-30)
    s = _geometry(SIZE, fov, phi, th0, 4.0, 1.0, DREF, h, gap, dbl)
    out["code_reel"] = s["ghost_px"][...,None]*s["ghost_dir"]
    return out

def ecart(a, b):
    na, nb = np.linalg.norm(a,axis=-1), np.linalg.norm(b,axis=-1)
    ok = nb > 1e-9
    e = np.linalg.norm(a-b, axis=-1)[ok]/nb[ok]
    return np.median(e), e.max()

print(f"{'cas':<26} {'variante':<14} {'ecart rel / exact':>20}")
print("-"*64)
for fov, frac in ((40, 0.6), (40, 0.95), (90, 0.6)):
    phi = 0.0
    half = np.arctan(np.tan(np.radians(fov)/2)*(abs(np.cos(phi))+abs(np.sin(phi))))
    th0 = frac*(np.radians(83.0) - half)
    V = variantes(fov, phi, th0, 6e-3, 0.0, False)
    nom = f"fov{fov} th0={np.degrees(th0):.1f}deg"
    for k in ("lin", "sans_radial", "code_formule", "code_reel"):
        m, M = ecart(V[k], V["exact"])
        print(f"{nom:<26} {k:<14} med {m:7.2%}  max {M:7.2%}")
    print()
