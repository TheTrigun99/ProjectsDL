"""Deplacement image du fantome : code vs parallaxe exacte (Snell + projection)."""
import sys, numpy as np
sys.path.insert(0, "/home/damien/ProjectsDL/reflection_removal")
import ref_physics as RP
from generating_sim_data.scene import _geometry
from generating_sim_data.config import GLASS_IOR as n

SIZE = 96

def th0max(fov, phi):
    half = np.arctan(np.tan(np.radians(fov)/2)*(abs(np.cos(phi))+abs(np.sin(phi))))
    return np.radians(83.0) - half

def code_offset(size, fov, phi, th0, h, gap, dbl, dref):
    s = _geometry(size, fov, phi, th0, 4.0, 1.0, dref, h, gap, dbl)
    return s["ghost_px"]*s["ghost_dir"][...,0], s["ghost_px"]*s["ghost_dir"][...,1]

def exact_offset(size, fov, phi, th0, h, gap, dbl, dref, plane="depth"):
    f = (size/2)/np.tan(np.radians(fov)/2)
    ys, xs = np.mgrid[0:size, 0:size].astype(float) - (size-1)/2
    nrm = np.array([np.sin(th0)*np.cos(phi), np.sin(th0)*np.sin(phi), np.cos(th0)])
    dx, dy, _ = RP.ghost_offset_exact(xs, ys, f, nrm, n, h, gap, dbl, dref, plane)
    return dx, dy

def cmp(fov, phi, th0, h, gap, dbl, dref=1.0, plane="depth"):
    cx, cy = code_offset(SIZE, fov, phi, th0, h, gap, dbl, dref)
    ex, ey = exact_offset(SIZE, fov, phi, th0, h, gap, dbl, dref, plane)
    rc, re = np.hypot(cx, cy), np.hypot(ex, ey)
    ok = re > 1e-6
    ratio = rc[ok]/re[ok]
    ang = np.degrees(np.arccos(np.clip((cx*ex+cy*ey)[ok]/(rc[ok]*re[ok]), -1, 1)))
    err = np.hypot(cx-ex, cy-ey)                     # erreur vectorielle, px
    return ratio, ang, err, re

print(f"{'cas':<30} {'|code|/|ref|':>13} {'p10..p90':>15} "
      f"{'ang med':>8} {'ang max':>8} | {'|err| med px':>12} {'|err| max px':>12} {'|ref| med px':>12}")
print("-"*126)
for fov in (40, 60, 90):
    for frac in (0.25, 0.6, 0.95):
        phi = 0.0
        th0 = frac*th0max(fov, phi)
        for dbl, h, gap in ((False, 6e-3, 0.0), (True, 6e-3, 12e-3)):
            r, a, e, re = cmp(fov, phi, th0, h, gap, dbl)
            tag = "IGU " if dbl else "lame"
            print(f"fov{fov:<3} th0={np.degrees(th0):5.1f}deg {tag:<5}        "
                  f"{np.median(r):13.4f} {np.percentile(r,10):6.3f}..{np.percentile(r,90):<7.3f} "
                  f"{np.median(a):8.2f} {a.max():8.2f} | {np.median(e):12.3f} {e.max():12.3f} {np.median(re):12.3f}")
    print()

print("--- phi = 45 deg (vitre inclinee en diagonale) ---")
for fov in (40, 60):
    phi = np.pi/4
    th0 = 0.8*th0max(fov, phi)
    r, a, e, re = cmp(fov, phi, th0, 6e-3, 0.0, False)
    print(f"fov{fov:<3} th0={np.degrees(th0):5.1f}  ratio med {np.median(r):.4f} "
          f"[{np.percentile(r,10):.3f},{np.percentile(r,90):.3f}]  ang med {np.median(a):.2f} max {a.max():.2f}")

print("\n--- au centre du champ exactement (doit coincider) ---")
for th0d in (5, 30, 45, 60, 75):
    cx, cy = code_offset(3, 40, 0.0, np.radians(th0d), 6e-3, 0.0, False, 1.0)
    ex, ey = exact_offset(3, 40, 0.0, np.radians(th0d), 6e-3, 0.0, False, 1.0)
    print(f"  th0={th0d:>3}  code=({cx[1,1]:+9.6f},{cy[1,1]:+9.6f})  "
          f"exact=({ex[1,1]:+9.6f},{ey[1,1]:+9.6f})  ratio={np.hypot(cx[1,1],cy[1,1])/np.hypot(ex[1,1],ey[1,1]):.6f}")

print("\n--- scene a distance constante (sphere) au lieu d'un plan ---")
for fov in (40, 90):
    th0 = 0.6*th0max(fov, 0.0)
    r1,_,_,_ = cmp(fov, 0.0, th0, 6e-3, 0.0, False, 1.0, "depth")
    r2,_,_,_ = cmp(fov, 0.0, th0, 6e-3, 0.0, False, 1.0, "sphere")
    print(f"  fov{fov}: ratio med plan={np.median(r1):.4f}  sphere={np.median(r2):.4f}")
