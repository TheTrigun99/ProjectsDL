"""Verifie une forme fermee corrigee du fantome, et le noyau 4 copies de l'IGU."""
import sys, numpy as np
sys.path.insert(0, "/home/damien/ProjectsDL/reflection_removal")
import ref_physics as RP
from generating_sim_data.config import GLASS_IOR as NG

SIZE = 128

def propose(size, fov, phi, th0, h, gap, dbl, d_refl):
    """Formule proposee : parallaxe complete, en coordonnees pixel.

    off = f_px * walk/d_refl * (t - x*t_z/f_px) / |t|,  t = v - (v.n) n
    """
    f = (size/2)/np.tan(np.radians(fov)/2)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64) - (size-1)/2
    nx, ny, nz = (np.sin(th0)*np.cos(phi), np.sin(th0)*np.sin(phi), np.cos(th0))
    p = xs*nx + ys*ny + f*nz
    tx, ty, tz = xs - p*nx, ys - p*ny, f - p*nz
    nt = np.sqrt(tx*tx + ty*ty + tz*tz)
    cos_i = np.abs(p)/np.sqrt(xs*xs + ys*ys + f*f)
    sin_t = np.sqrt(np.maximum(1-cos_i**2, 0))/NG
    tan_t = sin_t/np.sqrt(np.maximum(1-sin_t**2, 1e-12))
    tan_i = np.sqrt(np.maximum(1-cos_i**2, 0))/np.maximum(cos_i, 1e-8)
    walk = 2*(h*tan_t + (gap*tan_i if dbl else 0.0))
    qx = tx - xs*tz/f
    qy = ty - ys*tz/f
    amp = f*walk/(d_refl*np.maximum(nt, 1e-8))
    return amp*qx, amp*qy

def exact(size, fov, phi, th0, h, gap, dbl, d_refl):
    f = (size/2)/np.tan(np.radians(fov)/2)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64) - (size-1)/2
    nv = np.array([np.sin(th0)*np.cos(phi), np.sin(th0)*np.sin(phi), np.cos(th0)])
    dx, dy, _ = RP.ghost_offset_exact(xs, ys, f, nv, NG, h, gap, dbl, d_refl)
    return dx, dy

print("A. Forme fermee proposee vs trace de Snell + projection exacte")
print(f"   {'cas':<34} {'ecart rel med':>14} {'ecart rel max':>14} {'|ref| med px':>13}")
worst = 0.0
for fov in (40, 60, 90):
    for frac in (0.05, 0.4, 0.75, 0.98):
        for phi in (0.0, np.pi/4, 2.1):
            half = np.arctan(np.tan(np.radians(fov)/2)*(abs(np.cos(phi))+abs(np.sin(phi))))
            th0 = frac*(np.radians(83.0)-half)
            for dbl, h, gap in ((False, 7e-3, 0.0), (True, 14e-3, 8e-3)):
                px, py = propose(SIZE, fov, phi, th0, h, gap, dbl, 2.0)
                ex, ey = exact(SIZE, fov, phi, th0, h, gap, dbl, 2.0)
                nr = np.hypot(ex, ey); ok = nr > 1e-9
                e = np.hypot(px-ex, py-ey)[ok]/nr[ok]
                worst = max(worst, e.max())
                if frac in (0.4, 0.98) and phi == 0.0:
                    tag = "IGU" if dbl else "lame"
                    print(f"   fov{fov} th0={np.degrees(th0):5.1f} {tag:<4}"
                          f"{'':<14} {np.median(e):14.2e} {e.max():14.2e} {np.median(nr):13.3f}")
print(f"   ecart relatif max sur les 72 configurations : {worst:.2e}\n")

print("B. Noyau IGU : 2 copies (code) vs 4 copies geometriques, contre la serie exacte")
print(f"   {'theta':>6} {'copie':>8} {'exact':>9} {'4 copies':>9} {'2 copies':>9}")
for td in (20, 45, 60, 75):
    t = np.radians(td)
    h, gap = 14e-3, 8e-3
    we, se, _ = RP.reflect_kernel(t, NG, h, gap, True)
    tot = we.sum()
    # 4 copies : poids ~ A^j par polarisation, A = (1-R)^2, marches geometriques
    out4 = np.zeros(4)
    for pol in (0, 1):
        R = RP.fresnel_R(1.0, NG, t)[pol]
        A = (1-R)**2
        w = np.array([1.0, A, A**2, A**3])
        out4 += 0.5*w/w.sum()*(4*R/(1+3*R))
    tht = np.arcsin(np.sin(t)/NG)
    wg, wa = 2*h*np.tan(tht), 2*gap*np.tan(t)
    s4 = np.array([0.0, wg, wg+wa, 2*wg+wa])
    # 2 copies du code
    out2 = np.zeros(2)
    for pol in (0, 1):
        R = RP.fresnel_R(1.0, NG, t)[pol]
        a = (1-R)**4; f = 4*R/(1+3*R)
        out2 += 0.5*np.array([f/(1+a), f*a/(1+a)])
    s2 = np.array([0.0, wg/2+wa/2 + wg/2])   # = wg + wa ... (centre a centre)
    s2 = np.array([0.0, (h*np.tan(tht)+gap*np.tan(t))*2])
    print(f"   {td:>6}", end="")
    for j, s in enumerate(s4):
        k = np.argmin(np.abs(se - s))
        got = we[k] if abs(se[k]-s) < 1e-9 else np.nan
        print(f"{'' if j==0 else '         '}{s*1e3:8.2f}mm {got/tot:8.2%} {out4[j]/tot:8.2%} "
              f"{(out2[0]/tot if j==0 else (out2[1]/tot if abs(s-s2[1])<1e-9 else 0.0)):8.2%}")
    print(f"   {'':>6} {'somme':>8} {1.0:8.2%} {out4.sum()/tot:8.2%} {out2.sum()/tot:8.2%}")
