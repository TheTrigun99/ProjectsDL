"""Defocus, transmission, et effets non modelises."""
import sys, numpy as np
sys.path.insert(0, "/home/damien/ProjectsDL/reflection_removal")
import ref_physics as RP
from generating_sim_data.scene import _geometry
from generating_sim_data.config import GLASS_IOR as n, SENSOR_W

print("=== A. Cercle de confusion : code vs optique geometrique exacte ===")
print("   (lentille mince, diametre du flou sur le plan image conjugue de S1)")
print(f"   {'f(mm)':>6} {'N':>5} {'S1(m)':>7} {'S2(m)':>7} | {'exact(px)':>10} {'code(px)':>10} {'ecart':>9}")
SIZE = 384
for fov, N, S1, S2 in [(40, 1.4, 0.5, 50.0), (40, 8.0, 0.5, 1.0),
                       (60, 2.8, 1.0, 5.0), (90, 4.0, 2.0, 0.5),
                       (40, 1.4, 7.9, 8.0)]:
    f_px = (SIZE/2)/np.tan(np.radians(fov)/2)
    f = f_px/SIZE*SENSOR_W
    v1 = f*S1/(S1-f); v2 = f*S2/(S2-f)
    A = f/N
    coc_exact = A*abs(v2-v1)/v2                      # triangles semblables
    s = _geometry(SIZE, fov, 0.0, 0.3, N, S1, S2, 6e-3, 0.0, False)
    print(f"   {f*1e3:6.2f} {N:5.1f} {S1:7.2f} {S2:7.2f} | "
          f"{coc_exact/SENSOR_W*SIZE:10.4f} {s['defocus_px']:10.4f} "
          f"{s['defocus_px']/(coc_exact/SENSOR_W*SIZE)-1:+9.2e}")

print("\n=== B. Transmission : 1 - R_tot est-il exact ? ===")
print("   somme des poids emergents vers l'avant, serie incoherente")
for dbl in (False, True):
    for td in (0, 45, 70):
        t = np.radians(td)
        w, s, fuite = RP.reflect_kernel(t, n, 6e-3, 12e-3, dbl)
        print(f"   {'IGU ' if dbl else 'lame'} theta={td:>3} : R_tot={w.sum():.8f}  "
              f"1-R_tot={1-w.sum():.8f}  (verre sans absorption -> T = 1-R exact)")

print("\n=== C. Fantome de la TRANSMISSION (non modelise par le code) ===")
print("   poids relatif de la 2e copie transmise = R^2 environ")
for td in (0, 45, 70):
    t = np.radians(td)
    Rs, Rp = RP.fresnel_R(1.0, n, t)
    Rm = 0.5*(Rs+Rp)
    print(f"   theta={td:>3} : R={Rm:.4f}  copie transmise secondaire / principale "
          f"= R^2 = {Rm**2:.2e}  -> negligeable")

print("\n=== D. Deplacement de la TRANSMISSION par la lame (non modelise) ===")
print("   lame a faces paralleles : le rayon transmis est decale lateralement")
print(f"   {'theta':>6} {'decal(mm)':>10} {'px a d_focus=1m, fov40':>24}")
f_px = (SIZE/2)/np.tan(np.radians(40)/2)
for td in (10, 30, 45, 60, 80):
    t = np.radians(td); h = 6e-3
    tt = np.arcsin(np.sin(t)/n)
    d = h*np.sin(t-tt)/np.cos(tt)        # decalage perpendiculaire classique
    print(f"   {td:>6} {d*1e3:10.4f} {f_px*d/1.0:24.3f}")
print("   -> variation sur le champ << 1 px : distorsion geometrique negligeable.")

print("\n=== E. Defocus du fantome vs copie principale (non modelise) ===")
print("   la copie arriere parcourt 2h.n de chemin optique en plus")
for h in (6e-3, 20e-3):
    for dref in (0.5, 5.0):
        dz = 2*h/n                       # recul apparent de l'image virtuelle
        print(f"   h={h*1e3:.0f}mm d_refl={dref:4.1f}m : recul de l'image "
              f"fantome ~{dz*1e3:.1f}mm = {dz/dref:.2%} de d_refl "
              f"-> variation du CoC du meme ordre")
