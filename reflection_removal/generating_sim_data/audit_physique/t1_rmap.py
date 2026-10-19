"""R_map : code (polar / moyenne) vs serie incoherente exacte."""
import sys, numpy as np
sys.path.insert(0, "/home/damien/ProjectsDL/reflection_removal")
import ref_physics as RP
from generating_sim_data.scene import fresnel_single, fresnel_reflectance

n = 1.52
print("A. Fresnel une interface, code vs reference (max sur 0-83 deg)")
th = np.radians(np.linspace(0, 83, 2000))
code = fresnel_single(np.cos(th), n)
Rs, Rp = RP.fresnel_R(1.0, n, th)
ref = 0.5 * (Rs + Rp)
print(f"   ecart max |code - ref| = {np.abs(code-ref).max():.3e}   -> identique\n")

print("B. Reflectance totale de la vitre, lumiere naturelle")
print(f"   {'theta':>5} | {'exact':>9} {'polar':>9} {'moyenne':>9} "
      f"{'polar/ex-1':>11} {'moy/ex-1':>10} || "
      f"{'exact IGU':>9} {'polar':>9} {'moyenne':>9} {'polar/ex-1':>11} {'moy/ex-1':>10}")
rows = []
for td in [0, 10, 20, 30, 40, 45, 50, 60, 70, 80, 83]:
    t = np.radians(td)
    Rs, Rp = RP.fresnel_R(1.0, n, t)
    Rm = 0.5 * (Rs + Rp)
    line = [td]
    for dbl in (False, True):
        f = (lambda R: 4*R/(1+3*R)) if dbl else (lambda R: 2*R/(1+R))
        ex, _, fuite = RP.reflect_kernel(t, n, 6e-3, 12e-3, dbl)
        ex = ex.sum()
        pol = 0.5 * (f(Rs) + f(Rp))
        moy = f(Rm)
        line += [ex, pol, moy, pol/ex - 1, moy/ex - 1]
    rows.append(line)
    print(f"   {td:5d} | {line[1]:9.5f} {line[2]:9.5f} {line[3]:9.5f} "
          f"{line[4]:+10.2e} {line[5]:+9.2%} || "
          f"{line[6]:9.5f} {line[7]:9.5f} {line[8]:9.5f} {line[9]:+10.2e} {line[10]:+9.2%}")
print("\n   -> `polar` = la serie exacte (ecart = precision machine).")
print("   -> `moyenne` (v1.1) surestime, confirme.")
