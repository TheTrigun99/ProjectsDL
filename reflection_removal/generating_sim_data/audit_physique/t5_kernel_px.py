"""Combien de copies compte-t-on, et quel poids perd le modele 2 copies ?"""
import sys, numpy as np
sys.path.insert(0, "/home/damien/ProjectsDL/reflection_removal")
import ref_physics as RP
n = 1.52
for dbl in (False, True):
    cos_g, keys, W = RP.kernel_table(n, dbl)
    print(f"\n{'DOUBLE VITRAGE' if dbl else 'LAME SIMPLE'} : cles (k_verre, k_air) retenues = {keys}")
    print(f"  {'theta':>6} " + " ".join(f"{str(k):>10}" for k in keys) + f" {'somme':>9}")
    for td in (0, 30, 45, 60, 80):
        i = np.argmin(np.abs(cos_g - np.cos(np.radians(td))))
        w = W[i]; s = w.sum()
        print(f"  {td:>6} " + " ".join(f"{x/s:10.2%}" for x in w) + f" {s:9.5f}")
