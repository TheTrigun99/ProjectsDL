"""Auto-controle de la reference : elle doit retrouver les resultats connus."""
import numpy as np, ref_physics as RP

n = 1.52
print("=== A. Fresnel : incidence normale, Brewster, rasance ===")
Rs, Rp = RP.fresnel_R(1.0, n, 0.0)
print(f"  theta=0   R_s={Rs:.6f} R_p={Rp:.6f}   attendu ((n-1)/(n+1))^2={( (n-1)/(n+1) )**2:.6f}")
thB = np.arctan(n)
Rs, Rp = RP.fresnel_R(1.0, n, thB)
print(f"  Brewster {np.degrees(thB):.2f} deg  R_p={Rp:.2e} (doit etre 0)  R_s={Rs:.4f}")
Rs, Rp = RP.fresnel_R(1.0, n, np.radians(89.9))
print(f"  theta=89.9 R_s={Rs:.4f} R_p={Rp:.4f} (-> 1)")

print("\n=== B. Reflectance totale de l'empilement vs formules fermees ===")
print(f"  {'theta':>6} | {'lame exacte':>12} {'2R/(1+R)':>10} {'ecart':>9} | "
      f"{'IGU exacte':>11} {'4R/(1+3R)':>10} {'ecart':>9}")
for td in [0, 20, 40, 45, 60, 70, 80]:
    th = np.radians(td)
    for pol in (0,):   # verifie par polarisation (s)
        R1 = RP.fresnel_R(1.0, n, th)[pol]
        w2, _, f2x = RP.stack_copies(th, *RP.layers(n, 6e-3, 0.0, False), pol=pol)
        w4, _, f4x = RP.stack_copies(th, *RP.layers(n, 6e-3, 12e-3, True), pol=pol)
        f2, f4 = 2*R1/(1+R1), 4*R1/(1+3*R1)
        print(f"  {td:>6} | {w2.sum():12.8f} {f2:10.8f} {w2.sum()-f2:9.1e} | "
              f"{w4.sum():11.8f} {f4:10.8f} {w4.sum()-f4:9.1e}")

print("\n=== C. Marche laterale : trace de Snell vs 2h.tan(theta_t) ===")
print(f"  {'theta':>6} | {'trace (mm)':>11} {'2h tan_t (mm)':>13} | "
      f"{'perp trace':>11} {'walk.cos_i':>11}")
h = 6e-3
for td in [5, 20, 45, 60, 80]:
    th = np.radians(td)
    nrm = np.array([0.0, 0.0, -1.0])            # vitre perpendiculaire a l'axe
    v = np.array([[np.sin(th), 0.0, np.cos(th)]])
    wv, dout = RP.trace_walk(v, nrm, n, h, 0.0, False)
    tht = np.arcsin(np.sin(th)/n)
    ana = 2*h*np.tan(tht)
    vh = v/np.linalg.norm(v)
    perp = np.linalg.norm(wv - (wv*vh).sum()*vh)
    print(f"  {td:>6} | {np.linalg.norm(wv)*1e3:11.6f} {ana*1e3:13.6f} | "
          f"{perp*1e3:11.6f} {np.linalg.norm(wv)*np.cos(th)*1e3:11.6f}")

print("\n=== D. IGU : marche de la 1re copie du 2e panneau ===")
h, g = 6e-3, 12e-3
for td in [20, 45, 70]:
    th = np.radians(td)
    v = np.array([[np.sin(th), 0.0, np.cos(th)]])
    wv, _ = RP.trace_walk(v, np.array([0.,0.,1.]), n, h, g, True)
    tht = np.arcsin(np.sin(th)/n)
    ana = 2*(h*np.tan(tht) + g*np.tan(th))
    print(f"  theta={td:>3}  trace={np.linalg.norm(wv)*1e3:9.5f} mm   "
          f"2(h tan_t + g tan_i)={ana*1e3:9.5f} mm   ecart={abs(np.linalg.norm(wv)-ana)*1e3:.2e}")
