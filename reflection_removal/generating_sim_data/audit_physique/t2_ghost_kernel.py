"""Noyau du fantome : modele 2 copies du code vs serie exacte."""
import sys, numpy as np
sys.path.insert(0, "/home/damien/ProjectsDL/reflection_removal")
import ref_physics as RP

n = 1.52

def code_kernel(theta, h, gap, double):
    """Les deux copies du code (scene._geometry, polar=True) : poids et marches."""
    Rs, Rp = RP.fresnel_R(1.0, n, theta)
    k = 4 if double else 2
    tot = (lambda R: 4*R/(1+3*R)) if double else (lambda R: 2*R/(1+R))
    fs, fp = tot(Rs), tot(Rp)
    a_s, a_p = (1-Rs)**k, (1-Rp)**k
    c1 = 0.5*(fs/(1+a_s) + fp/(1+a_p))
    c2 = 0.5*(fs*a_s/(1+a_s) + fp*a_p/(1+a_p))
    tht = np.arcsin(np.sin(theta)/n)
    path = h*np.tan(tht) + (gap*np.tan(theta) if double else 0.0)
    return np.array([c1, c2]), np.array([0.0, 2*path])

def moments(w, s):
    W = w.sum()
    m1 = (w*s).sum()/W
    m2 = np.sqrt(max((w*(s-m1)**2).sum()/W, 0.0))
    return W, m1, m2

for (h, gap, double, titre) in [(6e-3, 0.0, False, "LAME SIMPLE h=6mm"),
                                (6e-3, 12e-3, True, "DOUBLE VITRAGE h=6mm gap=12mm"),
                                (20e-3, 6e-3, True, "DOUBLE VITRAGE h=20mm gap=6mm (verre epais, lame mince)")]:
    print(f"\n{'='*94}\n{titre}\n{'='*94}")
    for td in (30, 45, 60):
        t = np.radians(td)
        we, se, fuite = RP.reflect_kernel(t, n, h, gap, double)
        wc, sc = code_kernel(t, h, gap, double)
        # copies significatives seulement
        keep = we > 1e-4*we.sum()
        print(f"\n theta = {td} deg   (fuite de la serie : {fuite:.1e})")
        print(f"   exact : {len(we)} copies, {keep.sum()} au-dessus de 1e-4")
        print("     " + "  ".join(f"{s*1e3:6.2f}mm:{w/we.sum():5.1%}"
                                  for s, w in zip(se[keep], we[keep])))
        print(f"   code  :   0.00mm:{wc[0]/wc.sum():5.1%}   {sc[1]*1e3:6.2f}mm:{wc[1]/wc.sum():5.1%}")
        We, m1e, m2e = moments(we, se)
        Wc, m1c, m2c = moments(wc, sc)
        print(f"   poids total  exact {We:.5f}   code {Wc:.5f}   ecart {Wc/We-1:+.2e}")
        print(f"   barycentre   exact {m1e*1e3:7.3f} mm   code {m1c*1e3:7.3f} mm   ecart {m1c/m1e-1:+7.2%}")
        print(f"   etalement    exact {m2e*1e3:7.3f} mm   code {m2c*1e3:7.3f} mm   ecart {m2c/m2e-1:+7.2%}")
