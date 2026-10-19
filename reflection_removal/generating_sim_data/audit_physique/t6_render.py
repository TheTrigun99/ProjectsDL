import sys, numpy as np, cv2, glob
sys.path.insert(0, "/home/damien/ProjectsDL/reflection_removal")
from render_ref import *
from generating_sim_data.config import TAU

def srgb_dec(x):
    x = x.astype(np.float64)/255.0
    return np.where(x <= 0.04045, x/12.92, ((x+0.055)/1.055)**2.4)

ims = sorted(glob.glob("/home/damien/ProjectsDL/reflection_removal/real_photos_random/*"))
def load(p, size=384):
    a = cv2.imread(p)[..., ::-1]
    h, w = a.shape[:2]
    s = min(h, w)
    a = a[(h-s)//2:(h-s)//2+s, (w-s)//2:(w-s)//2+s]
    return srgb_dec(cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA))

SIZE = 384
R0, T0 = load(ims[1]), load(ims[2])
rng = np.random.default_rng(1009)
B = BIAS_PRESETS["defocus_high_v2"]

def psnr(a, b):
    e = np.mean((a-b)**2)
    return 99.0 if e <= 0 else 10*np.log10(1.0/e)

rows = []
for i in range(80):
    p = draw_scalars(rng, B)
    M = maps(SIZE, p)
    rc, sc = code_layer(R0, p, SIZE)
    dp = sc["defocus_px"]
    rD = ref_layer(R0, p, M, "exact", "exact", "defocus_last", dp)   # reference
    rB = ref_layer(R0, p, M, "exact", "code", "defocus_first", dp)   # geom seule
    rC = ref_layer(R0, p, M, "exact", "exact", "defocus_first", dp)  # + noyau
    tl = T0*(1.0 - sc["R_map"])
    e = TAU/max((tl+rD).mean(), 1e-8)
    mm = lambda rr: np.clip((tl+rr)*e, 0, 1)
    rows.append(dict(dbl=p["double_pane"], th0=np.degrees(p["theta0"]),
                     fov=p["fov"], dpx=dp, gmax=float(sc["ghost_px"].max()),
                     rel_AD=np.linalg.norm(rc-rD)/np.linalg.norm(rD),
                     rel_BD=np.linalg.norm(rB-rD)/np.linalg.norm(rD),
                     rel_CD=np.linalg.norm(rC-rD)/np.linalg.norm(rD),
                     psnr_m=psnr(mm(rc), mm(rD)),
                     psnr_m_B=psnr(mm(rB), mm(rD)),
                     psnr_m_C=psnr(mm(rC), mm(rD))))

import statistics as st
def q(k, sel=None):
    v = [r[k] for r in rows if sel is None or sel(r)]
    v.sort()
    n = len(v)
    return v[n//2], v[int(0.9*n)], v[-1], n

print("Couche de reflet : erreur relative L2 contre la reference complete")
print(f"{'variante':<46} {'med':>8} {'p90':>8} {'max':>8}")
for k, nom in (("rel_AD", "A code tel quel"),
               ("rel_BD", "B noyau du code + geometrie exacte"),
               ("rel_CD", "C noyau exact + geom exacte, defocus d'abord")):
    m, p9, mx, n = q(k)
    print(f"{nom:<46} {m:8.2%} {p9:8.2%} {mx:8.2%}")

print(f"\nPSNR de la mixture m contre la reference ({len(rows)} scenes)")
print(f"{'variante':<46} {'med':>8} {'p10':>8} {'min':>8}")
for k, nom in (("psnr_m", "A code tel quel"),
               ("psnr_m_B", "B noyau du code + geometrie exacte"),
               ("psnr_m_C", "C noyau exact + geom exacte, defocus d'abord")):
    v = sorted(r[k] for r in rows)
    print(f"{nom:<46} {v[len(v)//2]:8.2f} {v[int(0.1*len(v))]:8.2f} {v[0]:8.2f}")

for lab, sel in (("lame simple", lambda r: not r["dbl"]), ("double vitrage", lambda r: r["dbl"])):
    m, p9, mx, n = q("rel_AD", sel)
    vm = sorted(r["psnr_m"] for r in rows if sel(r))
    print(f"\n{lab} (n={n}) : erreur couche med {m:.2%} p90 {p9:.2%} max {mx:.2%}"
          f" | PSNR m med {vm[len(vm)//2]:.1f} min {vm[0]:.1f}")

gh = sorted(r["gmax"] for r in rows)
dp = sorted(r["dpx"] for r in rows)
print(f"\ncontexte : ghost_px max par scene  med {gh[len(gh)//2]:.2f}  p90 {gh[int(0.9*len(gh))]:.2f}  max {gh[-1]:.2f}")
print(f"           defocus_px               med {dp[len(dp)//2]:.2f}  p90 {dp[int(0.9*len(dp))]:.2f}  max {dp[-1]:.2f}")
