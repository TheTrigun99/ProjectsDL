import sys, numpy as np, cv2, glob
sys.path.insert(0, "/home/damien/ProjectsDL/reflection_removal")
from render_ref import *
from generating_sim_data.config import TAU
from generating_sim_data.scene import _geometry

def srgb_dec(x):
    x = x.astype(np.float64)/255.0
    return np.where(x <= 0.04045, x/12.92, ((x+0.055)/1.055)**2.4)
ims = sorted(glob.glob("/home/damien/ProjectsDL/reflection_removal/real_photos_random/*"))
def load(p, size=384):
    a = cv2.imread(p)[..., ::-1]; h, w = a.shape[:2]; s = min(h, w)
    a = a[(h-s)//2:(h-s)//2+s, (w-s)//2:(w-s)//2+s]
    return srgb_dec(cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA))
SIZE = 384
R0, T0 = load(ims[1]), load(ims[2])
def psnr(a,b):
    e = np.mean((a-b)**2); return 99.0 if e<=0 else 10*np.log10(1.0/e)

# ---- validation du moteur : mon rendu avec LE noyau ET LA geometrie du code
def code_geom_layer(r, p, size):
    sc = _geometry(size, p["fov"], p["phi"], p["theta0"], p["N"], p["d_focus"],
                   p["d_refl"], p["h_glass"], p["gap"], p["double_pane"])
    M = maps(size, p)
    src = defocus_continuous(r, sc["defocus_px"])
    dx = sc["ghost_px"]*sc["ghost_dir"][...,0]
    dy = sc["ghost_px"]*sc["ghost_dir"][...,1]
    a = sc["a_ghost"][...,None]
    g = warp(src, dx, dy, M)
    return sc["R_map"]*(src + a*g)/(1.0+a), sc

rng = np.random.default_rng(7)
print("validation : mon moteur avec le noyau+geometrie du code vs code_layer")
w = []
for _ in range(25):
    p = draw_scalars(rng, BIAS_PRESETS["hard"])
    a, sc = code_layer(R0, p, SIZE); b, _ = code_geom_layer(R0, p, SIZE)
    w.append(np.abs(a-b).max())
print(f"  ecart max absolu sur 25 scenes : {max(w):.2e}  -> moteur equivalent\n")

for preset in ("defocus_high_v2", "uniform", "hard"):
    rng = np.random.default_rng(1009)
    B = BIAS_PRESETS[preset]
    rows = []
    for i in range(140):
        p = draw_scalars(rng, B)
        M = maps(SIZE, p)
        rc, sc = code_layer(R0, p, SIZE); dp = sc["defocus_px"]
        rD = ref_layer(R0, p, M, "exact", "exact", "defocus_last", dp)
        rB = ref_layer(R0, p, M, "exact", "code", "defocus_first", dp)
        rC = ref_layer(R0, p, M, "exact", "exact", "defocus_first", dp)
        tl = T0*(1.0-sc["R_map"]); e = TAU/max((tl+rD).mean(), 1e-8)
        mm = lambda rr: np.clip((tl+rr)*e, 0, 1)
        rows.append(dict(dbl=p["double_pane"], g=float(sc["ghost_px"].max()),
                         th0=np.degrees(p["theta0"]), fov=p["fov"],
                         A=np.linalg.norm(rc-rD)/np.linalg.norm(rD),
                         B=np.linalg.norm(rB-rD)/np.linalg.norm(rD),
                         C=np.linalg.norm(rC-rD)/np.linalg.norm(rD),
                         pA=psnr(mm(rc), mm(rD)), pB=psnr(mm(rB), mm(rD)),
                         pC=psnr(mm(rC), mm(rD))))
    def show(lab, sel):
        s = [r for r in rows if sel(r)]
        if not s: print(f"  {lab:<34} (vide)"); return
        f = lambda k: sorted(r[k] for r in s)
        a, b, c = f("A"), f("B"), f("C"); pa = f("pA")
        n = len(s)
        print(f"  {lab:<34} n={n:>3} ({n/len(rows):4.0%})  "
              f"A {a[n//2]:6.2%}/{a[-1]:6.2%}  B {b[n//2]:6.2%}/{b[-1]:6.2%}  "
              f"C {c[n//2]:6.2%}/{c[-1]:6.2%}   PSNR m med {pa[n//2]:5.1f} min {pa[0]:5.1f}")
    print(f"=== preset {preset} ===   (erreur couche : mediane/max)")
    show("tout", lambda r: True)
    show("fantome >= 1 px", lambda r: r["g"] >= 1.0)
    show("fantome >= 3 px", lambda r: r["g"] >= 3.0)
    show("lame simple, fantome >= 1 px", lambda r: r["g"] >= 1.0 and not r["dbl"])
    show("double vitrage, fantome >= 1 px", lambda r: r["g"] >= 1.0 and r["dbl"])
    g = sorted(r["g"] for r in rows)
    print(f"  ghost_px max : med {g[len(g)//2]:.2f} p90 {g[int(.9*len(g))]:.2f} max {g[-1]:.2f}"
          f" | part de scenes avec fantome >= 1 px : {np.mean([r['g']>=1 for r in rows]):.0%}"
          f" | double vitrage : {np.mean([r['dbl'] for r in rows]):.0%}\n")
