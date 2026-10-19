"""De combien la correction change-t-elle l'amplitude du fantome en pratique ?"""
import sys, numpy as np
sys.path.insert(0, "/home/damien/ProjectsDL/reflection_removal")
from render_ref import draw_scalars
from generating_sim_data.scene import _geometry
from generating_sim_data.config import BIAS_PRESETS, GLASS_IOR as NG
from t9_fix import propose

SIZE = 384
for preset in ("defocus_high", "uniform", "hard"):
    rng = np.random.default_rng(1009)
    B = BIAS_PRESETS[preset]
    cod, ref, rat, n_act_c, n_act_r = [], [], [], 0, 0
    for _ in range(3000):
        p = draw_scalars(rng, B)
        s = _geometry(SIZE, p["fov"], p["phi"], p["theta0"], p["N"], p["d_focus"],
                      p["d_refl"], p["h_glass"], p["gap"], p["double_pane"])
        gx, gy = propose(SIZE, p["fov"], p["phi"], p["theta0"], p["h_glass"],
                         p["gap"], p["double_pane"], p["d_refl"])
        gc, gr = float(s["ghost_px"].max()), float(np.hypot(gx, gy).max())
        cod.append(gc); ref.append(gr); rat.append(gr/max(gc, 1e-12))
        n_act_c += gc >= 0.15; n_act_r += gr >= 0.15
    f = lambda v: (np.median(v), np.percentile(v, 90), np.max(v))
    mc, pc, xc = f(cod); mr, pr, xr = f(ref); mq, pq, xq = f(rat)
    print(f"=== {preset} (3000 scenes) ===")
    print(f"  ghost_px max  code      med {mc:6.2f}  p90 {pc:6.2f}  max {xc:7.2f}")
    print(f"  ghost_px max  corrige   med {mr:6.2f}  p90 {pr:6.2f}  max {xr:7.2f}")
    print(f"  rapport corrige/code    med {mq:6.3f}  p90 {pq:6.3f}  max {xq:7.3f}")
    print(f"  scenes ou le fantome est applique (>=0.15 px) : code {n_act_c/30:.1f}%  "
          f"corrige {n_act_r/30:.1f}%\n")
