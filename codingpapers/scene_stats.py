"""Statistics of the scene sampler used in data_clean.ipynb.

Reproduces sample_scene() exactly (same formulas, same ranges) but vectorised
over N draws, and turns each draw into the three quantities that actually
decide what the image looks like:

    defocus_px   radius (in px) of the disk kernel applied to the reflection
    R_bar        mean Fresnel reflectance over the patch (2 interfaces)
    ghost_px     lateral shift of the second-surface echo, in px

The per-pixel maps (R_map, ghost_px) are evaluated on a coarse angular grid
(GRID x GRID) spanning the same field of view as the real patch: both are
smooth functions of the ray direction, so the mean / percentiles converge
long before 512x512.
"""
import numpy as np

GLASS_IOR = 1.52
SENSOR_W = 0.024
GRID = 48


def sample_stats(n=10_000, size=512, seed=0, grid=GRID):
    rng = np.random.default_rng(seed)

    # --- scalar draws, identical ranges to sample_scene ---------------------
    fov = rng.uniform(40, 90, n)
    f_px = (size / 2) / np.tan(np.radians(fov) / 2)
    theta0 = np.radians(rng.uniform(0, 55, n))
    phi = rng.uniform(0, 2 * np.pi, n)
    N = rng.uniform(1.8, 8.0, n)
    d_focus = 10.0 ** rng.uniform(-0.3, 0.9, n)
    d_glass = rng.uniform(0.3, np.maximum(0.4, np.minimum(3.0, d_focus)))
    d_refl = d_glass + 10.0 ** rng.uniform(0.0, 1.7, n)
    h_glass = rng.uniform(0.003, 0.015, n)

    # --- defocus: scalar, one radius per image -----------------------------
    f_m = f_px / size * SENSOR_W                       # focal length (m)
    coc = f_m ** 2 / (N * np.maximum(d_focus - f_m, 1e-3)) * np.abs(d_refl - d_focus) / d_refl
    defocus_px = coc / SENSOR_W * size

    # --- angular grid: same FOV as the patch, coarser sampling --------------
    ax = np.linspace(-(size - 1) / 2, (size - 1) / 2, grid)
    ys, xs = np.meshgrid(ax, ax, indexing="ij")
    xs, ys = xs.ravel(), ys.ravel()                    # (G,)

    nx = np.sin(theta0) * np.cos(phi)
    ny = np.sin(theta0) * np.sin(phi)
    nz = np.cos(theta0)

    p = xs * nx[:, None] + ys * ny[:, None] + (f_px * nz)[:, None]
    inv = 1.0 / np.sqrt(xs ** 2 + ys ** 2 + (f_px ** 2)[:, None])
    cos_i = np.clip(np.abs(p) * inv, 1e-6, 1.0)        # (n, G)

    # --- Fresnel (unpolarised), then the two-interface geometric sum --------
    sin_t = np.sqrt(1 - cos_i ** 2) / GLASS_IOR
    cos_t = np.sqrt(1 - sin_t ** 2)
    rs = ((cos_i - GLASS_IOR * cos_t) / (cos_i + GLASS_IOR * cos_t)) ** 2
    rp = ((GLASS_IOR * cos_i - cos_t) / (GLASS_IOR * cos_i + cos_t)) ** 2
    R1 = 0.5 * (rs + rp)
    R = 2 * R1 / (1 + R1)

    # --- ghost: second-surface echo ----------------------------------------
    ghost = (sin_t / cos_t) * cos_i * (2 * f_px * h_glass / d_glass)[:, None]

    return dict(
        fov=fov, theta0=np.degrees(theta0), N=N, d_focus=d_focus, d_glass=d_glass,
        d_refl=d_refl, h_glass=h_glass, f_mm=f_m * 1000,
        depth_sep=np.abs(d_refl - d_focus) / d_refl,
        defocus_px=defocus_px,
        incidence=np.degrees(np.arccos(cos_i)).mean(1),
        R_bar=R.mean(1), R_min=R.min(1), R_max=R.max(1),
        R_span=R.max(1) - R.min(1),
        ghost_bar=ghost.mean(1), ghost_max=ghost.max(1),
        transmit=(1 - R).mean(1),
    )


def bucket(x, edges, labels):
    idx = np.digitize(x, edges)
    return np.array(labels)[idx]


def table(name, labels, counts, n):
    print(f"\n{name}")
    print("-" * 78)
    for lab, c in zip(labels, counts):
        bar = "#" * int(round(60 * c / n))
        print(f"  {lab:<34s} {c / n:6.1%}  {bar}")


if __name__ == "__main__":
    N_DRAW = 10_000
    for SIZE in (512, 256):
        s = sample_stats(N_DRAW, size=SIZE, seed=1009)
        print("=" * 78)
        print(f"{N_DRAW} draws, patch = {SIZE} px, sensor width = {SENSOR_W * 1000:.0f} mm")
        print("=" * 78)

        for k in ("fov", "f_mm", "theta0", "incidence", "N", "d_focus", "d_glass",
                  "d_refl", "depth_sep", "defocus_px", "R_bar", "R_span",
                  "ghost_bar", "ghost_max"):
            v = s[k]
            q = np.percentile(v, [1, 5, 25, 50, 75, 95, 99])
            print(f"{k:>11s}  mean={v.mean():8.3f}  "
                  + "  ".join(f"p{p}={x:8.3f}" for p, x in zip([1, 5, 25, 50, 75, 95, 99], q)))

        # ---- sharpness of the reflection (radius of the disk kernel) -------
        d_edges = [1, 3, 8, 20]
        d_labels = ["sharp   (r < 1 px, no blur)", "soft    (1-3 px)",
                    "blurry  (3-8 px)", "v.blurry(8-20 px)", "wash    (> 20 px)"]
        dc = np.bincount(np.digitize(s["defocus_px"], d_edges), minlength=5)
        table(f"REFLECTION SHARPNESS  (defocus radius, patch {SIZE})", d_labels, dc, N_DRAW)

        # ---- strength of the reflection ------------------------------------
        r_edges = [0.09, 0.12, 0.20]
        r_labels = ["faint  (R < 0.09, near-normal)", "moderate (0.09-0.12)",
                    "strong   (0.12-0.20)", "grazing  (R > 0.20)"]
        rc = np.bincount(np.digitize(s["R_bar"], r_edges), minlength=4)
        table("REFLECTION STRENGTH  (mean Fresnel R over patch)", r_labels, rc, N_DRAW)

        # ---- ghost ----------------------------------------------------------
        g_edges = [1, 3, 8]
        g_labels = ["none   (max shift < 1 px)", "subtle (1-3 px)",
                    "visible(3-8 px)", "split  (> 8 px)"]
        gc = np.bincount(np.digitize(s["ghost_max"], g_edges), minlength=4)
        table("GHOST (second surface)", g_labels, gc, N_DRAW)

        # ---- joint: sharpness x strength ------------------------------------
        di = np.digitize(s["defocus_px"], d_edges)
        ri = np.digitize(s["R_bar"], r_edges)
        print("\nJOINT  rows = sharpness, cols = strength   (% of dataset)")
        print("-" * 78)
        print(f"{'':<26s}" + "".join(f"{l.split('(')[0]:>13s}" for l in r_labels))
        for i, lab in enumerate(d_labels):
            row = [(100.0 * ((di == i) & (ri == j)).sum() / N_DRAW) for j in range(4)]
            print(f"{lab:<26s}" + "".join(f"{v:12.1f}%" for v in row))
        print()


# ---------------------------------------------------------------------------
# What the pipeline ACTUALLY applies, and which draws produce which class.
# ---------------------------------------------------------------------------
def applied_radius(defocus_px):
    """simulate_example does defocus(r, ceil(defocus_px)), and defocus()
    returns the image untouched when radius < 1. So any strictly positive
    defocus_px gets rounded UP to a radius-1 disk (a 5-point cross)."""
    return np.ceil(defocus_px)


def conditional(s, mask, keys):
    """p10 / median / p90 of each input parameter, restricted to a class."""
    out = {}
    for k in keys:
        v = s[k][mask]
        out[k] = np.percentile(v, [10, 50, 90]) if v.size else np.full(3, np.nan)
    return out


def recipes(n=10_000, size=512, seed=1009):
    s = sample_stats(n, size=size, seed=seed)
    r_app = applied_radius(s["defocus_px"])
    keys = ["fov", "N", "d_focus", "d_refl", "depth_sep", "theta0", "incidence",
            "d_glass", "h_glass"]

    print("=" * 96)
    print(f"APPLIED disk radius = ceil(defocus_px), patch {size}")
    print("=" * 96)
    vals, cnt = np.unique(np.minimum(r_app, 12), return_counts=True)
    for v, c in zip(vals, cnt):
        lab = f"r = {int(v)}" + (" (kernel skipped)" if v < 1 else "")
        lab = "r >= 12" if v == 12 else lab
        print(f"  {lab:<22s} {c / n:6.2%}  {'#' * int(round(60 * c / n))}")

    classes = {
        "SHARP reflection   (r <= 1)":  r_app <= 1,
        "SOFT               (2-3)":     (r_app >= 2) & (r_app <= 3),
        "BLURRY             (4-8)":     (r_app >= 4) & (r_app <= 8),
        "VERY BLURRY        (> 8)":     r_app > 8,
        "FAINT refl.  (R < 0.09)":      s["R_bar"] < 0.09,
        "STRONG refl. (R > 0.15)":      s["R_bar"] > 0.15,
        "GRAZING refl.(R > 0.25)":      s["R_bar"] > 0.25,
        "NO ghost     (< 1 px)":        s["ghost_max"] < 1,
        "SPLIT ghost  (> 8 px)":        s["ghost_max"] > 8,
        "HARD: sharp + strong":         (r_app <= 1) & (s["R_bar"] > 0.15),
        "EASY: blurry + faint":         (r_app >= 4) & (s["R_bar"] < 0.09),
    }

    hdr = "  ".join(f"{k:>16s}" for k in keys)
    print("\n" + "=" * 96)
    print(f"PARAMETER BANDS PER CLASS  (p10 - median - p90), patch {size}")
    print("=" * 96)
    print(f"{'class':<28s}{'share':>7s}   {hdr}")
    for lab, m in classes.items():
        c = conditional(s, m, keys)
        cells = "  ".join(f"{c[k][0]:5.2f}-{c[k][1]:4.2f}-{c[k][2]:5.2f}" for k in keys)
        print(f"{lab:<28s}{m.mean():6.1%}   {cells}")
    print(f"{'(full dataset)':<28s}{1.0:6.1%}   "
          + "  ".join(f"{np.percentile(s[k], 10):5.2f}-{np.median(s[k]):4.2f}-"
                      f"{np.percentile(s[k], 90):5.2f}" for k in keys))
    print("\nkeys order:", ", ".join(keys))
    return s, r_app


if __name__ == "__main__" and True:
    recipes(10_000, size=512)
