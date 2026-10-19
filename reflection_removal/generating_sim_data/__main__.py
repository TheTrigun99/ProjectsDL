"""Command-line interface.

    # stage 1: DNG -> XYZ cache (to be done once per source)
    python -m generating_sim_data cache \
        --dng-dir /home/damien/data_fivek_dng/raise_outdoor_dng \
        --cache-dir /home/damien/data_fivek_dng/pool_raise_cache \
        --index-json /home/damien/data_fivek_dng/raise_outdoor.json

    # stage 2: generation
    python -m generating_sim_data generate --out /tmp/essai --n-groups 50
    python -m generating_sim_data generate --out .../dataset_weak --n-groups 15000 \
        --p-weak 1.0 --cull std --weak-band 24 27.5 --seed 1009

    # extend an existing dataset: a new seed AND an offset on the indices
    python -m generating_sim_data generate --out .../dataset_weak \
        --n-groups 5000 --seed 2009 --group-offset 15000 --p-weak 1.0

    # a look at a few examples, without writing anything durable
    python -m generating_sim_data preview --out /tmp/apercu -n 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import (BIAS_PRESETS, DEFAULT_BIAS, CacheConfig, GenConfig,
                     WeakConfig)


def _add_pool_args(p):
    p.add_argument("--pools", nargs="+", default=["fivek", "raise", "ibl"],
                   choices=["fivek", "raise", "ibl"],
                   help="pool sources (the canonical order is imposed)")
    p.add_argument("--with-exr", action="store_true",
                   help="adds mc_exr_cache to the ibl sources")


def _build_parser():
    ap = argparse.ArgumentParser(
        prog="generating_sim_data",
        description="Generation of the simulated reflection-removal dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ---------------------------------------------------------------- cache
    c = sub.add_parser("cache", help="DNG -> float16 XYZ cache")
    c.add_argument("--dng-dir", type=Path, required=True)
    c.add_argument("--cache-dir", type=Path, required=True)
    c.add_argument("--index-json", type=Path, default=None,
                   help="json {name: {categories, path}} for loc and cats")
    c.add_argument("--loc", default="outdoor",
                   help="fallback loc when the json gives none")
    c.add_argument("--workers", type=int, default=16)
    c.add_argument("--full-size", action="store_true",
                   help="no 2x2 binning: 4x more pixels and disk")
    c.add_argument("--rebuild", action="store_true")
    c.add_argument("--limit", type=int, default=None)

    # ------------------------------------------------------------- generate
    g = sub.add_parser("generate", help="XYZ cache -> simulated examples")
    g.add_argument("--out", type=Path, required=True, help="output directory")
    g.add_argument("--n-groups", type=int, default=200)
    g.add_argument("-k", type=int, default=1,
                   help="examples per group (same transmission)")
    g.add_argument("--min-kept", type=int, default=1)
    g.add_argument("--max-r-tries", type=int, default=70)
    g.add_argument("--max-group-tries-factor", type=int, default=200,
                   help="trial budget = factor x n_groups; the a_tdom culling "
                        "accepts only ~0.3 %% (measured), so allow >= 2000")
    g.add_argument("--p-weak", type=float, default=0.0,
                   help="probability of weak mode (reflection inside a PSNR band)")
    g.add_argument("--workers", type=int, default=16,
                   help="threads; beyond ~16 the GIL becomes the bottleneck again")
    g.add_argument("--seed", type=int, default=GenConfig.seed)
    g.add_argument("--patch", type=int, default=GenConfig.patch)
    g.add_argument("--bias", default=DEFAULT_BIAS, choices=sorted(BIAS_PRESETS),
                   help="sampling preset for the pane")
    g.add_argument("--cull", default="std", choices=["std", "a_tdom", "kee"])
    g.add_argument("--group-offset", type=int, default=0,
                   help="offset on the group numbers (continuation)")
    g.add_argument("--no-maps", action="store_true",
                   help="do not write R_map / ghost_px (~0.5 MB per example)")
    g.add_argument("--weak-band", type=float, nargs=2, default=(24.0, 27.5),
                   metavar=("LO", "HI"), help="targeted JPEG PSNR band")
    g.add_argument("--weak-logl", type=float, nargs=2, default=(-0.45, 0.45),
                   metavar=("LO", "HI"), help="band of log10(L_r / L_t)")
    g.add_argument("--weak-alpha", type=float, default=1.0)
    g.add_argument("--fresnel", default="polar", choices=["polar", "moyenne"],
                   help="polar: Fresnel per polarization (v2); moyenne: generator v1.1")
    g.add_argument("--wb", default="mix", choices=["mix", "t"],
                   help="mix: white balance of the mixture (v2); t: of the transmission (v1.1)")
    g.add_argument("--wb-max-mired", type=float, default=60.0,
                   help="with --wb mix, reject when the white moves more than this")
    _add_pool_args(g)

    # --------------------------------------------------------- generate-mid
    gm = sub.add_parser("generate-mid", help="sharp hard-but-readable / same-scene reflections (mission 4 sampling)")
    gm.add_argument("--out", type=Path, required=True)
    gm.add_argument("--n", type=int, required=True, help="examples to accept")
    gm.add_argument("--seed", type=int, required=True)
    gm.add_argument("--offset", type=int, default=0)
    gm.add_argument("--k0", type=int, default=0, help="first file index (continuation)")
    gm.add_argument("--workers", type=int, default=16)
    gm.add_argument("--cap", type=int, default=100000, help="max uses per source (mission 4: none)")
    gm.add_argument("--meme-scene", action="store_true", help="same-scene buckets (ms_ibl, ms_photo)")
    gm.add_argument("--fenetres", type=Path, default=None, help="luminance windows json (default: fenetres_mid.json)")
    gm.add_argument("--bias", default="defocus_low", choices=sorted(BIAS_PRESETS))
    gm.add_argument("--fresnel", default="polar", choices=["polar", "moyenne"])
    gm.add_argument("--wb", default="mix", choices=["mix", "t"])
    gm.add_argument("--wb-max-mired", type=float, default=60.0)
    gm.add_argument("--duree-max", type=float, default=1e9, help="seconds")

    # -------------------------------------------------------- generate-weak2
    gw = sub.add_parser("generate-weak2", help="weak reflections with a bounded alpha (gen_weak2 recipe)")
    gw.add_argument("--out", type=Path, required=True)
    gw.add_argument("--n", type=int, required=True)
    gw.add_argument("--seed", type=int, required=True)
    gw.add_argument("--workers", type=int, default=16)
    gw.add_argument("--lo-db", type=float, default=27.5)
    gw.add_argument("--hi-db", type=float, default=30.0)
    gw.add_argument("--ibl-frac", type=float, default=0.4)
    gw.add_argument("--max-use", type=int, default=6)
    gw.add_argument("--marge-min", type=float, default=2.5)
    gw.add_argument("--bias", default="defocus_high", choices=sorted(BIAS_PRESETS))
    gw.add_argument("--fresnel", default="polar", choices=["polar", "moyenne"])
    gw.add_argument("--wb", default="mix", choices=["mix", "t"])
    gw.add_argument("--wb-max-mired", type=float, default=60.0)
    gw.add_argument("--group-offset", type=int, default=0)

    # -------------------------------------------------------------- preview
    p = sub.add_parser("preview", help="a few examples as PNG, just to look")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("-n", type=int, default=8)
    p.add_argument("--seed", type=int, default=GenConfig.seed)
    p.add_argument("--patch", type=int, default=GenConfig.patch)
    p.add_argument("--bias", default=DEFAULT_BIAS, choices=sorted(BIAS_PRESETS))
    p.add_argument("--cull", default="std", choices=["std", "a_tdom", "kee"])
    p.add_argument("--p-weak", type=float, default=0.0)
    p.add_argument("--fresnel", default="polar", choices=["polar", "moyenne"])
    p.add_argument("--wb", default="mix", choices=["mix", "t"])
    p.add_argument("--wb-max-mired", type=float, default=60.0)
    _add_pool_args(p)
    return ap


def _gen_config(a):
    return GenConfig(
        out_dir=a.out, n_groups=a.n_groups, k=a.k, min_kept=a.min_kept,
        max_r_tries=a.max_r_tries, p_weak=a.p_weak, n_workers=a.workers,
        max_group_tries_factor=a.max_group_tries_factor,
        seed=a.seed, patch=a.patch, bias=a.bias, cull=a.cull,
        group_offset=a.group_offset, pools=tuple(a.pools),
        with_exr=a.with_exr, save_params=not a.no_maps,
        fresnel=a.fresnel, wb=a.wb, wb_max_mired=a.wb_max_mired,
        weak=WeakConfig(band=tuple(a.weak_band), logl=tuple(a.weak_logl),
                        alpha=a.weak_alpha))


def _preview(a):
    """Generates n accepted examples and writes them side by side as PNG."""
    import random

    import cv2
    import numpy as np

    from ._paths import ensure
    from .generate import Generator
    from .pool import build_pool
    ensure()
    from acr_tone_fast import finish

    cfg = GenConfig(out_dir=a.out, n_groups=a.n, n_workers=1, seed=a.seed,
                    patch=a.patch, bias=a.bias, cull=a.cull, p_weak=a.p_weak,
                    pools=tuple(a.pools), with_exr=a.with_exr, fresnel=a.fresnel,
                    wb=a.wb, wb_max_mired=a.wb_max_mired)
    pool = build_pool(cfg.pools, with_exr=cfg.with_exr)
    print(f"pool : {len(pool)} scenes {pool.counts()}")

    gen = Generator(cfg, pool)
    a.out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng([a.seed, 10_000])
    rnd = random.Random(a.seed * 1_000_003 + 10_000)
    kept = 0
    tried = 0
    while kept < a.n and tried < 400 * a.n:
        tried += 1
        combo = pool.pick_pair_type(rnd)
        draw = gen._draw_weak if rnd.random() < cfg.p_weak else gen._draw_std
        ex, d, ia, ib, _ = draw(combo, rng, rnd)
        if ex is None:
            continue
        row = np.concatenate([finish(ex[k], ev=0.0, tone=True)
                              for k in ("m", "true_t", "r")], axis=1)
        name = (f"apercu_{kept:02d}_{combo[0]}_{combo[1]}_"
                f"{pool[ia]['name']}_{pool[ib]['name']}.png")
        cv2.imwrite(str(a.out / name), row[..., ::-1])
        print(f"{name}  defocus={ex['params']['defocus_px']:.2f}px  "
              + " ".join(f"{k}={v:.3f}" for k, v in d.items()
                         if isinstance(v, float))[:120])
        kept += 1
    print(f"{kept} apercus (m | true_t | r) sur {tried} essais -> {a.out}")


def main(argv=None):
    a = _build_parser().parse_args(argv)
    if a.cmd == "cache":
        from .cache import build_cache
        build_cache(CacheConfig(dng_dir=a.dng_dir, cache_dir=a.cache_dir,
                                index_json=a.index_json, loc=a.loc,
                                n_workers=a.workers, half_size=not a.full_size,
                                rebuild=a.rebuild, limit=a.limit))
    elif a.cmd == "generate":
        from .generate import generate
        generate(_gen_config(a))
    elif a.cmd == "generate-mid":
        from .mid import run_mid
        raise SystemExit(run_mid(a.out, a.n, a.seed, offset=a.offset, k0=a.k0, workers=a.workers, cap=a.cap,
                                 meme_scene=a.meme_scene, fenetres=a.fenetres, bias=a.bias, fresnel=a.fresnel,
                                 wb=a.wb, wb_max_mired=a.wb_max_mired, duree_max=a.duree_max))
    elif a.cmd == "generate-weak2":
        from .weak2 import run_weak2
        raise SystemExit(run_weak2(a.out, a.n, a.seed, workers=a.workers, lo_db=a.lo_db, hi_db=a.hi_db,
                                   ibl_frac=a.ibl_frac, max_use=a.max_use, marge_min=a.marge_min, bias=a.bias,
                                   fresnel=a.fresnel, wb=a.wb, wb_max_mired=a.wb_max_mired,
                                   group_offset=a.group_offset))
    elif a.cmd == "preview":
        _preview(a)


if __name__ == "__main__":
    main()
