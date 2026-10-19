"""Generation of the simulated reflection-removal dataset.

Two runnable stages, each with its own subcommand:

    python -m generating_sim_data cache    --dng-dir ... --cache-dir ...
    python -m generating_sim_data generate --out ... --n-groups 15000

The modules, bottom up:

    config          paths, constants, GenConfig / CacheConfig / WeakConfig
    colorimetry     XYZ/sRGB matrices, Bradford, gamma
    dng_metadata    DNG tags, as-shot white point, EXIF exposure
    raw_processing  RAW -> XYZ + saturation mask (ACR steps 1-4)
    cache           stage 1: every DNG -> .npy cache + pool_index.json
    pool            cache -> memmap entries, crops, pair sampling
    scene           sampling of the glass, Fresnel, defocus, ghosting
    simulate        Func. 1 and S1: composition of m, t, r, true_t
    culling         culling measurements and verdicts
    generate        stage 2: the multi-threaded loop and the npz writing
    display         visualisation (matplotlib, off the generation path)

Library usage:

    from generating_sim_data import GenConfig, build_pool, generate
    pool = build_pool()
    generate(GenConfig(out_dir=Path("/tmp/essai"), n_groups=50), pool)
"""
from __future__ import annotations

from .config import (BIAS_PRESETS, CULL_SIZE, PAIR_PROBS, PATCH, SEED, TAU,
                     CacheConfig, GenConfig, WeakConfig)
from .colorimetry import (D50, LUM, XYZ_D50_TO_SRGB, XYZ_D65_TO_SRGB,
                          apply_matrix, cat_matrix, luminance, srgb_decode,
                          srgb_encode, xy_to_xyz)
from .culling import (CULLERS, descripteurs, is_useful, measure, measure_a_tdom,
                      psnr, psnr_raw_to_jpeg, ssim_fast)
from .generate import Generator, generate
from .pool import (Pool, attach_entry, build_pool, kind, levels, load_pool,
                   make_t_crop, random_crop)
from .scene import (DET_PARAM_PRESETS, defocus_continuous,
                    deterministic_scene2, double_reflection, fresnel_angle,
                    fresnel_reflectance, fresnel_single, sample_scene2,
                    sample_scene_lowres, tilt)
from .simulate import (apply_exposure, compose_weak, compute_exposure, prep_t,
                       simulate_example, simulate_example_enc, simulate_weak)

__all__ = [
    "BIAS_PRESETS", "CULLERS", "CULL_SIZE", "D50", "DET_PARAM_PRESETS", "LUM",
    "PAIR_PROBS", "PATCH", "SEED", "TAU", "XYZ_D50_TO_SRGB", "XYZ_D65_TO_SRGB",
    "CacheConfig", "GenConfig", "Generator", "Pool", "WeakConfig",
    "apply_exposure", "apply_matrix", "attach_entry", "build_pool",
    "cat_matrix", "compose_weak", "compute_exposure", "defocus_continuous",
    "descripteurs", "deterministic_scene2", "double_reflection",
    "fresnel_angle", "fresnel_reflectance", "fresnel_single", "generate",
    "is_useful", "kind", "levels", "load_pool", "luminance", "make_t_crop",
    "measure", "measure_a_tdom", "prep_t", "psnr", "psnr_raw_to_jpeg",
    "random_crop", "sample_scene2", "sample_scene_lowres", "simulate_example",
    "simulate_example_enc", "simulate_weak", "srgb_decode", "srgb_encode",
    "ssim_fast", "tilt", "xy_to_xyz",
]
