"""Paths, constants and configuration objects for the pipeline.

Everything that used to be a "GLOBAL PARAMS" cell of the notebook is here. The
two runnable stages each have their own dataclass:

  CacheConfig  : RAW (DNG) -> float16 XYZ cache + saturation mask
  GenConfig    : XYZ cache -> simulated and culled (m, true_t) examples

The physical constants (GLASS_IOR, SENSOR_W) are read at import time by
`scene.py`; to sweep other values, pass them as arguments to
`sample_scene2(..., ior=, sensor_w=)` rather than mutating this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
DATA_DIR = Path("/home/damien/data_fivek_dng")

DNG_DIR       = DATA_DIR / "dng"                 # FiveK
RAISE_DNG_DIR = DATA_DIR / "raise_outdoor_dng"   # RAISE outdoor

POOL_CACHE  = DATA_DIR / "pool_cache"        # FiveK indoor + outdoor
RAISE_CACHE = DATA_DIR / "pool_raise_cache"  # RAISE outdoor
IBL_CACHE   = DATA_DIR / "ibl_cache"         # HDRI (Poly Haven ...)
EXR_CACHE   = DATA_DIR / "mc_exr_cache"      # EXR Laval
LEVELS_DIR  = DATA_DIR / "cache_tot"         # cache of mean radiances

# --------------------------------------------------------------------------
# Simulation constants
# --------------------------------------------------------------------------
PATCH = 384             # side of the simulated patch (the paper works at 256)
TAU = 0.1329            # = srgb_decode(0.4), target for the mean pixel (Func. S1)
SAT_TOL = 0.02          # above this, exposure switches to "do not clip" mode
CULL_SIZE = 256         # resolution at which we measure in order to cull
JPEG_QUALITY = 95
SEED = 1009

GLASS_IOR = 1.52
SENSOR_FULL_MIN = 0.024   # full frame 36x24, short side (the patch is square)
SENSOR_APS_C_MIN = 0.0156
SENSOR_W = SENSOR_FULL_MIN
SENSOR_W_c = SENSOR_APS_C_MIN
BANDE_FOV = (10, 45)          # degrés, petit côté (la cible SIR2 est mal connue)
N_FERME   = (10, 32)          # SIR2 : F11-F32
N_OUVERT  = (2.8, 8)          # minorité ouverte : real20 / nature20 / OpenRR
P_OUVERT  = 0.25
P_PROCHE = 0.5
DEFOCUS_MAX_PX = 128.0
K_MAP = 6
FOV_JSON = DATA_DIR / "fivek_camera.json"
S_PROCHE = (0.20, 0.50)
FOV_IBL = 90.0

# --------------------------------------------------------------------------
# Sampling bias for the glass / camera (sample_scene2's `biais_params` argument)
#
# g = 1.0 -> uniform (RNG stream identical to rng.uniform); g > 1 shifts the
# mass towards the bottom of the support, g < 1 towards the top. See scene.tilt.
# --------------------------------------------------------------------------
BIAS_PRESETS: dict[str, dict[str, float]] = {
    # uniform over the whole support, one pane in two is double glazing
    "uniform": {"fov": 1.0, "N": 1.0, "d_focus": 1.0, "d_glass": 1.0,
                "h_glass": 1.0, "p_double": 0.5, "theta0": 1.0, "gap": 1.0,
                "d_refl": 1.0},
    # low defocus: sharp reflections (dataset net_atdom*)
    "defocus_low": {"fov": 0.6, "N": 0.6, "d_focus": 0.6, "d_glass": 1.0,
                    "h_glass": 1.0, "p_double": 0.1, "theta0": 1.0, "gap": 1.0,
                    "d_refl": 1.0},
    # strong defocus: blurry reflections (dataset_biggest_ever)
    "defocus_high": {"fov": 1.6, "N": 3.0, "d_focus": 2.5, "d_glass": 1.0,
                     "h_glass": 1.0, "p_double": 0.15, "theta0": 1.0, "gap": 1.0,
                     "d_refl": 1.0},
    # MISSION 7 (10/10, Damien): defocus_high with one double glazing in TEN, as the old biggest_ever (9.9 %
    # measured) -- defocus_high (0.15) is kept unchanged so that v1.1 datasets stay reproducible bit for bit.
    # p_double only moves a threshold: the RNG stream is the same as defocus_high.
    "defocus_high_v2": {"fov": 1.6, "N": 3.0, "d_focus": 2.5, "d_glass": 1.0,
                        "h_glass": 1.0, "p_double": 0.10, "theta0": 1.0, "gap": 1.0,
                        "d_refl": 1.0},
    # hard cases: strong defocus AND pronounced ghosting
    "hard": {"fov": 1.8, "N": 2.5, "d_focus": 3.0, "d_glass": 2.5,
             "h_glass": 1.8, "p_double": 0.50, "theta0": 1.0, "gap": 1.0,
             "d_refl": 1.0},
}
DEFAULT_BIAS = "defocus_high"

# --------------------------------------------------------------------------
# Sampling of (transmission, reflection) pairs by source type
#   indoor / outdoor = FiveK, raise = RAISE outdoor, ibl = HDRI + EXR
# The weights are normalised, they need not sum to 1.
# --------------------------------------------------------------------------
PAIR_PROBS: dict[tuple[str, str], float] = {
    ("indoor", "raise"):  0.10,
    ("indoor", "ibl"):    0.20,
    ("indoor", "indoor"): 0.20,
    ("outdoor", "raise"): 0.10,
    ("outdoor", "outdoor"): 0.10,
    ("raise", "raise"):   0.05,
    ("raise", "ibl"):     0.10,
    ("ibl", "indoor"):    0.15
}

# Re-exposure of the HDR pools: the cache was written with a median SRC_MED,
# we bring it back to the physical radiance expected per location (cells 35-36).
HDR_SRC_MED = 0.056
HDR_TARGET = {"indoor": 0.0618, "outdoor": 2.710}


@dataclass
class WeakConfig:
    """Weak mode: reflections that leave the mixture inside a PSNR band.

    The band is measured on m vs true_t after tone mapping + a JPEG round trip
    (culling.psnr_raw_to_jpeg), not on the linear data.

    NOTE: `tol` and `marge_min` are currently unused by the package; they are
    kept because they are recorded in meta.json of past runs.
    """
    band: tuple[float, float] = (24.0, 27.5)   # targeted JPEG PSNR (dB)
    logl: tuple[float, float] = (-0.45, 0.45)  # band of log10(L_r / L_t)
    alpha: float = 1.0                         # gain applied to the reflection
    tol: float = 0.5                           # tolerance around the target (unused)
    marge_min: float = 2.5                     # reflection / attenuation margin, dB (unused)


@dataclass
class CacheConfig:
    """Stage 1: DNG -> XYZ cache."""
    dng_dir: Path = RAISE_DNG_DIR
    cache_dir: Path = RAISE_CACHE
    index_json: Path | None = None   # HuggingFace/RAISE json, for loc + cats
    loc: str = "outdoor"             # fallback loc when the json gives none
    n_workers: int = 16
    half_size: bool = True           # half resolution (2x2 demosaicing)
    rebuild: bool = False
    limit: int | None = None


@dataclass
class GenConfig:
    """Stage 2: XYZ cache -> simulated examples."""
    out_dir: Path = DATA_DIR / "net_atdom3"
    n_groups: int = 200
    k: int = 1                  # examples per group (same t, different reflections)
    min_kept: int = 1           # a group below this is rejected
    max_r_tries: int = 70       # reflections tried per extra member
    max_group_tries_factor: int = 200
    p_weak: float = 0.0         # probability of weak mode
    n_workers: int = 16
    seed: int = SEED
    patch: int = PATCH
    bias: str = DEFAULT_BIAS
    cull: str = "a_tdom"        # culling.CULLERS
    group_offset: int = 0       # to extend an existing dataset
    pools: tuple[str, ...] = ("fivek", "raise", "ibl")
    with_exr: bool = False      # adds mc_exr_cache to the "ibl" pool
    save_params: bool = True    # R_map and ghost_px inside the npz
    # MISSION 7 (10/10), "simulator v2". fresnel="moyenne" + wb="t" = generator v1.1.
    fresnel: str = "polar"      # "polar": Fresnel per polarization; "moyenne": average first
    wb: str = "mix"             # "mix": white of the mixture; "t": white of the transmission
    wb_max_mired: float = 60.0  # reject when the white moves more than this from t's (wb="mix")
    weak: WeakConfig = field(default_factory=WeakConfig)

    @property
    def max_group_tries(self) -> int:
        return self.max_group_tries_factor * self.n_groups

    @property
    def bias_dict(self) -> dict[str, float]:
        return BIAS_PRESETS[self.bias]

    def to_json(self) -> dict:
        d = asdict(self)
        d["out_dir"] = str(self.out_dir)
        d["pools"] = list(self.pools)
        d["weak"] = {k: list(v) if isinstance(v, tuple) else v
                     for k, v in d["weak"].items()}
        return d
