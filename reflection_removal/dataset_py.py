"""
SIRR dataloaders: three sources (simulated NPZ, synthetic PNG, real JPEG).

Output contract -- every dataset yields a dict:
    mixed         float32 CHW in [0, 1], sRGB / display domain
    transmission  idem
    reflection    idem (absent for nature20 / real20 / real89)
    source        str
    size          tensor([H, W]) before any padding, to crop the output back

Two places must be adapted to your existing code, both marked ADAPT below:
    (1) load_npz_linear()  -- the array keys inside your .npz files
    (2) finish_layers()    -- the call signature of your finish()
Nothing else is source-specific.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import (ConcatDataset, DataLoader, Dataset,
                              WeightedRandomSampler)

# OpenCV starts its own thread pool. Inside DataLoader workers this fights
# with the workers themselves and can cost more than the decoding does.
cv2.setNumThreads(0)

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# JPEG round trip
# ---------------------------------------------------------------------------

_SAMPLING = {
    name: getattr(cv2, f"IMWRITE_JPEG_SAMPLING_FACTOR_{name}")
    for name in ("444", "440", "422", "420")
    if hasattr(cv2, f"IMWRITE_JPEG_SAMPLING_FACTOR_{name}")
}
if not _SAMPLING:
    raise RuntimeError("OpenCV >= 4.3 needed for IMWRITE_JPEG_SAMPLING_FACTOR")


@dataclass
class JpegAug:
    """Random in-memory JPEG compression.

    The drawn quality/sampling is shared by every layer of a sample, but each
    layer is encoded separately. That is deliberate: in the real benchmarks M
    and T are two distinct photographs, each compressed on its own, so their
    artefacts do not cancel. Encoding them jointly would be easier than the
    real task and would leak information.

    Re-weight `sampling` once you know what the test files actually are:
        identify -verbose real20/blended/103.jpg | grep sampling-factor
    """
    quality: tuple[int, int] = (90, 100)
    sampling: tuple[str, ...] = ("444", "444", "444", "422", "420")

    def draw(self) -> dict:
        return {"quality": random.randint(*self.quality),
                "sampling": random.choice(self.sampling)}


def jpeg_roundtrip(rgb_u8: np.ndarray, quality: int, sampling: str) -> np.ndarray:
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb_u8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [
        cv2.IMWRITE_JPEG_QUALITY, int(quality),
        cv2.IMWRITE_JPEG_SAMPLING_FACTOR, _SAMPLING[sampling],
    ])
    if not ok:
        return rgb_u8
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Small array helpers
# ---------------------------------------------------------------------------

def imread_rgb(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cv2 could not read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def pad_to_at_least(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    ph, pw = max(0, size - h), max(0, size - w)
    if not (ph or pw):
        return img
    return cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)

def pad_to_multiple(img: np.ndarray, m: int) -> np.ndarray:
    h, w = img.shape[:2]
    ph, pw = (-h) % m, (-w) % m
    if not (ph or pw):
        return img
    return cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)


def to_tensor(rgb_u8: np.ndarray) -> torch.Tensor:
    arr = np.ascontiguousarray(rgb_u8.transpose(2, 0, 1))
    return torch.from_numpy(arr).float()/255.0

# ---------------------------------------------------------------------------
# Transform, shared by all sources
# ---------------------------------------------------------------------------

class PairedTransform:
    """Same geometry and same JPEG setting applied to every layer of a sample.

    Order is crop -> flip -> JPEG. Cropping first means the 8x8 block grid is
    aligned on the crop rather than on the original sensor grid, which
    randomises the block phase across samples. That is cheap and arguably a
    useful augmentation; encoding the full image first would be faithful but
    costs an encode of the whole frame per sample.
    """

    def __init__(self, crop: int = 256, train: bool = True,
                 jpeg: JpegAug | None = None, hflip: bool = True,
                 vflip: bool = False, pad_multiple: int = 32):
        self.crop = crop
        self.train = train
        self.jpeg = jpeg
        self.hflip = hflip
        self.vflip = vflip
        self.pad_multiple = pad_multiple

    def __call__(self, layers: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        shapes = {v.shape[:2] for v in layers.values()}
        if len(shapes) != 1:
            raise ValueError(f"layers are not aligned: {shapes}")
        h, w = shapes.pop()

        if self.train:
            layers = {k: pad_to_at_least(v, self.crop) for k, v in layers.items()}
            hh, ww = next(iter(layers.values())).shape[:2]
            y = random.randint(0, hh - self.crop)
            x = random.randint(0, ww - self.crop)
            layers = {k: v[y:y + self.crop, x:x + self.crop]
                      for k, v in layers.items()}
            if self.hflip and random.random() < 0.5:
                layers = {k: v[:, ::-1] for k, v in layers.items()}
            if self.vflip and random.random() < 0.5:
                layers = {k: v[::-1, :] for k, v in layers.items()}
        elif self.pad_multiple:
            layers = {k: pad_to_multiple(v, self.pad_multiple)
                      for k, v in layers.items()}

        if self.jpeg is not None:
            cfg = self.jpeg.draw()
            layers = {k: jpeg_roundtrip(v, **cfg) for k, v in layers.items()}

        out = {k: to_tensor(v) for k, v in layers.items()}
        out["size"] = torch.tensor([h, w])
        return out


# ---------------------------------------------------------------------------
# Source 1 & 2: anything already on disk as image files
# ---------------------------------------------------------------------------

class PairedImageDataset(Dataset):
    def __init__(self, samples: list[dict], transform: PairedTransform,
                 source: str = ""):
        if not samples:
            raise ValueError(f"empty sample list for source {source!r}")
        self.samples = samples
        self.transform = transform
        self.source = source

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        layers = {k: imread_rgb(p) for k, p in self.samples[i].items()}
        out = self.transform(layers)
        out["source"] = self.source
        return out


def _index_by_stem(folder) -> dict[str, Path]:
    """Stem -> path. Lookup by stem, not by rebuilt filename: the SIR2 folders
    mix .png and .jpg inside a single directory."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    return {p.stem: p for p in sorted(folder.iterdir())
            if p.suffix.lower() in IMG_EXT}


def build_voc(root, with_reflection: bool = True) -> list[dict]:
    """blended/2007_000032-2008_006587.png
         -> transmission_layer/2007_000032.png
         -> reflection_layer/2008_006587.png
    VOC ids contain no hyphen, so partitioning on the first one is safe.
    (reflection_mask_layer/ is keyed by the reflection id if you want it.)"""
    root = Path(root)
    trans = _index_by_stem(root / "transmission_layer")
    refl = _index_by_stem(root / "reflection_layer")
    samples, missing = [], 0
    for stem, m in _index_by_stem(root / "blended").items():
        t_stem, _, r_stem = stem.partition("-")
        if t_stem not in trans or (with_reflection and r_stem not in refl):
            missing += 1
            continue
        rec = {"mixed": m, "transmission": trans[t_stem]}
        if with_reflection:
            rec["reflection"] = refl[r_stem]
        samples.append(rec)
    if missing:
        print(f"[voc] {missing} blended files without counterpart, skipped")
    return samples


def _swap_sir2_tag(stem: str, new_tag: str) -> str | None:
    """ab-10-m-3 -> ab-10-g-3   |   1-Focus-11-m -> 1-Focus-11-g   |  1-m -> 1-g
    The tag is not at a fixed position, so replace the unique standalone 'm'."""
    parts = stem.split("-")
    hits = [i for i, p in enumerate(parts) if p == "m"]
    if len(hits) != 1:
        return None
    parts[hits[0]] = new_tag
    return "-".join(parts)


def build_sir2(root, with_reflection: bool = True) -> list[dict]:
    """root = .../SIR2/PostcardDataset (or SolidObjectDataset, WildSceneDataset)"""
    root = Path(root)
    trans = _index_by_stem(root / "transmission_layer")
    refl = _index_by_stem(root / "reflection")
    samples, missing = [], 0
    for stem, m in _index_by_stem(root / "blended").items():
        t_stem = _swap_sir2_tag(stem, "g")
        r_stem = _swap_sir2_tag(stem, "r")
        if t_stem not in trans or (with_reflection and r_stem not in refl):
            missing += 1
            continue
        rec = {"mixed": m, "transmission": trans[t_stem]}
        if with_reflection:
            rec["reflection"] = refl[r_stem]
        samples.append(rec)
    if missing:
        print(f"[{root.name}] {missing} blended files without counterpart, skipped")
    return samples


def build_paired_by_stem(root, blended="blended",
                         transmission="transmission_layer",
                         reflection=None) -> list[dict]:
    """nature20 / real20 / real89: identical stems across folders.
    For real89, pass the inner directory (.../real89/real89)."""
    root = Path(root)
    trans = _index_by_stem(root / transmission)
    refl = _index_by_stem(root / reflection) if reflection else None
    samples = []
    for stem, m in _index_by_stem(root / blended).items():
        if stem not in trans:
            continue
        rec = {"mixed": m, "transmission": trans[stem]}
        if refl is not None and stem in refl:
            rec["reflection"] = refl[stem]
        samples.append(rec)
    return samples

# ---------------------------------------------------------------------------
# Source 3: simulated NPZ, linear domain
# ---------------------------------------------------------------------------

# ------------------------------------------------------------------ ADAPT (1)
def load_npz_linear(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Return (linear layers, metadata stored in the npz).

    Layers are HWC float arrays in the linear rendered space, i.e. exactly
    what your compositing step produces BEFORE finish(). Rename the keys to
    match your files -- run `np.load(p).files` on one to check.
    """
    z = np.load(path)
    layers = {
        "mixed": z["mixed"],
        "transmission": z["transmission"],
        "reflection": z["reflection"],
    }
    meta = {k: z[k] for k in z.files if k not in layers}
    return layers, meta


# ------------------------------------------------------------------ ADAPT (2)
def finish_layers(layers_lin: dict[str, np.ndarray], meta: dict,
                  finish) -> dict[str, np.ndarray]:
    """Run your ISP with IDENTICAL parameters on every layer, then quantise.

    Trap worth checking before you trust any of this: if finish() estimates
    anything from the image itself (AWB gains, auto exposure), estimating it
    per layer silently breaks the M / T / R relation after the ISP. Estimate
    once on `mixed` and reuse the result for the other two.

    Also filter `meta` down to the kwargs finish() actually accepts -- the
    parquet will carry columns it does not want.
    """
    out = {}
    for name, lin in layers_lin.items():
        srgb = np.asarray(finish(lin, **meta))          # <- your real signature
        if srgb.dtype != np.uint8:
            srgb = np.clip(srgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
        out[name] = srgb
    return out


class SimulatedNpzDataset(Dataset):
    """Stores linear, renders and compresses on the fly. Nothing is ever
    written to disk as JPEG, so changing the ISP or the compression policy
    does not mean regenerating 5000 files."""

    def __init__(self, npz_dir, finish, transform: PairedTransform,
                 trials_parquet=None, source: str = "sim"):
        self.paths = sorted(Path(npz_dir).glob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"no .npz under {npz_dir}")
        self.finish = finish
        self.transform = transform
        self.source = source
        self.trials = None
        if trials_parquet is not None:
            import pandas as pd
            df = pd.read_parquet(trials_parquet)
            key = next((c for c in ("id", "name", "stem", "file", "path")
                        if c in df.columns), None)
            self.trials = df.set_index(key) if key else df
            self.keyed = key is not None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> dict:
        path = self.paths[i]
        layers_lin, meta = load_npz_linear(path)
        if self.trials is not None:
            row = self.trials.loc[path.stem] if self.keyed else self.trials.iloc[i]
            meta = {**row.to_dict(), **meta}       # npz wins on key collision
        layers = finish_layers(layers_lin, meta, self.finish)
        out = self.transform(layers)
        out["source"] = self.source
        return out


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def worker_init(worker_id: int) -> None:
    cv2.setNumThreads(0)


def make_mixing_sampler(datasets, weights, num_samples=None):
    """Draw each dataset with the given probability regardless of its size.
    ConcatDataset alone would give you 5000 / (5000 + 7643) simulated."""
    w = []
    for ds, p in zip(datasets, weights):
        w.extend([p / len(ds)] * len(ds))
    n = num_samples or sum(len(d) for d in datasets)
    return WeightedRandomSampler(torch.tensor(w, dtype=torch.double), n,
                                 replacement=True)

OTHERS = Path("/home/damien/data_others")
SIM = Path("/home/damien/data_fivek_dng/simulated")

def build_train_loader(finish, batch_size=8, crop=256, workers=8,
                       sim_weight=0.7, jpeg=None):
    tf = PairedTransform(crop=crop, train=True, jpeg=jpeg or JpegAug())

    sim = SimulatedNpzDataset(SIM, finish, tf, trials_parquet=SIM / "trials.parquet")
    voc = PairedImageDataset(build_voc(OTHERS / "VOC2012"), tf, source="voc")

    datasets = [sim, voc]
    return DataLoader(
        ConcatDataset(datasets),
        batch_size=batch_size,
        sampler=make_mixing_sampler(datasets, [sim_weight, 1 - sim_weight]),
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=workers > 0,
        worker_init_fn=worker_init,
    )


def build_eval_loaders(pad_multiple=32, workers=4):
    """One loader per benchmark. Kept separate because the key sets differ
    (SIR2 has a reflection layer, nature20 and real20 do not) and default
    collate needs every item in a batch to carry the same keys.

    No JPEG augmentation here: these files are already compressed, re-encoding
    them would stack a second generation of artefacts on top."""
    tf = PairedTransform(train=False, jpeg=None, pad_multiple=pad_multiple)
    rs = OTHERS / "robustsirr_test_dataset"

    specs = {
        "postcard": build_sir2(rs / "SIR2/PostcardDataset"),
        "solid": build_sir2(rs / "SIR2/SolidObjectDataset"),
        "wild": build_sir2(rs / "SIR2/WildSceneDataset"),
        "nature20": build_paired_by_stem(rs / "nature20"),
        "real20": build_paired_by_stem(rs / "real20"),
        "real89": build_paired_by_stem(OTHERS / "real89/real89"),
    }
    return {
        name: DataLoader(PairedImageDataset(s, tf, source=name), batch_size=1,
                         shuffle=False, num_workers=workers,
                         worker_init_fn=worker_init)
        for name, s in specs.items()
    }


if __name__ == "__main__":
    # Index-only smoke test: checks every pairing rule without touching finish().
    rs = OTHERS / "robustsirr_test_dataset"
    for name, s in [
        ("voc", build_voc(OTHERS / "VOC2012")),
        ("postcard", build_sir2(rs / "SIR2/PostcardDataset")),
        ("solid", build_sir2(rs / "SIR2/SolidObjectDataset")),
        ("wild", build_sir2(rs / "SIR2/WildSceneDataset")),
        ("nature20", build_paired_by_stem(rs / "nature20")),
        ("real20", build_paired_by_stem(rs / "real20")),
        ("real89", build_paired_by_stem(OTHERS / "real89/real89")),
    ]:
        print(f"{name:10s} {len(s):6d} pairs")