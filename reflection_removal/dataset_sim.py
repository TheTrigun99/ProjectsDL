"""
SIRR data: one contract for the simulated NPZ and for the real image folders.

Every dataset yields:
    mixed         float32 CHW in [0, 1], display sRGB
    transmission  idem
    source        str, which dataset it came from
    name          str, the file stem (traceability)

Only the transmission is supervised, so no reflection layer anywhere.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import (ConcatDataset, DataLoader, Dataset, Subset,
                              WeightedRandomSampler)

cv2.setNumThreads(0)          # one thread per worker, not one pool per worker

OTHERS = Path("/home/damien/data_others")
SIM = Path("/home/damien/data_fivek_dng/simulated")
SIM_NM = Path("/home/damien/data_fivek_dng/simulated_not_modified")
SIM_biased = Path("/home/damien/data_fivek_dng/simulated_biaised1")
SIM_biased2 = Path("/home/damien/data_fivek_dng/simulated_biaised2")
SIM_big = Path("/home/damien/data_fivek_dng/simulated_big_biaised2")
IMG_EXT = {".png", ".jpg", ".jpeg"}


# ---------------------------------------------------------------------------
# JPEG round trip
# ---------------------------------------------------------------------------

_SAMPLING = {n: getattr(cv2, f"IMWRITE_JPEG_SAMPLING_FACTOR_{n}")
             for n in ("444", "440", "422", "420")}


@dataclass
class JpegAug:
    """One quality/sampling draw per sample, shared by both layers but encoded
    separately: in a real benchmark M and T are two distinct photographs, so
    their artefacts must not cancel each other."""
    quality: tuple[int, int] = (90, 100)
    sampling: tuple[str, ...] = ("444", "444", "444", "422", "420")

    def draw(self, rnd):
        return {"quality": rnd.randint(*self.quality),
                "sampling": rnd.choice(self.sampling)}


def jpeg_roundtrip(rgb_u8, quality, sampling):
    """Encode/decode in memory. cv2 assumes BGR: the swap is what puts the luma
    weights and the chroma subsampling on the right channels."""
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(rgb_u8[..., ::-1]), [
        cv2.IMWRITE_JPEG_QUALITY, int(quality),
        cv2.IMWRITE_JPEG_SAMPLING_FACTOR, _SAMPLING[sampling]])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1] if ok else rgb_u8

def to_tensor(rgb_u8):
    return torch.from_numpy(
        np.ascontiguousarray(rgb_u8.transpose(2, 0, 1))).float().div_(255.0)


# ---------------------------------------------------------------------------
# Geometry: the same policy the simulation itself used
# ---------------------------------------------------------------------------

def square_resize(img, size, rnd=None, frac=(0.4, 1.0)):
    """Square crop of the frame, then downscale to size x size.

    This is random_crop() from data_clean.ipynb, applied to real photos. It is
    the only policy that keeps real and simulated data at the same scale: the
    simulation renders a whole camera frame (FOV 40-90 deg) onto a size x size
    grid, so defocus and ghost offsets are expressed in *frame* pixels. Taking
    a 224-pixel crop out of a 3456-pixel photo would instead be a ~5 deg field
    of view, where the same blur spans ten times more pixels.

    rnd=None gives the deterministic center crop of the full short side.
    """
    h, w = img.shape[:2]
    if rnd is None:
        c = min(h, w)
        y, x = (h - c) // 2, (w - c) // 2
    else:
        c = max(int(min(h, w) * rnd.uniform(*frac)), min(size, h, w))
        y, x = rnd.randint(0, h - c), rnd.randint(0, w - c)
    crop = img[y:y + c, x:x + c]
    if c == size:
        return crop
    interp = cv2.INTER_AREA if c > size else cv2.INTER_LINEAR
    return cv2.resize(crop, (size, size), interpolation=interp)


def pad_to_multiple(img, m):
    h, w = img.shape[:2]
    ph, pw = (-h) % m, (-w) % m
    return img if not (ph or pw) else cv2.copyMakeBorder(
        img, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)


# ---------------------------------------------------------------------------
# Source 1: simulated NPZ, stored linear
# ---------------------------------------------------------------------------

class SimulatedNpzDataset(Dataset):
    """Renders on the fly, so changing the ISP or the compression policy costs
    a re-read instead of regenerating the 5000 files.

    finish    : acr_tone.finish(lin_srgb, ev=0.0, tone=True) -> uint8 RGB
    ev_jitter : exposure jitter in stops, drawn once per sample and applied to
                both layers identically. 0.0 disables it.
    jpeg      : compression policy, always applied. The shared default
                instance is safe: JpegAug holds only tuples and draw() takes
                the RNG as an argument.
    seed      : None -> the global RNG (training). An int makes each index
                reproducible, for a fixed validation split.
    """

    def __init__(self, npz_dir=SIM, finish=None, *, tone=False, ev_jitter=(0, 0),
                 jpeg: JpegAug = JpegAug(), seed=None, source="sim", t_key="t"):
        self.paths = sorted(Path(npz_dir).glob("ex_*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"no ex_*.npz under {npz_dir}")
        self.finish, self.tone = finish, tone
        self.jpeg = jpeg
        if ev_jitter is not None:
            self.ev_j = True
            self.ev_j_min, self.ev_j_max = ev_jitter
        else:
            self.ev_j = False
        self.seed, self.source = seed, source
        # quelle cle du npz sert de transmission supervisee. "t" = la couche
        # telle que composee ; un autre nom (ex. "t_true") permet de comparer
        # deux definitions de la cible sans regenerer les fichiers.
        self.t_key = t_key

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        rnd = random.Random(self.seed + i) if self.seed is not None else random

        with np.load(path) as z:
            if self.t_key not in z:
                raise KeyError(
                    f"{path.name} ne contient pas '{self.t_key}' "
                    f"(cles presentes : {list(z.keys())})")
            lin = {"mixed": z["m"].astype(np.float32),
                   "transmission": z[self.t_key].astype(np.float32)}

        # one render setting for both layers: the ISP must not add a difference
        # between M and T that the compositing did not put there
        if self.ev_j:
            ev = rnd.uniform(self.ev_j_min, self.ev_j_max)
        else:
             ev = 0
        img = {k: self.finish(v, ev=ev, tone=self.tone) for k, v in lin.items()}

        # always compressed: the benchmarks are JPEG, a pristine simulation
        # would be a domain gap we create ourselves
        cfg = self.jpeg.draw(rnd)
        img = {k: jpeg_roundtrip(v, **cfg) for k, v in img.items()}

        out = {k: to_tensor(v) for k, v in img.items()}
        out["source"], out["name"] = self.source, path.stem
        return out


# ---------------------------------------------------------------------------
# Source 2: real photographs already on disk
# ---------------------------------------------------------------------------

class PairedImageDataset(Dataset):
    """size=int  -> square_resize to that size (training, and eval at the scale
                    the model was trained on).
       size=None -> native resolution, padded to a multiple of pad_multiple.
                    Use batch_size=1: the images have different shapes.
       frac      -> fraction of the short side the square crop covers.
                    (1.0, 1.0) keeps the whole field of view and only rescales,
                    which is what you want for photos that are already small."""

    def __init__(self, samples, *, size=224, train=True, jpeg=None,
                 frac=(0.4, 1.0), pad_multiple=32, seed=None, source=""):
        if not samples:
            raise ValueError(f"empty sample list for {source!r}")
        self.samples, self.size, self.train = samples, size, train
        self.jpeg, self.frac, self.pad_multiple = jpeg, frac, pad_multiple
        self.seed, self.source = seed, source

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        rec = self.samples[i]
        rnd = random.Random(self.seed + i) if self.seed is not None else random

        img = {}
        for k in ("mixed", "transmission"):
            a = cv2.imread(str(rec[k]), cv2.IMREAD_COLOR)
            if a is None:
                raise FileNotFoundError(rec[k])
            img[k] = a[..., ::-1]                      # BGR -> RGB, like finish()

        if self.size is not None:
            # a fresh RNG *reseeded identically* per layer: sharing one object
            # would advance the stream and crop the two layers differently
            gs = rnd.getrandbits(32) if self.train else None
            img = {k: square_resize(v, self.size,
                                    random.Random(gs) if self.train else None,
                                    self.frac)
                   for k, v in img.items()}
            if self.train and rnd.random() < 0.5:
                img = {k: v[:, ::-1] for k, v in img.items()}
        else:
            img = {k: pad_to_multiple(v, self.pad_multiple) for k, v in img.items()}

        if self.jpeg is not None:
            cfg = self.jpeg.draw(rnd)
            img = {k: jpeg_roundtrip(np.ascontiguousarray(v), **cfg)
                   for k, v in img.items()}

        out = {k: to_tensor(np.ascontiguousarray(v)) for k, v in img.items()}
        out["source"], out["name"] = self.source, rec["mixed"].stem
        return out


def _index(folder):
    """stem -> path. Lookup by stem because SIR2 mixes .png and .jpg."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    return {p.stem: p for p in sorted(folder.iterdir())
            if p.suffix.lower() in IMG_EXT}


def build_by_stem(root, blended="blended", transmission="transmission_layer"):
    """nature_dataset, nature20, real20, real89: identical stems in both folders."""
    root = Path(root)
    trans = _index(root / transmission)
    return [{"mixed": m, "transmission": trans[s]}
            for s, m in _index(root / blended).items() if s in trans]


def build_voc(root=OTHERS / "VOC2012"):
    """blended/2007_000032-2008_006587.png -> transmission_layer/2007_000032.png
    VOC ids contain no hyphen, so splitting on the first one is safe.
    (reflection_layer/ is NOT aligned with blended/ -- it keeps the uncropped
    source image. Another reason not to supervise R here.)"""
    root = Path(root)
    trans = _index(root / "transmission_layer")
    out = []
    for stem, m in _index(root / "blended").items():
        t = stem.partition("-")[0]
        if t in trans:
            out.append({"mixed": m, "transmission": trans[t]})
    return out


def build_sir2(root):
    """ab-10-m-3 -> ab-10-g-3 | 1-Focus-11-m -> 1-Focus-11-g | 1-m -> 1-g
    The tag is not at a fixed position: replace the one standalone 'm'."""
    root = Path(root)
    trans = _index(root / "transmission_layer")
    out = []
    for stem, m in _index(root / "blended").items():
        parts = stem.split("-")
        hits = [i for i, p in enumerate(parts) if p == "m"]
        if len(hits) != 1:
            continue
        parts[hits[0]] = "g"
        t = "-".join(parts)
        if t in trans:
            out.append({"mixed": m, "transmission": trans[t]})
    return out


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def worker_init(worker_id):
    cv2.setNumThreads(0)


def _nature():
    return build_by_stem(OTHERS / "nature_dataset", "natural_I", "natural_T")


def build_train_loader(finish, batch_size=16, size=224, workers=8,
                       weights=(0.6, 0.1, 0.1), jpeg=None,simulated=False, **kw):
    """weights = (simulated, VOC, nature), as sampling probabilities rather
    than dataset sizes: nature has 200 pairs against 7643 for VOC, plain
    concatenation would show it 2% of the time."""
    jpeg = JpegAug() if jpeg is None else jpeg
    if simulated:
        sets = [
        SimulatedNpzDataset(finish=finish, jpeg=jpeg, source="sim", **kw),
        PairedImageDataset(_nature(), size=size, jpeg=jpeg,
                            source="nature"),
            PairedImageDataset(build_by_stem(OTHERS / "real89/real89"), size=size,
                           jpeg=jpeg, source="real89"),                            
        ]
    else:
        sets = [
            # Subset, not [:5000]: slicing a Dataset calls __getitem__ with a
            # slice object, which lands in cv2.imread as a key
            Subset(PairedImageDataset(build_voc(), size=size, jpeg=jpeg
                                      ,frac=(1.0,1.0), source="voc"),
                   range(5000)),
            PairedImageDataset(_nature(), size=size, jpeg=jpeg,
                                source="nature"),
            PairedImageDataset(build_by_stem(OTHERS / "real89/real89"), size=size,
                           jpeg=jpeg, source="real89"),

        ]


    weights = weights[:len(sets)] 
    w = [p / len(d) for d, p in zip(sets, weights) for _ in range(len(d))]
    return DataLoader(
        ConcatDataset(sets), batch_size=batch_size,
        sampler=WeightedRandomSampler(torch.tensor(w, dtype=torch.double),
                                      sum(len(d) for d in sets), replacement=True),
        num_workers=workers, pin_memory=True, drop_last=True,
        persistent_workers=workers > 0, worker_init_fn=worker_init)


def eval_specs():
    rs = OTHERS / "robustsirr_test_dataset"
    return {"postcard": build_sir2(rs / "SIR2/PostcardDataset"),
            "solid":    build_sir2(rs / "SIR2/SolidObjectDataset"),
            "wild":     build_sir2(rs / "SIR2/WildSceneDataset"),
            "nature20": build_by_stem(rs / "nature20"),
            "real20":   build_by_stem(rs / "real20"),}


def build_eval_loaders(size=224, workers=4):
    """No JPEG augmentation: these files are already compressed, re-encoding
    would stack a second generation of artefacts on top.
    size=None evaluates at native resolution -- comparable with published
    numbers, but real20 goes up to 2304x3456."""
    return {name: DataLoader(
                PairedImageDataset(s, size=size, train=False, jpeg=None,
                                   source=name),
                batch_size=1 if size is None else 8, shuffle=False,
                num_workers=workers, worker_init_fn=worker_init)
            for name, s in eval_specs().items()}


if __name__ == "__main__":
    from acr_tone import finish

    print(f"{'sim':10s} {len(SimulatedNpzDataset(SIM, finish, )):6d}")
    print(f"{'voc':10s} {len(build_voc()[:5000]):6d}")
    print(f"{'nature':10s} "
          f"{len(build_by_stem(OTHERS / 'nature_dataset', 'natural_I', 'natural_T')):6d}")
    for n, s in eval_specs().items():
        print(f"{n:10s} {len(s):6d}")

    b = next(iter(build_train_loader(finish, batch_size=8, workers=2)))
    print("\ntrain batch:", {k: tuple(v.shape) if torch.is_tensor(v) else v[:3]
                             for k, v in b.items()})
    for n, dl in build_eval_loaders(size=224, workers=0).items():
        e = next(iter(dl))
        print(f"eval {n:9s} mixed {tuple(e['mixed'].shape)}")
