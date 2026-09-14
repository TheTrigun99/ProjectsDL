"""Ecart de distribution entre les donnees d'entrainement et les benchmarks reels.

    import domain_gap as dg
    P = dg.collect_all(n=400)                  # toutes les sources, pretraitement commun
    P = dg.from_loaders(eval_loaders, data)    # ou directement les loaders du notebook
    dg.report(P)                               # tableau des distances
    dg.plot(P, "domain_gap.png")               # ACP + histogrammes

Trois decisions sont cablees ici, parce qu'elles changent la reponse :

1. ESTIMATEUR. real20 et nature20 font 20 images. La FID est un estimateur
   biaise qui demande n >~ 2000 : a n=20 elle mesure surtout la taille de
   l'echantillon. On utilise donc MMD^2 non biaisee (l'estimateur de la KID)
   plus un test de permutation, qui reste honnete a petit n et rend une
   p-valeur au lieu d'un nombre non calibre.

2. CONFusION CONTENU / APPARENCE. SIR2 est du studio (cartes postales, objets
   poses), MIT5K est du paysage et du portrait. N'importe quelle distance sur
   des features apprises sera dominee par le CONTENU, que la simulation ne peut
   pas corriger et n'a pas a corriger. D'ou deux espaces separes :
     - `lowlevel` : statistiques d'apparence, robustes au contenu, et surtout
       lisibles une par une. C'est celui qui dit quoi changer.
     - `vgg` / `vae` : features apprises, pour la vue d'ensemble. A lire en
       relatif seulement.

3. REFERENCE. « Mes donnees sont-elles loin du reel ? » n'a pas de reponse dans
   l'absolu : tout est loin. La question utile est « plus loin que VOC ? »,
   puisque VOC est le melange sur lequel l'entrainement tient. VOC est donc
   toujours dans le tableau, comme etalon.

Et une precaution : les eval loaders font un crop central sans JPEG, le loader
d'entrainement fait un crop aleatoire avec JPEG et ev_jitter. Les comparer tels
quels mesure la propre augmentation, pas l'ecart de domaine. Par defaut
collect_all donne a tout le monde le pretraitement des benchmarks (crop central,
pas de jitter ; JPEG q100 4:4:4 sur le simule, que SimulatedNpzDataset ne permet
pas d'enlever) ; match_train_pipeline=True et from_loaders(eval_loaders, data)
donnent l'autre mesure, celle que le modele voit.
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch

from torch.utils.data import ConcatDataset, DataLoader, Subset

from acr_tone import finish
from dataset_sim import (OTHERS, JpegAug, PairedImageDataset,
                         SimulatedNpzDataset, build_by_stem,
                         build_eval_loaders, build_train_loader, build_voc,
                         worker_init)

cv2.setNumThreads(0)

SIZE = 384
SIM_DIRS = {
    "sim 150k":  ("/home/damien/data_fivek_dng/dataset_final_test1", "true_t"),
    "sim 10k":   ("/home/damien/data_fivek_dng/dataset_soomin_final_biased", "true_t"),
}


# ---------------------------------------------------------------------------
# Lecture : directement par les Dataset / DataLoader de dataset_sim
# ---------------------------------------------------------------------------
# Tout passe par les classes qui nourrissent l'entrainement et l'evaluation :
# la mesure porte sur exactement ce que le modele voit, sans reimplementation
# qui pourrait deriver. Les tenseurs sortent en float [0,1] CHW ; ils viennent
# d'un uint8 / 255, donc le retour en uint8 est exact.

def _to_u8(x):
    return (x.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)


def _sources(ds):
    """Les `source` presentes dans un Dataset, a travers Concat et Subset."""
    if isinstance(ds, ConcatDataset):
        return set().union(*(_sources(d) for d in ds.datasets))
    if isinstance(ds, Subset):
        return _sources(ds.dataset)
    return {getattr(ds, "source", None)}


def collect_loader(loader, n=None, rename=None, keep=None, max_batches=None):
    """Itere un DataLoader et range les paires par batch["source"].

    n=None prend tout (benchmarks). Avec n, s'arrete des que chaque source du
    dataset en a n. max_batches borne le cout d'un loader desequilibre : avec
    weights=(0.96, 0.02, 0.02), 400 `nature` demandent ~20 000 tirages."""
    rename = rename or {}
    want = {rename.get(s, s) for s in _sources(loader.dataset)}
    if keep is not None:
        want &= set(keep)
    out = {}
    for b, batch in enumerate(loader):
        for M, T, src in zip(batch["mixed"], batch["transmission"], batch["source"]):
            src = rename.get(src, src)
            if src not in want:
                continue
            lst = out.setdefault(src, [])
            if n is None or len(lst) < n:
                lst.append((_to_u8(M), _to_u8(T)))
        if n is not None and all(len(out.get(s, [])) >= n for s in want):
            break
        if max_batches is not None and b + 1 >= max_batches:
            short = {s: len(out.get(s, [])) for s in want if len(out.get(s, [])) < n}
            if short:
                print(f"[domain_gap] max_batches atteint, sources incompletes : {short}")
            break
    return out


def from_loaders(eval_loaders, train_loaders=(), n=400, max_batches=5000):
    """Les loaders du notebook, tels quels :

        P = dg.from_loaders(eval_loaders, data, n=400)

    eval_loaders  : dict nom -> DataLoader, lus en entier (520 images en tout).
    train_loaders : un DataLoader ou une liste ; les paires sont rangees par
                    source, avec le pretraitement d'entrainement (crop
                    aleatoire, JPEG, ev_jitter) puisque c'est le loader lui-meme."""
    P = {}
    if isinstance(train_loaders, DataLoader):
        train_loaders = [train_loaders]
    for dl in train_loaders:
        P.update(collect_loader(dl, n=n, max_batches=max_batches))
    for name, dl in eval_loaders.items():
        got = collect_loader(dl)
        P[f"EVAL {name}"] = next(iter(got.values()))
    return P


def _subset_loader(ds, n, rnd, workers):
    idx = rnd.sample(range(len(ds)), min(n, len(ds)))
    return DataLoader(Subset(ds, idx), batch_size=16, shuffle=False,
                      num_workers=workers, worker_init_fn=worker_init)


def collect_all(n=400, seed=0, match_train_pipeline=False, workers=8):
    """{nom: [(M, T), ...]} pour chaque source d'entrainement et chaque benchmark.

    match_train_pipeline=False : le pretraitement des benchmarks pour tout le
        monde (crop central, pas de jitter). Le simule passe quand meme par un
        JPEG q100 4:4:4, le minimum que SimulatedNpzDataset autorise.
    match_train_pipeline=True : build_train_loader lui-meme, donc crop
        aleatoire, JPEG et ev_jitter exactement comme a l'entrainement.
    Les benchmarks viennent toujours de build_eval_loaders, lus en entier."""
    rnd = random.Random(seed)
    P = {}
    if match_train_pipeline:
        common = dict(batch_size=16, workers=workers, size=SIZE, weights=(1, 1, 1),
                      tone=True, ev_jitter=(-1.9, 0.9))
        first = True
        for lab, (d, k) in SIM_DIRS.items():
            if not Path(d).is_dir():
                continue
            dl = build_train_loader(finish, simulated=True, npz_dir=d, t_key=k, **common)
            keep = {lab, "nature", "real89"} if first else {lab}
            P.update(collect_loader(dl, n=n, rename={"sim": lab}, keep=keep))
            first = False
            del dl
        dl = build_train_loader(finish, simulated=False, **common)
        P.update(collect_loader(dl, n=n, rename={"voc": "VOC"}, keep={"VOC"}))
        del dl
    else:
        q100 = JpegAug(quality=(100, 100), sampling=("444",))
        for lab, (d, k) in SIM_DIRS.items():
            if Path(d).is_dir():
                ds = SimulatedNpzDataset(d, finish, tone=True, ev_jitter=None,
                                         jpeg=q100, t_key=k, source=lab)
                P.update(collect_loader(_subset_loader(ds, n, rnd, workers)))
        for lab, samples in [("VOC", build_voc()),
                             ("nature", build_by_stem(OTHERS / "nature_dataset",
                                                      "natural_I", "natural_T")),
                             ("real89", build_by_stem(OTHERS / "real89/real89"))]:
            ds = PairedImageDataset(samples, size=SIZE, train=False, jpeg=None, source=lab)
            P.update(collect_loader(_subset_loader(ds, n, rnd, workers)))
    P.update(from_loaders(build_eval_loaders(size=SIZE, workers=workers)))
    return P


# ---------------------------------------------------------------------------
# Espace 1 : statistiques d'apparence, lisibles une par une
# ---------------------------------------------------------------------------

_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def _radial_slope(y):
    """Pente log-log du spectre de puissance radial. Mesure la repartition
    grossier/fin : une pente plus raide = image plus douce."""
    f = np.abs(np.fft.rfft2(y - y.mean())) ** 2
    h, w = f.shape
    fy = np.fft.fftfreq(y.shape[0])[:, None]
    fx = np.fft.rfftfreq(y.shape[1])[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2).ravel()
    p = f.ravel()
    keep = (r > 1e-3) & (p > 0)
    if keep.sum() < 32:
        return 0.0
    lr, lp = np.log(r[keep]), np.log(p[keep])
    return float(np.polyfit(lr, lp, 1)[0])


def _stats_one(m_u8, t_u8):
    m = m_u8.astype(np.float32) / 255.0
    t = t_u8.astype(np.float32) / 255.0
    r = m - t                                   # le residu : ce que le modele doit enlever
    Ym, Yt = m @ _LUMA, t @ _LUMA
    mx, mn = m.max(-1), m.min(-1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    gy, gx = np.diff(Ym, axis=0), np.diff(Ym, axis=1)
    absr = np.abs(r)
    rc = r - r.mean()
    return {
        # --- apparence du melange
        "Y moyen":        float(Ym.mean()),
        "Y ecart-type":   float(Ym.std()),
        "saturation":     float(sat.mean()),
        "frac crame":     float((mx >= 254 / 255).mean()),
        "frac noir":      float((mx <= 2 / 255).mean()),
        "energie grad":   float(np.abs(gx).mean() + np.abs(gy).mean()),
        "pente spectre":  _radial_slope(Ym),
        # --- le residu M-T, c'est lui qui est simule
        "|R| moyen":      float(absr.mean()),
        "|R| p99":        float(np.percentile(absr, 99)),
        "R kurtosis":     float((rc ** 4).mean() / max((rc ** 2).mean() ** 2, 1e-12)),
        "frac |R|>0.05":  float((absr.max(-1) > 0.05).mean()),
        "R pente spec":   _radial_slope(r @ _LUMA),
        "R vs T contr.":  float(r.std() / max(t.std(), 1e-6)),
        "frac T>M":       float((t > m + 1 / 255).mean()),
    }


def lowlevel(pairs):
    rows = [_stats_one(m, t) for m, t in pairs]
    keys = list(rows[0])
    return np.array([[r[k] for k in keys] for r in rows], np.float64), keys


# ---------------------------------------------------------------------------
# Espace 2 : features apprises (VGG19 deja en cache, ou le VAE FLUX)
# ---------------------------------------------------------------------------

def vgg_features(pairs, device="cuda", layer=20, batch=16):
    """Moyenne spatiale de relu4_1. VGG est deja le backbone de la loss
    perceptuelle, donc cet espace est litteralement celui que la loss regarde."""
    from torchvision.models import VGG19_Weights, vgg19
    net = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:layer + 1]
    net = net.eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch):
            x = np.stack([m for m, _ in pairs[i:i + batch]])
            x = torch.from_numpy(x).permute(0, 3, 1, 2).float().div_(255).to(device)
            out.append(net((x - mean) / std).mean((2, 3)).cpu().numpy())
    del net
    torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float64), [f"vgg{i}" for i in range(out[0].shape[1])]


def vae_features(pairs, vae, device="cuda", batch=8):
    """Latents du VAE, moyennes spatialement. A n'utiliser que si le modele
    consomme vraiment ces latents : le VAE est quasi inversible, donc ses
    latents gardent le contenu et la distance sera dominee par lui."""
    out = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch):
            x = np.stack([m for m, _ in pairs[i:i + batch]])
            x = torch.from_numpy(x).permute(0, 3, 1, 2).float().div_(255)
            x = (x * 2 - 1).to(device, dtype=vae.dtype)
            z = vae.encode(x).latent_dist.mean
            out.append(z.float().mean((2, 3)).cpu().numpy())
    return np.concatenate(out).astype(np.float64), None


# ---------------------------------------------------------------------------
# Distance : MMD^2 non biaisee + test de permutation
# ---------------------------------------------------------------------------

def _rbf(X, Y, gamma):
    d = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    return np.exp(-gamma * d)


def mmd2(X, Y, gamma=None):
    """Estimateur non biaise de MMD^2 (celui de la KID). Vaut ~0 si X et Y
    viennent de la meme loi, quel que soit n. Peut etre legerement negatif."""
    n, m = len(X), len(Y)
    if n < 2 or m < 2:
        return float("nan"), 1.0
    if gamma is None:                       # heuristique de la mediane
        Z = np.concatenate([X, Y])
        d = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
        med = np.median(d[d > 0]) if (d > 0).any() else 1.0
        gamma = 1.0 / max(med, 1e-12)
    Kxx, Kyy, Kxy = _rbf(X, X, gamma), _rbf(Y, Y, gamma), _rbf(X, Y, gamma)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    return float(Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1))
                 - 2 * Kxy.mean()), gamma


def mmd2_test(X, Y, n_perm=200, seed=0):
    """Rend (MMD^2, p-valeur). p petite = les deux echantillons sont
    distinguables ; p ~ 0.5 = indiscernables a cette taille d'echantillon."""
    obs, gamma = mmd2(X, Y)
    Z = np.concatenate([X, Y])
    n = len(X)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(Z)
        s, _ = mmd2(Z[:n], Z[n:], gamma)
        ge += s >= obs
    return obs, (ge + 1) / (n_perm + 1)


def _standardize(F, ref, robust=False, clip=None):
    """Centre-reduit sur la reference : sans ca, une seule statistique a grande
    echelle (la pente du spectre) ecrase toutes les autres dans la distance.

    robust=True utilise mediane / IQR au lieu de moyenne / ecart-type. Necessaire
    des qu'une statistique est quasi constante sur la reference : `frac noir`
    vaut ~0 sur tous les benchmarks, son ecart-type est minuscule, et la version
    naive envoie les quelques VOC concernes a +40 sigma -- ce qui ecrase toute
    figure. clip borne ensuite les queues."""
    if robust:
        mu = np.median(ref, 0)
        q1, q3 = np.percentile(ref, [25, 75], axis=0)
        sd = (q3 - q1) / 1.349
        sd = np.where(sd < 1e-6, np.maximum(ref.std(0), 1e-6), sd)
    else:
        mu, sd = ref.mean(0), ref.std(0)
        sd = np.where(sd < 1e-9, 1.0, sd)
    Z = (F - mu) / sd
    return np.clip(Z, -clip, clip) if clip else Z


# ---------------------------------------------------------------------------
# Sorties
# ---------------------------------------------------------------------------

def report(P, space="lowlevel", n_perm=200, vae=None, device="cuda"):
    extract = {"lowlevel": lambda p: lowlevel(p),
               "vgg": lambda p: vgg_features(p, device),
               "vae": lambda p: vae_features(p, vae, device)}[space]
    F = {k: extract(v)[0] for k, v in P.items() if len(v) >= 4}
    keys = lowlevel(next(iter(P.values())))[1] if space == "lowlevel" else None

    evals = [k for k in F if k.startswith("EVAL")]
    trains = [k for k in F if not k.startswith("EVAL")]
    ref = np.concatenate([F[k] for k in evals])          # reference = tout le reel de test

    print(f"=== MMD^2 vers chaque benchmark   (espace: {space})")
    print(f"{'source':12s} " + " ".join(f"{e.replace('EVAL ',''):>12s}" for e in evals)
          + f" {'moyenne':>10s}")
    rows = {}
    for t in trains:
        vals = []
        for e in evals:
            Xs = _standardize(F[t], ref)
            Ys = _standardize(F[e], ref)
            d, p = mmd2_test(Xs, Ys, n_perm)
            vals.append((d, p))
        rows[t] = vals
        cells = " ".join(f"{d:8.3f}{'*' if p < 0.05 else ' '}{'':3s}" for d, p in vals)
        print(f"{t:12s} " + cells + f" {np.mean([d for d, _ in vals]):10.3f}")
    print("  (* = p<0.05 au test de permutation : distinguable du benchmark)")

    if keys is not None:
        print(f"\n=== Ecart par statistique, en ecarts-types du reel de test")
        print(f"{'statistique':16s} " + " ".join(f"{t:>10s}" for t in trains))
        mu, sd = ref.mean(0), np.where(ref.std(0) < 1e-9, 1.0, ref.std(0))
        for i, k in enumerate(keys):
            z = [(F[t][:, i].mean() - mu[i]) / sd[i] for t in trains]
            print(f"{k:16s} " + " ".join(f"{v:+10.2f}" for v in z))
        print("  (0 = aligne sur les benchmarks ; |z|>1 = franchement decale)")
    return rows


def plot(P, path="domain_gap.png", space="lowlevel", vae=None, device="cuda", dpi=130):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extract = {"lowlevel": lambda p: lowlevel(p),
               "vgg": lambda p: vgg_features(p, device),
               "vae": lambda p: vae_features(p, vae, device)}[space]
    F, keys = {}, None
    for k, v in P.items():
        if len(v) >= 4:
            F[k], ks = extract(v)
            keys = ks or keys

    evals = [k for k in F if k.startswith("EVAL")]
    ref = np.concatenate([F[k] for k in evals])
    S = {k: _standardize(v, ref, robust=True, clip=8) for k, v in F.items()}

    # ACP sur les benchmarks : les axes sont ceux du reel, les sources s'y projettent
    R = _standardize(ref, ref, robust=True, clip=8)
    U, s, Vt = np.linalg.svd(R - R.mean(0), full_matrices=False)
    W = Vt[:2].T

    ncol = 4
    show = keys[:12] if keys else []
    fig = plt.figure(figsize=(4.2 * ncol, 4.0 * (1 + int(np.ceil(len(show) / ncol)))), dpi=dpi)
    ax = fig.add_subplot(int(np.ceil(len(show) / ncol)) + 1, 1, 1)
    cmap = plt.get_cmap("tab10")
    for i, (k, v) in enumerate(S.items()):
        p = (v - R.mean(0)) @ W
        real = k.startswith("EVAL")
        ax.scatter(p[:, 0], p[:, 1], s=26 if real else 10,
                   marker="o" if real else "x", alpha=.75 if real else .45,
                   color=cmap(i % 10), label=k, zorder=3 if real else 2)
    ax.set_title("ACP des statistiques d'apparence — axes definis par les benchmarks reels\n"
                 "ronds = reel de test, croix = sources d'entrainement")
    ax.legend(fontsize=8, ncol=4, loc="best")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(alpha=.2)

    for j, k in enumerate(show):
        a = fig.add_subplot(int(np.ceil(len(show) / ncol)) + 1, ncol, ncol + j + 1)
        i = keys.index(k)
        lo = min(v[:, i].min() for v in F.values())
        hi = max(np.percentile(v[:, i], 99) for v in F.values())
        bins = np.linspace(lo, hi, 40)
        for n, (lab, v) in enumerate(F.items()):
            a.hist(v[:, i], bins=bins, density=True, histtype="step", lw=2.0 if lab.startswith("EVAL") else 1.2,
                   ls="-" if lab.startswith("EVAL") else "--", color=cmap(n % 10), label=lab)
        a.set_title(k, fontsize=10)
        a.tick_params(labelsize=7)
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"ecrit: {path}")
    return fig


if __name__ == "__main__":
    P = collect_all(n=200)
    for k, v in P.items():
        print(f"  {k:16s} {len(v):4d} paires")
    report(P)
    plot(P)
