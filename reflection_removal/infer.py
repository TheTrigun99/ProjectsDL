"""Recharge les poids d'un run et affiche mixture / modèle / vérité.

    python3 infer.py SM_big_biased_seed_2_64                  # 1 image par jeu de test
    python3 infer.py SM_big_biased_seed_2_64 --set real20 -n 4
    python3 infer.py SM_big_biased_seed_2_64 --gpu 6          # sinon CPU

Sort runs/<nom>/preview.png. La taille d'éval est relue depuis le ckpt :
un run entraîné en 384 est évalué en 384, sinon les chiffres ne veulent rien dire.
"""
import argparse, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/damien/ProjectsDL/reflection_removal")

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--set", dest="sets", action="append", help="par défaut : les 5")
p.add_argument("-n", type=int, default=1, help="images par jeu")
p.add_argument("--gpu", default=None)
p.add_argument("--out", default=None)
p.add_argument("--name", default="preview")
a = p.parse_args()

import os
if a.gpu: os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
dev = "cuda" if a.gpu else "cpu"

from dataset_sim import build_eval_loaders
from models.unet_simple import Unet

ck = torch.load(f"runs/{a.run}/ckpt.pt", map_location="cpu", weights_only=False)
c = ck["config"]
net = Unet(dim=c["dim"], dim_mults=tuple(c["dim_mults"]), channel=c["channel"])
net.load_state_dict(ck["model"]); net.eval().to(dev)
print(f"{a.run} : it {ck['iter']}, dim {c['dim']}, éval en {c['size']}px, device {dev}")

loaders = build_eval_loaders(size=c["size"], workers=0)
names = a.sets or list(loaders)
rows = []
with torch.no_grad():
    for nm in names:
        for i, b in enumerate(loaders[nm]):
            if i >= a.n: break
            m, t = b["mixed"].to(dev), b["transmission"].to(dev)
            pred = net(m).clamp(0, 1).float()
            for j in range(min(len(m), 1)):
                psnr = lambda x, y: 10 * np.log10(1 / max(float(((x - y) ** 2).mean()), 1e-10))
                mm, pp, tt = (v[j].cpu().numpy().transpose(1, 2, 0) for v in (m, pred, t))
                rows.append((f"{nm}/{b['name'][j]}", mm, pp, tt, psnr(mm, tt), psnr(pp, tt)))

fig, ax = plt.subplots(len(rows), 3, figsize=(10.5, 3.5 * len(rows)), squeeze=False)
for r, (nm, mm, pp, tt, p0, p1) in enumerate(rows):
    for k, (img, ttl) in enumerate([(mm, f"mixture  {p0:.2f} dB"),
                                    (pp, f"modèle  {p1:.2f} dB  ({p1-p0:+.2f})"),
                                    (tt, "vérité")]):
        ax[r][k].imshow(img); ax[r][k].axis("off")
        ax[r][k].set_title(ttl, fontsize=9)
    ax[r][0].set_ylabel(nm)
    ax[r][0].text(-0.04, 0.5, nm, rotation=90, va="center", ha="right",
                  fontsize=9, transform=ax[r][0].transAxes)
out = a.out or f"runs/{a.run}/{a.name}.png"
fig.tight_layout(); fig.savefig(out, dpi=110, bbox_inches="tight")
print("->", out)
