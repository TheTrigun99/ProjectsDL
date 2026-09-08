---
tags:
  - sirr
  - sweep
  - datasets
date: 2026-09-08
---

# Caractéristiques des datasets

Retour à [[Sweep 2026-09-08]].

## Comment est mesurée la force du reflet

Il n'existe pas d'étiquette « reflet fort » dans ces jeux. Je la dérive du **PSNR entre la mixture M et la transmission de référence T** : plus la mixture est loin de la transmission, plus le reflet pèse. C'est exactement la quantité que le modèle doit améliorer, donc les seuils sont directement comparables aux colonnes de [[Sweep 2026-09-08 - Résultats]].

| Bande | Lecture |
| --- | --- |
| < 18 dB | reflet fort |
| 18 – 22 dB | modéré |
| 22 – 28 dB | faible |
| > 28 dB | très faible |

Tout est calculé dans le **domaine d'affichage** que voit le modèle : crop carré centré redimensionné en 224x224 pour les photos, rendu ACR (`tone=True`, `ev=0`) pour les `.npz`. Le SSIM est celui de `utils/ssim_ref.py`, réimplémenté en numpy et vérifié identique à la version torch à 6e-8 près.

## Jeux d'entraînement

### Force du reflet

| Jeu | n | PSNR moy | méd | p10 | p90 | SSIM |
| --- | --- | --- | --- | --- | --- | --- |
| `simulated` | 5000 | 17.33 | 17.18 | 13.67 | 21.34 | 0.804 |
| `simulated_big_biaised2` | 15000 | 17.32 | 17.15 | 13.63 | 21.36 | 0.804 |
| `simulated_biaised1` → `t` | 5000 | 16.34 | 15.95 | 12.31 | 21.05 | 0.771 |
| `simulated_biaised2` → `t` | 5000 | 16.30 | 15.90 | 12.26 | 21.02 | 0.771 |
| `simulated_biaised2` → `true_t` | 5000 | 17.20 | 16.76 | 12.73 | 22.40 | 0.779 |
| `simulated_not_modified` → `true_t` | 5000 | 15.60 | 15.61 | 10.43 | 20.72 | 0.741 |
| `VOC2012` (5 000 premières) | 5000 | 14.20 | 13.71 | 10.01 | 18.86 | 0.747 |
| `nature_dataset` | 200 | 24.09 | 24.17 | 17.92 | 29.88 | 0.837 |
| `real89` | 89 | 19.30 | 17.82 | 12.94 | 26.84 | 0.764 |

### Répartition

| Jeu | fort < 18 dB | modéré 18-22 | faible 22-28 | très faible > 28 | Utilisé par |
| --- | --- | --- | --- | --- | --- |
| `simulated` | 59.7 % | 34.1 % | 6.2 % | 0.0 % | SM, SMonly, SMnotone, SMnoperc |
| `simulated_big_biaised2` | 59.5 % | 34.2 % | 6.3 % | 0.0 % | SM_big |
| `simulated_biaised1` → `t` | 68.3 % | 26.2 % | 5.5 % | 0.0 % | BIAS1 |
| `simulated_biaised2` → `t` | 69.1 % | 25.4 % | 5.5 % | 0.0 % | BIAS2 |
| `simulated_biaised2` → `true_t` | 60.6 % | 27.4 % | 11.9 % | 0.0 % | BIAS2true |
| `simulated_not_modified` → `true_t` | 71.9 % | 22.8 % | 5.4 % | 0.0 % | NM |
| `VOC2012` (5 000 premières) | 86.5 % | 9.9 % | 3.1 % | 0.5 % | VOC, VOConly |
| `nature_dataset` | 10.5 % | 17.0 % | 49.5 % | 23.0 % | 10 % de SM, VOC, BIAS*, NM |
| `real89` | 51.7 % | 19.1 % | 21.3 % | 7.9 % | 10 % de SM, VOC, BIAS*, NM |

## Jeux de test

### Force du reflet

| Jeu | n | PSNR moy | méd | p10 | p90 | SSIM |
| --- | --- | --- | --- | --- | --- | --- |
| `postcard` | 179 | 20.82 | 20.85 | 18.30 | 22.88 | 0.863 |
| `solid` | 200 | 24.02 | 23.17 | 20.68 | 28.76 | 0.890 |
| `wild` | 101 | 24.95 | 24.72 | 17.36 | 33.21 | 0.893 |
| `nature20` | 20 | 21.48 | 21.69 | 15.02 | 27.06 | 0.834 |
| `real20` | 20 | 18.82 | 17.92 | 14.72 | 23.77 | 0.745 |

### Répartition

| Jeu | fort < 18 dB | modéré 18-22 | faible 22-28 | très faible > 28 |
| --- | --- | --- | --- | --- |
| `postcard` | 5.0 % | 72.6 % | 22.3 % | 0.0 % |
| `solid` | 0.0 % | 29.5 % | 55.5 % | 15.0 % |
| `wild` | 10.9 % | 23.8 % | 34.7 % | 30.7 % |
| `nature20` | 30.0 % | 25.0 % | 35.0 % | 10.0 % |
| `real20` | 60.0 % | 15.0 % | 20.0 % | 5.0 % |

> [!important] L'entraînement et le test ne voient pas les mêmes reflets
> Les jeux simulés sont **beaucoup plus agressifs** que les benchmarks : `simulated` a 59.7 % de paires sous 18 dB, contre 5.0 % pour postcard, 0.0 % pour solid, 10.9 % pour wild. Les jeux biaisés aggravent l'écart (68-69 %). C'est l'explication directe du classement : BIAS1 et BIAS2 perdent 1.5 à 2 dB sur postcard, le jeu dont la fenêtre de reflets est la plus étroite (p10-p90 = 18.3 à 22.9 dB).

Le seul jeu d'entraînement plus dur que les simulés est **VOC2012** : PSNR médian 13.71 dB, 86.5 % des paires sous 18 dB. Le blend VOC additionne deux photos à poids à peu près égaux, sans modèle de vitre — ce n'est pas la même statistique de reflet, et c'est cohérent avec `VOConly` qui ne transfère pas.

## Énergie du reflet dans les jeux simulés

`vis` = moyenne(r) / moyenne(m) en sRGB **linéaire**, la fraction d'énergie apportée par le reflet. C'est le critère de tri de `measure()` dans `data_clean.ipynb`.

| Jeu | n | vis moy | méd | p10 | p90 | max | part > 0.40 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `simulated` | 5000 | 0.258 | 0.256 | 0.139 | 0.379 | 0.531 | 5.4 % |
| `simulated_big_biaised2` | 15000 | 0.258 | 0.256 | 0.141 | 0.379 | 0.531 | 5.2 % |
| `simulated_biaised1` | 5000 | 0.304 | 0.300 | 0.145 | 0.466 | 0.585 | 26.2 % |
| `simulated_biaised2` | 5000 | 0.306 | 0.303 | 0.147 | 0.469 | 0.591 | 26.3 % |
| `simulated_not_modified` | 5000 | 0.258 | 0.256 | 0.139 | 0.379 | 0.531 | 5.4 % |

### Écrêtage

| Jeu | pixels écrêtés | images à plus de 1 % écrêté |
| --- | --- | --- |
| `simulated` | 1.38 % | 22.4 % |
| `simulated_big_biaised2` | 1.44 % | 22.7 % |
| `simulated_biaised1` | 1.37 % | 23.1 % |
| `simulated_biaised2` | 1.37 % | 22.9 % |
| `simulated_not_modified` | 1.38 % | 22.5 % |

Deux surprises dans ce tableau, détaillées dans [[Sweep 2026-09-08 - Anomalies]] : `simulated_big_biaised2` a la distribution de `simulated` et non celle de `simulated_biaised2` malgré son nom, et le `true_t` de `simulated_not_modified` a un gain incohérent.