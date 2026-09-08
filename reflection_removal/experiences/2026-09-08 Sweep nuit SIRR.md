---
tags:
  - sirr
  - reflection-removal
  - sweep
  - resultats
date: 2026-09-08
runs: 35
gpu: 5 (file séquentielle)
---

# Sweep nuit 7 → 8 septembre 2026 — suppression de reflets

35 runs terminés, tous à 10 000 itérations, lancés en file séquentielle sur le GPU 5 entre 23:16 (07/09) et 09:19 (08/09). Aucun crash, aucun warning dans les logs, les 35 `log.txt` sont identiques aux `*.launch.log`.

> [!info] Comment lire les chiffres
> Le PSNR affiché est celui du **modèle contre la transmission de référence**. La ligne *mixture* donne le PSNR de l'image d'entrée non traitée : c'est le seuil à battre. Un modèle sous cette ligne **dégrade** l'image.

## Protocole commun

| Paramètre | Valeur |
| --- | --- |
| Architecture | U-Net `dim=32`, `dim_mults=(1, 2, 4)`, 1.3 M paramètres |
| Itérations | 10000 (batch 16, patchs 224×224) |
| Optimiseur | Adam + OneCycle, `max_lr=0.001` |
| Loss | `pixel=1.0` + `perceptual=0.1` + `gradient=0.1` |
| Éval | toutes les 500 it, 100 batches max, 5 jeux de test |
| Jitter d'exposition | (-1.9, 0.9) EV |
| Durée | 17–22 min par run (12.6 min sans terme perceptuel) |

### Mixture de référence (seuil à battre)

| | postcard | solid | wild | nature20 | real20 |
| --- | --- | --- | --- | --- | --- |
| PSNR mixture (dB) | 20.81 | 24.02 | 24.79 | 20.62 | 18.78 |
| SSIM mixture | 0.863 | 0.890 | 0.893 | 0.834 | 0.745 |

## Les 11 expériences

| Expérience | Source principale | Cible | Poids (sim/VOC · nature · real89) | Ce qui change | Seeds |
| --- | --- | --- | --- | --- | --- |
| **SM** | simulated (5 k) | `t` | 0.8 / 0.1 / 0.1 | référence : simulé + nature + real89 | 0, 0, 1, 2 |
| **SM_big** | simulated_big_biaised2 (15 k) | `t` | 0.8 / 0.1 / 0.1 | 3× plus d'exemples simulés, même distribution que `simulated` | 0, 1, 2 |
| **SMonly** | simulated (5 k) | `t` | 1.0 / 0 / 0 | simulé seul, sans photos réelles | 0, 1, 2 |
| **SMnotone** | simulated (5 k) | `t` | 0.8 / 0.1 / 0.1 | rendu gamma seul (`tone=False`), pas de courbe ACR | 0, 1, 2 |
| **SMnoperc** | simulated (5 k) | `t` | 0.8 / 0.1 / 0.1 | sans terme perceptuel (`perceptual=0`) | 0, 1, 2 |
| **BIAS1** | simulated_biaised1 (5 k) | `t` | 0.8 / 0.1 / 0.1 | reflets plus forts (vis jusqu'à ~0.59) | 0, 1, 2 |
| **BIAS2** | simulated_biaised2 (5 k) | `t` | 0.8 / 0.1 / 0.1 | reflets plus forts, second tirage | 0, 1, 2 |
| **BIAS2true** | simulated_biaised2 (5 k) | `true_t` | 0.8 / 0.1 / 0.1 | même données, cible = transmission **non atténuée** par la vitre | 0, 1, 2 |
| **NM** | simulated_not_modified (5 k) | `true_t` | 0.8 / 0.1 / 0.1 | cible `true_t` sans clé `t` disponible | 0, 1, 2 |
| **VOC** | VOC2012 (5 000 paires) | `—` | 0.8 / 0.1 / 0.1 | blend synthétique VOC + nature + real89 | 0, 1, 2, 3 |
| **VOConly** | VOC2012 (5 000 paires) | `—` | 1.0 / 0 / 0 | VOC seul, sans photos réelles | 0, 1, 2 |

## Résultats par expérience

PSNR en dB / SSIM, à l'itération 10 000. `Δ` = écart au PSNR de la mixture.

### SM

*référence : simulé + nature + real89. Données : simulated (5 k), cible `t`, poids 0.8 / 0.1 / 0.1. Durée moyenne 17.7 min.*

| Run | seed | postcard | solid | wild | nature20 | real20 | moy. 5 jeux |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `SMseed0_0` | 0 | 20.78 / 0.849 | 25.44 / 0.915 | 25.97 / 0.914 | 22.22 / 0.843 | 20.67 / 0.789 | 23.02 |
| `SMseed0_1` | 0 | 20.85 / 0.853 | 25.86 / 0.917 | 26.01 / 0.914 | 22.11 / 0.842 | 20.92 / 0.792 | 23.15 |
| `SMseed_1` | 1 | 19.97 / 0.838 | 25.51 / 0.912 | 25.83 / 0.913 | 22.28 / 0.844 | 20.69 / 0.791 | 22.86 |
| `SMseed_2` | 2 | 21.18 / 0.853 | 25.37 / 0.915 | 25.82 / 0.913 | 22.40 / 0.846 | 20.83 / 0.790 | 23.12 |
| **moyenne** | — | **20.69** ± 0.52 / **0.848** | **25.55** ± 0.22 / **0.915** | **25.91** ± 0.10 / **0.913** | **22.25** ± 0.12 / **0.844** | **20.78** ± 0.12 / **0.791** | **23.04** |
| Δ vs mixture | — | -0.12 | +1.53 | +1.12 | +1.63 | +1.99 | — |

### SM_big

*3× plus d'exemples simulés, même distribution que `simulated`. Données : simulated_big_biaised2 (15 k), cible `t`, poids 0.8 / 0.1 / 0.1. Durée moyenne 17.4 min.*

| Run | seed | postcard | solid | wild | nature20 | real20 | moy. 5 jeux |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `SM_big_seed_0` | 0 | 21.72 / 0.863 | 25.59 / 0.917 | 25.79 / 0.912 | 22.32 / 0.845 | 20.90 / 0.790 | 23.26 |
| `SM_big_seed_1` | 1 | 20.90 / 0.856 | 25.46 / 0.912 | 25.73 / 0.911 | 22.25 / 0.844 | 21.03 / 0.792 | 23.07 |
| `SM_big_seed_2` | 2 | 21.39 / 0.851 | 25.74 / 0.917 | 25.89 / 0.913 | 22.45 / 0.846 | 21.04 / 0.794 | 23.30 |
| **moyenne** | — | **21.33** ± 0.41 / **0.856** | **25.60** ± 0.14 / **0.915** | **25.80** ± 0.08 / **0.912** | **22.34** ± 0.10 / **0.845** | **20.99** ± 0.08 / **0.792** | **23.21** |
| Δ vs mixture | — | +0.53 | +1.58 | +1.02 | +1.72 | +2.21 | — |

### SMonly

*simulé seul, sans photos réelles. Données : simulated (5 k), cible `t`, poids 1.0 / 0 / 0. Durée moyenne 18.3 min.*

| Run | seed | postcard | solid | wild | nature20 | real20 | moy. 5 jeux |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `SMonly_seed0` | 0 | 20.87 / 0.854 | 24.94 / 0.911 | 25.63 / 0.912 | 21.66 / 0.834 | 20.64 / 0.791 | 22.75 |
| `SMonly_seed1` | 1 | 21.47 / 0.861 | 25.07 / 0.913 | 25.21 / 0.910 | 22.01 / 0.838 | 20.35 / 0.791 | 22.82 |
| `SMonly_seed2` | 2 | 21.29 / 0.858 | 25.05 / 0.909 | 25.34 / 0.909 | 21.74 / 0.837 | 20.69 / 0.792 | 22.82 |
| **moyenne** | — | **21.21** ± 0.31 / **0.857** | **25.02** ± 0.07 / **0.911** | **25.39** ± 0.21 / **0.910** | **21.81** ± 0.18 / **0.836** | **20.56** ± 0.18 / **0.792** | **22.80** |
| Δ vs mixture | — | +0.40 | +1.00 | +0.61 | +1.19 | +1.78 | — |

### SMnotone

*rendu gamma seul (`tone=False`), pas de courbe ACR. Données : simulated (5 k), cible `t`, poids 0.8 / 0.1 / 0.1. Durée moyenne 17.7 min.*

| Run | seed | postcard | solid | wild | nature20 | real20 | moy. 5 jeux |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `SMnotone_seed0` | 0 | 21.49 / 0.851 | 25.91 / 0.915 | 26.01 / 0.911 | 22.13 / 0.842 | 20.53 / 0.784 | 23.22 |
| `SMnotone_seed1` | 1 | 21.76 / 0.853 | 25.70 / 0.913 | 25.98 / 0.912 | 22.31 / 0.844 | 20.70 / 0.785 | 23.29 |
| `SMnotone_seed2` | 2 | 20.47 / 0.839 | 25.66 / 0.913 | 25.99 / 0.912 | 22.47 / 0.844 | 20.63 / 0.788 | 23.04 |
| **moyenne** | — | **21.24** ± 0.68 / **0.848** | **25.76** ± 0.14 / **0.914** | **25.99** ± 0.01 / **0.912** | **22.31** ± 0.17 / **0.843** | **20.62** ± 0.09 / **0.786** | **23.18** |
| Δ vs mixture | — | +0.43 | +1.74 | +1.21 | +1.69 | +1.84 | — |

### SMnoperc

*sans terme perceptuel (`perceptual=0`). Données : simulated (5 k), cible `t`, poids 0.8 / 0.1 / 0.1. Durée moyenne 12.6 min.*

| Run | seed | postcard | solid | wild | nature20 | real20 | moy. 5 jeux |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `SMnoperc_seed0` | 0 | 19.53 / 0.819 | 25.00 / 0.906 | 25.24 / 0.900 | 21.99 / 0.836 | 20.07 / 0.781 | 22.37 |
| `SMnoperc_seed1` | 1 | 19.06 / 0.816 | 23.67 / 0.898 | 24.87 / 0.897 | 21.87 / 0.836 | 20.05 / 0.784 | 21.90 |
| `SMnoperc_seed2` | 2 | 19.49 / 0.817 | 25.26 / 0.908 | 25.38 / 0.901 | 22.12 / 0.837 | 20.37 / 0.783 | 22.53 |
| **moyenne** | — | **19.36** ± 0.26 / **0.817** | **24.64** ± 0.85 / **0.904** | **25.16** ± 0.26 / **0.899** | **22.00** ± 0.13 / **0.836** | **20.16** ± 0.18 / **0.782** | **22.26** |
| Δ vs mixture | — | -1.45 | +0.62 | +0.38 | +1.38 | +1.38 | — |

### BIAS1

*reflets plus forts (vis jusqu'à ~0.59). Données : simulated_biaised1 (5 k), cible `t`, poids 0.8 / 0.1 / 0.1. Durée moyenne 19.5 min.*

| Run | seed | postcard | solid | wild | nature20 | real20 | moy. 5 jeux |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `BIAS1_seed0` | 0 | 19.57 / 0.835 | 25.36 / 0.915 | 25.54 / 0.909 | 22.38 / 0.844 | 20.30 / 0.778 | 22.63 |
| `BIAS1_seed1` | 1 | 19.10 / 0.825 | 25.25 / 0.914 | 25.62 / 0.911 | 22.17 / 0.841 | 20.38 / 0.777 | 22.50 |
| `BIAS1_seed2` | 2 | 19.04 / 0.825 | 25.71 / 0.917 | 25.82 / 0.912 | 22.22 / 0.843 | 20.59 / 0.784 | 22.68 |
| **moyenne** | — | **19.24** ± 0.29 / **0.828** | **25.44** ± 0.24 / **0.915** | **25.66** ± 0.14 / **0.911** | **22.26** ± 0.11 / **0.843** | **20.42** ± 0.15 / **0.780** | **22.60** |
| Δ vs mixture | — | -1.57 | +1.42 | +0.87 | +1.64 | +1.64 | — |

### BIAS2

*reflets plus forts, second tirage. Données : simulated_biaised2 (5 k), cible `t`, poids 0.8 / 0.1 / 0.1. Durée moyenne 20.0 min.*

| Run | seed | postcard | solid | wild | nature20 | real20 | moy. 5 jeux |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `BIAS2_seed0` | 0 | 18.93 / 0.828 | 25.44 / 0.917 | 25.70 / 0.909 | 22.07 / 0.841 | 20.60 / 0.782 | 22.55 |
| `BIAS2_seed1` | 1 | 18.42 / 0.817 | 25.48 / 0.914 | 25.72 / 0.912 | 22.27 / 0.844 | 20.57 / 0.781 | 22.49 |
| `BIAS2_seed2` | 2 | 18.69 / 0.826 | 25.38 / 0.916 | 25.62 / 0.910 | 22.27 / 0.844 | 20.65 / 0.783 | 22.52 |
| **moyenne** | — | **18.68** ± 0.26 / **0.823** | **25.43** ± 0.05 / **0.916** | **25.68** ± 0.06 / **0.910** | **22.20** ± 0.11 / **0.843** | **20.61** ± 0.04 / **0.782** | **22.52** |
| Δ vs mixture | — | -2.13 | +1.41 | +0.89 | +1.58 | +1.82 | — |

### BIAS2true

*même données, cible = transmission **non atténuée** par la vitre. Données : simulated_biaised2 (5 k), cible `true_t`, poids 0.8 / 0.1 / 0.1. Durée moyenne 18.5 min.*

| Run | seed | postcard | solid | wild | nature20 | real20 | moy. 5 jeux |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `BIAS2true_seed0` | 0 | 20.13 / 0.832 | 25.65 / 0.912 | 25.45 / 0.910 | 22.18 / 0.844 | 20.72 / 0.782 | 22.82 |
| `BIAS2true_seed1` | 1 | 19.65 / 0.823 | 25.68 / 0.914 | 25.44 / 0.910 | 22.16 / 0.843 | 20.86 / 0.783 | 22.76 |
| `BIAS2true_seed2` | 2 | 19.49 / 0.824 | 25.64 / 0.916 | 25.60 / 0.909 | 22.21 / 0.843 | 20.72 / 0.783 | 22.73 |
| **moyenne** | — | **19.76** ± 0.33 / **0.826** | **25.66** ± 0.02 / **0.914** | **25.50** ± 0.09 / **0.910** | **22.18** ± 0.03 / **0.843** | **20.76** ± 0.08 / **0.783** | **22.77** |
| Δ vs mixture | — | -1.05 | +1.64 | +0.71 | +1.57 | +1.98 | — |

### NM

*cible `true_t` sans clé `t` disponible. Données : simulated_not_modified (5 k), cible `true_t`, poids 0.8 / 0.1 / 0.1. Durée moyenne 17.6 min.*

| Run | seed | postcard | solid | wild | nature20 | real20 | moy. 5 jeux |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `NM_seed0` | 0 | 19.46 / 0.835 | 24.32 / 0.911 | 25.12 / 0.908 | 22.09 / 0.839 | 20.38 / 0.778 | 22.27 |
| `NM_seed1` | 1 | 19.35 / 0.832 | 24.39 / 0.907 | 24.74 / 0.905 | 21.85 / 0.839 | 20.86 / 0.784 | 22.24 |
| `NM_seed2` | 2 | 19.77 / 0.837 | 24.16 / 0.908 | 24.79 / 0.905 | 21.78 / 0.838 | 20.47 / 0.780 | 22.19 |
| **moyenne** | — | **19.53** ± 0.22 / **0.835** | **24.29** ± 0.12 / **0.909** | **24.89** ± 0.21 / **0.906** | **21.91** ± 0.16 / **0.839** | **20.57** ± 0.25 / **0.781** | **22.24** |
| Δ vs mixture | — | -1.28 | +0.27 | +0.10 | +1.29 | +1.79 | — |

### VOC

*blend synthétique VOC + nature + real89. Données : VOC2012 (5 000 paires), cible `—`, poids 0.8 / 0.1 / 0.1. Durée moyenne 17.3 min.*

| Run | seed | postcard | solid | wild | nature20 | real20 | moy. 5 jeux |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `VOCseed_0` | 0 | 20.38 / 0.856 | 25.46 / 0.911 | 25.56 / 0.905 | 21.87 / 0.839 | 20.61 / 0.785 | 22.78 |
| `VOCseed_1` | 1 | 19.68 / 0.851 | 25.57 / 0.913 | 25.25 / 0.903 | 22.04 / 0.840 | 20.53 / 0.786 | 22.61 |
| `VOCseed_2` | 2 | 20.74 / 0.860 | 25.52 / 0.912 | 25.50 / 0.907 | 22.11 / 0.840 | 20.22 / 0.782 | 22.82 |
| `VOCseed_3` | 3 | 20.17 / 0.854 | 25.33 / 0.911 | 25.24 / 0.905 | 22.07 / 0.840 | 20.44 / 0.781 | 22.65 |
| **moyenne** | — | **20.24** ± 0.44 / **0.855** | **25.47** ± 0.10 / **0.912** | **25.39** ± 0.17 / **0.905** | **22.02** ± 0.11 / **0.840** | **20.45** ± 0.17 / **0.783** | **22.71** |
| Δ vs mixture | — | -0.57 | +1.45 | +0.60 | +1.40 | +1.67 | — |

### VOConly

*VOC seul, sans photos réelles. Données : VOC2012 (5 000 paires), cible `—`, poids 1.0 / 0 / 0. Durée moyenne 18.1 min.*

| Run | seed | postcard | solid | wild | nature20 | real20 | moy. 5 jeux |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `VOConly_seed0` | 0 | 20.38 / 0.856 | 24.36 / 0.902 | 24.15 / 0.891 | 19.75 / 0.782 | 19.73 / 0.769 | 21.67 |
| `VOConly_seed1` | 1 | 20.30 / 0.856 | 24.18 / 0.901 | 23.95 / 0.893 | 19.45 / 0.786 | 19.89 / 0.773 | 21.55 |
| `VOConly_seed2` | 2 | 20.74 / 0.860 | 24.27 / 0.901 | 24.40 / 0.898 | 19.47 / 0.782 | 19.85 / 0.774 | 21.74 |
| **moyenne** | — | **20.47** ± 0.23 / **0.858** | **24.27** ± 0.09 / **0.902** | **24.17** ± 0.22 / **0.894** | **19.56** ± 0.17 / **0.783** | **19.82** ± 0.08 / **0.772** | **21.66** |
| Δ vs mixture | — | -0.34 | +0.25 | -0.62 | -1.06 | +1.04 | — |

## Synthèse — classement

Trié par PSNR moyen sur les 5 jeux de test.

| Rang | Expérience | postcard | solid | wild | nature20 | real20 | moy. 5 | moy. SIR² | moy. nature20+real20 | SSIM moy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **SM_big** | 21.33 | 25.60 | 25.80 | 22.34 | 20.99 | **23.21** | 24.25 | 21.66 | 0.864 |
| 2 | **SMnotone** | 21.24 | 25.76 | 25.99 | 22.31 | 20.62 | **23.18** | 24.33 | 21.46 | 0.860 |
| 3 | **SM** | 20.69 | 25.55 | 25.91 | 22.25 | 20.78 | **23.04** | 24.05 | 21.51 | 0.862 |
| 4 | **SMonly** | 21.21 | 25.02 | 25.39 | 21.81 | 20.56 | **22.80** | 23.87 | 21.18 | 0.861 |
| 5 | **BIAS2true** | 19.76 | 25.66 | 25.50 | 22.18 | 20.76 | **22.77** | 23.64 | 21.47 | 0.855 |
| 6 | **VOC** | 20.24 | 25.47 | 25.39 | 22.02 | 20.45 | **22.71** | 23.70 | 21.23 | 0.859 |
| 7 | **BIAS1** | 19.24 | 25.44 | 25.66 | 22.26 | 20.42 | **22.60** | 23.45 | 21.34 | 0.855 |
| 8 | **BIAS2** | 18.68 | 25.43 | 25.68 | 22.20 | 20.61 | **22.52** | 23.26 | 21.40 | 0.855 |
| 9 | **SMnoperc** | 19.36 | 24.64 | 25.16 | 22.00 | 20.16 | **22.26** | 23.06 | 21.08 | 0.848 |
| 10 | **NM** | 19.53 | 24.29 | 24.89 | 21.91 | 20.57 | **22.24** | 22.90 | 21.24 | 0.854 |
| 11 | **VOConly** | 20.47 | 24.27 | 24.17 | 19.56 | 19.82 | **21.66** | 22.97 | 19.69 | 0.842 |
| — | *mixture* | 20.81 | 24.02 | 24.79 | 20.62 | 18.78 | *21.80* | *23.20* | *19.70* | — |

> [!warning] `VOConly` est en dessous de l'image d'entrée
> Moyenne 21.66 dB contre 21.80 dB pour la mixture non traitée. Il ne dégrade pas partout — il gagne
> 1.04 dB sur real20 — mais il perd 1.06 dB sur nature20 et 0.62 dB sur wild. **Entraîné sur le seul blend
> VOC2012, le réseau abîme plus d'images qu'il n'en répare.** C'est le seul groupe dans ce cas.

Les dix autres groupes battent la mixture en moyenne, de +0.44 dB (NM) à +1.41 dB (SM_big).
Le gain se concentre sur `nature20` et `real20`, où tous sauf VOConly gagnent 1.2 à 2.2 dB ;
`postcard` est le seul jeu où la majorité des groupes **perdent** face à l'entrée.
## Comparaisons

Écart de PSNR entre deux expériences, ± l'erreur-type de la différence. `*` marque les écarts supérieurs à 2 erreurs-types — avec 3 ou 4 seeds c'est **indicatif, pas un test**.

| Comparaison | postcard | solid | wild | nature20 | real20 | moy. 5 | Lecture |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **SM** − VOC | +0.45 ± 0.34 | +0.07 ± 0.12 | +0.52 ± 0.10\* | +0.23 ± 0.08\* | +0.33 ± 0.10\* | **+0.32** | Simulation vs blend VOC, à recette identique |
| **SMonly** − VOConly | +0.74 ± 0.22\* | +0.75 ± 0.07\* | +1.23 ± 0.18\* | +2.25 ± 0.14\* | +0.74 ± 0.11\* | **+1.14** | Idem, sans les photos réelles pour isoler la source |
| **SM** − SMonly | -0.51 ± 0.31 | +0.53 ± 0.12\* | +0.51 ± 0.13\* | +0.45 ± 0.12\* | +0.22 ± 0.12 | **+0.24** | Apport des 20 % de photos réelles (nature + real89) au simulé |
| **VOC** − VOConly | -0.23 ± 0.26 | +1.20 ± 0.07\* | +1.22 ± 0.15\* | +2.47 ± 0.11\* | +0.62 ± 0.10\* | **+1.06** | Le même apport, côté VOC |
| **SM_big** − SM | +0.64 ± 0.35 | +0.05 ± 0.14 | -0.10 ± 0.07 | +0.09 ± 0.08 | +0.21 ± 0.08\* | **+0.18** | 5 000 → 15 000 exemples simulés |
| **BIAS1** − SM | -1.45 ± 0.31\* | -0.10 ± 0.18 | -0.25 ± 0.10\* | +0.00 ± 0.09 | -0.35 ± 0.10\* | **-0.43** | Reflets plus forts à l'entraînement (biaised1) |
| **BIAS2** − SM | -2.02 ± 0.30\* | -0.11 ± 0.11 | -0.23 ± 0.06\* | -0.05 ± 0.09 | -0.17 ± 0.06\* | **-0.52** | Reflets plus forts (biaised2) |
| **BIAS1** − BIAS2 | +0.56 ± 0.22\* | +0.01 ± 0.14 | -0.02 ± 0.09 | +0.05 ± 0.09 | -0.18 ± 0.09\* | **+0.08** | Les deux tirages « biaisés » entre eux |
| **BIAS2true** − BIAS2 | +1.08 ± 0.24\* | +0.22 ± 0.03\* | -0.18 ± 0.06\* | -0.02 ± 0.07 | +0.16 ± 0.05\* | **+0.25** | Cible `true_t` (non atténuée) vs `t`, mêmes données |
| **NM** − SM | -1.17 ± 0.29\* | -1.25 ± 0.13\* | -1.02 ± 0.13\* | -0.34 ± 0.11\* | -0.21 ± 0.16 | **-0.80** | Cible `true_t` de `simulated_not_modified` |
| **SMnotone** − SM | +0.55 ± 0.47 | +0.21 ± 0.14 | +0.09 ± 0.05 | +0.05 ± 0.12 | -0.15 ± 0.08 | **+0.15** | Rendu gamma seul vs courbe ACR |
| **SMnoperc** − SM | -1.34 ± 0.30\* | -0.90 ± 0.50 | -0.74 ± 0.16\* | -0.26 ± 0.10\* | -0.61 ± 0.12\* | **-0.77** | Retrait du terme perceptuel |

### Ce qui ressort

1. **La simulation bat le blend VOC**, surtout sur les jeux réels : SM − VOC = +0.33 dB sur real20 et +0.52 dB sur wild. Sans photos réelles à l'entraînement l'écart explose : SMonly − VOConly = +1.14 dB en moyenne, dont +2.25 dB sur nature20.
2. **Les 20 % de photos réelles portent presque tout le transfert de VOC**, et beaucoup moins celui du simulé : VOC − VOConly = +1.06 dB contre SM − SMonly = +0.24 dB. Autrement dit VOConly seul ne généralise pas (il est **sous la mixture** sur wild et nature20), alors que SMonly tient debout tout seul.
3. **Tripler le volume simulé ne rapporte presque rien** : SM_big − SM = +0.18 dB en moyenne, le seul écart net étant +0.21 dB sur real20. À 10 000 itérations et 1.3 M paramètres, ce n'est pas la taille du jeu qui limite.
4. **Entraîner sur des reflets plus forts dégrade le test**, et surtout postcard : BIAS2 − SM = -2.02 dB. Les jeux de test ont des reflets nettement plus faibles que les jeux biaisés (voir la section datasets) : on entraîne à côté de la cible.
5. **`true_t` n'est pas une cible équivalente à `t`.** Sur les mêmes données, BIAS2true − BIAS2 = +1.08 dB sur postcard mais -0.18 dB sur wild : la cible non atténuée déplace le compromis plus qu'elle ne l'améliore. Et NM, dont le `true_t` est incohérent en gain (voir anomalies), est **le pire groupe** : -0.80 dB.
6. **Le terme perceptuel gagne son coût.** SMnoperc − SM = -0.77 dB en PSNR et -0.0143 en SSIM, pour 12.6 min au lieu de 17.7 min. Le PSNR **et** le SSIM baissent : ce n'est pas un simple arbitrage netteté/fidélité.
7. **La courbe ACR n'apporte rien de mesurable** : SMnotone − SM = +0.15 dB, aucun écart au-delà du bruit inter-seed. Le rendu gamma seul suffit sur ces benchmarks.

### Bruit inter-seed — l'échelle à laquelle lire les écarts

| Jeu de test | écart-type inter-seed (SM, n=4) | écart max−min (SM) | écart entre les deux runs seed 0 |
| --- | --- | --- | --- |
| postcard | 0.52 dB | 1.22 dB | 0.07 dB |
| solid | 0.22 dB | 0.50 dB | 0.43 dB |
| wild | 0.10 dB | 0.18 dB | 0.04 dB |
| nature20 | 0.12 dB | 0.30 dB | 0.11 dB |
| real20 | 0.12 dB | 0.25 dB | 0.25 dB |

> [!warning] Deux runs à seed identique ne donnent pas le même résultat
> `SMseed0_0` et `SMseed0_1` ont **la même config et le même seed**. Ils divergent quand même de 0.43 dB sur solid et 0.25 dB sur real20 — les pertes divergent dès la 2ᵉ itération. Causes attendues : cuDNN benchmark, AMP, `torch.compile`, et le `random` global du dataset (les workers ne sont pas seedés).
> **Conséquence pratique : tout écart inférieur à ~0.3 dB sur solid/real20 et ~1 dB sur postcard n'est pas interprétable ici.**
## Caractéristiques des datasets

**Comment lire la force du reflet.** Il n'existe pas d'étiquette « reflet fort » dans ces jeux : je la dérive du **PSNR entre la mixture M et la transmission de référence T**. Plus la mixture est loin de la transmission, plus le reflet pèse. C'est exactement la quantité que le modèle doit améliorer, donc les seuils ci-dessous sont directement comparables aux colonnes de résultats.

Tous les chiffres sont calculés dans le **domaine d'affichage** que voit le modèle : crop carré centré redimensionné en 224×224 pour les photos, rendu ACR (`tone=True`, `ev=0`) pour les `.npz` simulés. Le SSIM est celui de `utils/ssim_ref.py` (boîte 7×7, pondération par canal), réimplémenté en numpy et vérifié identique au tenseur de référence à 6·10⁻⁸ près.

### Jeux d'entraînement

| Jeu | n paires | PSNR(M,T) moy | méd | p10 | p90 | SSIM(M,T) moy | reflet fort<br>< 18 dB | modéré<br>18–22 | faible<br>22–28 | très faible<br>> 28 dB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `simulated` (SM, SMonly, SMnotone, SMnoperc) | 5000 | 17.33 | 17.18 | 13.67 | 21.34 | 0.804 | 59.7 % | 34.1 % | 6.2 % | 0.0 % |
| `simulated_big_biaised2` (SM_big) | 15000 | 17.32 | 17.15 | 13.63 | 21.36 | 0.804 | 59.5 % | 34.2 % | 6.3 % | 0.0 % |
| `simulated_biaised1` — cible `t` (BIAS1) | 5000 | 16.34 | 15.95 | 12.31 | 21.05 | 0.771 | 68.3 % | 26.2 % | 5.5 % | 0.0 % |
| `simulated_biaised2` — cible `t` (BIAS2) | 5000 | 16.30 | 15.90 | 12.26 | 21.02 | 0.771 | 69.1 % | 25.4 % | 5.5 % | 0.0 % |
| `simulated_biaised2` — cible `true_t` (BIAS2true) | 5000 | 17.20 | 16.76 | 12.73 | 22.40 | 0.779 | 60.6 % | 27.4 % | 11.9 % | 0.0 % |
| `simulated_not_modified` — cible `true_t` (NM) | 5000 | 15.60 | 15.61 | 10.43 | 20.72 | 0.741 | 71.9 % | 22.8 % | 5.4 % | 0.0 % |
| `VOC2012` (VOC, VOConly — 5 000 premières paires) | 5000 | 14.20 | 13.71 | 10.01 | 18.86 | 0.747 | 86.5 % | 9.9 % | 3.1 % | 0.5 % |
| `nature_dataset` (10 % dans SM / VOC / BIAS* / NM) | 200 | 24.09 | 24.17 | 17.92 | 29.88 | 0.837 | 10.5 % | 17.0 % | 49.5 % | 23.0 % |
| `real89` (10 % dans SM / VOC / BIAS* / NM) | 89 | 19.30 | 17.82 | 12.94 | 26.84 | 0.764 | 51.7 % | 19.1 % | 21.3 % | 7.9 % |

### Jeux de test

| Jeu | n paires | PSNR(M,T) moy | méd | p10 | p90 | SSIM(M,T) moy | reflet fort<br>< 18 dB | modéré<br>18–22 | faible<br>22–28 | très faible<br>> 28 dB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `postcard` | 179 | 20.82 | 20.85 | 18.30 | 22.88 | 0.863 | 5.0 % | 72.6 % | 22.3 % | 0.0 % |
| `solid` | 200 | 24.02 | 23.17 | 20.68 | 28.76 | 0.890 | 0.0 % | 29.5 % | 55.5 % | 15.0 % |
| `wild` | 101 | 24.95 | 24.72 | 17.36 | 33.21 | 0.893 | 10.9 % | 23.8 % | 34.7 % | 30.7 % |
| `nature20` | 20 | 21.48 | 21.69 | 15.02 | 27.06 | 0.834 | 30.0 % | 25.0 % | 35.0 % | 10.0 % |
| `real20` | 20 | 18.82 | 17.92 | 14.72 | 23.77 | 0.745 | 60.0 % | 15.0 % | 20.0 % | 5.0 % |

### Énergie du reflet dans les jeux simulés

`vis` = moyenne(r) / moyenne(m) en sRGB **linéaire** — la fraction d'énergie apportée par le reflet. C'est le critère de tri utilisé à la génération (`measure()` dans `data_clean.ipynb`).

| Jeu | n | vis moy | méd | p10 | p90 | max | part > 0.40 | pixels écrêtés | images > 1 % écrêté |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `simulated` | 5000 | 0.258 | 0.256 | 0.139 | 0.379 | 0.531 | 5.4 % | 1.38 % | 22.4 % |
| `simulated_big_biaised2` | 15000 | 0.258 | 0.256 | 0.141 | 0.379 | 0.531 | 5.2 % | 1.44 % | 22.7 % |
| `simulated_biaised1` | 5000 | 0.304 | 0.300 | 0.145 | 0.466 | 0.585 | 26.2 % | 1.37 % | 23.1 % |
| `simulated_biaised2` | 5000 | 0.306 | 0.303 | 0.147 | 0.469 | 0.591 | 26.3 % | 1.37 % | 22.9 % |
| `simulated_not_modified` | 5000 | 0.258 | 0.256 | 0.139 | 0.379 | 0.531 | 5.4 % | 1.38 % | 22.5 % |

> [!important] L'entraînement et le test ne voient pas les mêmes reflets
> Les jeux simulés sont **beaucoup plus agressifs** que les benchmarks : `simulated` a 59.7 % de paires sous 18 dB, contre 5.0 % pour postcard, 0.0 % pour solid et 10.9 % pour wild. Les jeux « biaisés » aggravent l'écart (68–69 % sous 18 dB). Cela explique directement le classement : BIAS1/BIAS2 perdent 1.5–2 dB sur postcard, le jeu de test dont la fenêtre de reflets est la plus étroite (p10–p90 = 18.3–22.9 dB).

Le seul jeu d'entraînement plus dur que les simulés est **VOC2012** : PSNR(M,T) médian 13.71 dB, 86.5 % des paires sous 18 dB. Le blend VOC additionne deux photos à poids ~égaux, sans modèle de vitre — ce n'est pas la même statistique de reflet, et c'est cohérent avec VOConly qui ne transfère pas.
## Anomalies

Classées par impact sur l'interprétation des résultats.

### 🔴 1. `simulated_big_biaised2` n'est pas biaisé

Malgré son nom, sa distribution de force de reflet est **celle de `simulated`**, pas celle de `simulated_biaised2` :

| | vis moy | vis max | part > 0.40 |
| --- | --- | --- | --- |
| `simulated` | 0.258 | 0.531 | 5.4 % |
| `simulated_big_biaised2` | 0.258 | 0.531 | 5.2 % |
| `simulated_biaised2` | 0.306 | 0.591 | 26.3 % |

Les deux `trials.parquet` (`simulated` et `simulated_big_biaised2`) le confirment : même critère d'acceptation, `vis` plafonné à 0.4000 exactement, mêmes taux de rejet à 0.1 % près (71.4 / 71.6 % rejetés sur le SSIM, 10.1 % acceptés dans les deux cas).

**Conséquence :** `SM_big` n'est pas « BIAS2 en plus gros », c'est « SM en plus gros ». La comparaison SM_big − SM est bien un test de volume à distribution constante — mais si tu voulais tester « reflets forts + volume », ce run ne le fait pas. À renommer avant de publier quoi que ce soit.

### 🔴 2. `true_t` a un gain incohérent dans `biaised1` et `not_modified`

Rapport `moyenne(true_t) / moyenne(t)` sur 400 exemples :

| Jeu | p10 | médiane | p90 | corr(t, true_t) médiane |
| --- | --- | --- | --- | --- |
| `simulated_biaised2` | 1.088 | 1.102 | 1.158 | 0.999 |
| `simulated_biaised1` | 0.438 | 1.015 | 2.023 | 0.998 |
| `simulated_not_modified` | 0.414 | 0.979 | 1.793 | 0.998 |

Dans `biaised2`, `true_t` est proprement la transmission **dé-atténuée** : un facteur serré ≈ 1/(1−R) ≈ 1.10, exactement ce que `t = t * (1 - R_map)` retire dans `simulate_example()`. Dans `biaised1` et `not_modified`, le même rapport varie d'un facteur 4 d'un exemple à l'autre, alors que la **structure** reste parfaitement alignée (corrélation médiane 0.998). C'est donc un facteur d'échelle global aléatoire, pas un décalage de contenu.

**Cause la plus probable :** `true_t` y a été sauvé **avant** `compute_exposure()` (Func. S1). La ré-exposition normalise `m`, `t` et `r` ensemble pour amener `moyenne(m)` à `TAU = 0.1329` — et de fait `moyenne(m)` vaut 0.1329 partout — mais si `true_t` n'est pas passé dans le même appel, il garde l'échelle de la scène d'origine, qui varie énormément.

**Conséquence :** `NM` (seul groupe entraîné sur ce `true_t`-là) demande au réseau de deviner une exposition non déductible de l'entrée. C'est cohérent avec ce qu'on observe : NM a la plus grosse perte pixel de tout le sweep (0.091 contre 0.050 pour SM) et le pire PSNR moyen (−0.80 dB vs SM). **Le résultat de NM ne dit rien sur `true_t` comme cible** — il dit que ce fichier-là est cassé. Le vrai test de `true_t` est BIAS2true, et lui se tient (+0.25 dB vs BIAS2).

`BIAS1` n'est pas affecté : il utilise `t_key="t"`.

### 🟠 3. Fuite train → test : une image de `real20` est dans `real89`

`data_others/real89/real89/blended/77.jpg` est **identique octet à octet** à `robustsirr_test_dataset/real20/blended/58.jpg`, et pareil pour les deux `transmission_layer/`. (Diff max = 0 sur les deux couches, 736×1133.)

`real89` pèse 10 % de l'échantillonnage dans SM, SM_big, SMnotone, SMnoperc, BIAS1, BIAS2, BIAS2true, NM et VOC. Ces groupes ont donc vu 1 des 20 paires de real20 — **5 % du jeu de test**. `SMonly` et `VOConly` (poids 1/0/0) sont les seuls propres.

Aucun autre doublon : contrôle par hash sur toutes les paires train × test, aucun recouvrement de noms non plus.

### 🟠 4. Le PSNR d'éval est une moyenne de moyennes de batchs

`evaluate()` empile une moyenne par batch puis en fait la moyenne. Avec `batch_size=8`, un jeu dont la taille n'est pas multiple de 8 surpondère son dernier batch :

| Jeu | n | découpage | PSNR loggé | PSNR par image | biais |
| --- | --- | --- | --- | --- | --- |
| `nature20` | 20 | 8+8+**4** | 20.62 | 21.48 | **−0.86 dB** |
| `wild` | 101 | 12×8+**5** | 24.79 | 24.95 | −0.16 dB |
| `postcard` | 179 | 22×8+**3** | 20.81 | 20.82 | −0.01 dB |
| `real20` | 20 | 8+8+**4** | 18.78 | 18.82 | −0.04 dB |
| `solid` | 200 | 25×8 | 24.02 | 24.02 | 0 |

Le biais est **identique pour tous les runs**, donc les comparaisons entre expériences de cette note restent valides. En revanche les valeurs absolues, en particulier sur nature20, ne sont pas comparables aux chiffres publiés. Correctif : `drop_last=False` existe déjà, il faut moyenner sur les images et non sur les batchs (accumuler somme et compte), ou évaluer avec `batch_size=1`.

### 🟠 5. `postcard` n'est pas exploitable pour sélectionner un modèle

| Groupe | évals sous la mixture (sur 21) | meilleur point − point final |
| --- | --- | --- |
| SM | 9.5 | +1.81 dB |
| SM_big | 6.3 | +1.48 dB |
| SMonly | 6.3 | +1.43 dB |
| SMnotone | 5.3 | +1.78 dB |
| SMnoperc | 9.0 | +3.33 dB |
| BIAS1 | 17.3 | +2.29 dB |
| BIAS2 | 17.0 | +2.99 dB |
| BIAS2true | 14.3 | +2.19 dB |
| NM | 13.7 | +2.43 dB |
| VOC | 11.8 | +1.57 dB |
| VOConly | 15.7 | +0.86 dB |

Sur BIAS1 et BIAS2, le modèle est **sous la mixture dans 17 évals sur 21** : il abîme postcard plus souvent qu'il ne l'améliore. Et partout, le meilleur point de la trajectoire dépasse le point final de 1.4 à 3.3 dB — l'écart entre deux évals consécutives monte jusqu'à 4.62 dB (BIAS1_seed0, it 4000). postcard a la fenêtre de reflets la plus étroite des cinq jeux (p10–p90 = 18.3–22.9 dB) et 179 images seulement : à prendre comme un diagnostic, pas comme un critère.

Les quatre autres jeux sont stables : sur `nature20` et `real20`, aucun groupe sauf VOConly ne passe sous la mixture.

### 🟡 6. Aucun poids sauvegardé

Les 35 runs ont `"ckpt": false` — `find runs -name ckpt.pt` ne renvoie rien. Seuls `hist.json` et `log.txt` subsistent. Impossible de rejouer un modèle, de sortir des images, ou de refaire une éval en pleine résolution sans tout relancer (≈ 10 h de GPU).

### 🟡 7. Le groupe `SM` a 4 runs mais 3 seeds

`SMseed0_0` et `SMseed0_1` sont deux exécutions du **même seed 0**. L'écart-type « inter-seed » de SM (n=4) mélange donc variance de seed et non-déterminisme d'exécution. Ce n'est pas grave — c'est même utile pour calibrer le bruit — mais l'écart-type de SM n'est pas homogène à celui des autres groupes.

### 🟡 8. Les configs VOC portent des paramètres qui ne servent pas

`VOCseed_*` et `VOConly_*` ont `simulated=false` mais gardent `npz_dir`, `t_key="t"`, `ev_jitter=[-1.9, 0.9]` et `tone=true` dans leur `config.json`. Dans `build_train_loader()`, la branche `simulated=False` n'utilise aucun de ces arguments (`**kw` est ignoré).

Ce n'est pas un bug d'exécution, mais deux différences réelles s'ajoutent à « simulé vs VOC » et se confondent avec :

- **pas de jitter d'exposition** côté VOC (le simulé en reçoit ±1.9/+0.9 EV, les photos réelles jamais) ;
- **politique de crop différente** : VOC utilise `frac=(1.0, 1.0)` (champ complet), `nature` et `real89` `frac=(0.4, 1.0)`.

À garder en tête : SM − VOC n'isole pas la seule source de données.

### 🟢 9. Divers, sans conséquence

- **Paires (t, r) dupliquées** dans les jeux simulés : 4 984 paires distinctes pour 5 000 fichiers (14 823 pour 15 000). Les rendus diffèrent, seul le couple d'images source se répète.
- **`trials.parquet` compte plus d'acceptés que de fichiers** : 5 001 pour 5 000, 15 005 pour 15 000. Course sur le compteur dans la génération multi-thread ; les fichiers écrits sont corrects.
- **VOC contient 15 paires quasi dégénérées** sur 5 000 (PSNR(M,T) > 30 dB, jusqu'à 70 dB) : mixture ≈ transmission, reflet quasi nul. 0.3 %, négligeable.
- **`simulated` et `simulated_not_modified` sont le même tirage** : 1 487 mixtures identiques sur 1 500 comparées. `not_modified` remplace simplement la clé `t` par `true_t` (et n'a **pas** de clé `t`).
- **GPU 4 exclu de la file** (`queue.sh`) pour cause de *misaligned address / illegal memory access*. Tout le sweep a tourné sur le seul GPU 5, en séquentiel.

## Runs en cours (lancés 11:34, non inclus ci-dessus)

Trois runs `dim=64` tournent sur les GPU 3, 5 et 6 au moment de l'écriture — aucun n'est terminé, ils changent **trois choses à la fois** par rapport à `SM_big` :

| | SM_big | SM_big_*_64 |
| --- | --- | --- |
| `dim` | 32 | **64** |
| `iters` | 10 000 | **15 000** |
| poids (sim / nature / real89) | 0.8 / 0.1 / 0.1 | **0.96 / 0.02 / 0.02** |
| `eval_every` | 500 | 1 000 |

Le passage à 0.96/0.02/0.02 est le changement le plus lourd de conséquence : il ramène presque à `SMonly`, dont on sait qu'il perd 0.2 à 0.5 dB sur solid, wild et nature20. Si le run gagne ou perd, on ne saura pas lequel des trois facteurs en est responsable.

## Fichiers

- Runs : `~/ProjectsDL/reflection_removal/runs/<nom>/{config.json, hist.json, log.txt}`
- Journal de file : `~/ProjectsDL/reflection_removal/runs/queue.log`
- Configs du sweep : `~/ProjectsDL/reflection_removal/sweep/*.json`
- Code : `train.py`, `dataset_sim.py`, `loss_ref.py`, `acr_tone.py`, `utils/ssim_ref.py`
- Génération des données : `data_clean.ipynb`
- Jeux simulés : `~/data_fivek_dng/simulated*/` (+ `trials.parquet` pour `simulated` et `simulated_big_biaised2`)
- Jeux réels : `~/data_others/{VOC2012, nature_dataset, real89, robustsirr_test_dataset}`