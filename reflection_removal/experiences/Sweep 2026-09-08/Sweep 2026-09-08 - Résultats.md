---
tags:
  - sirr
  - sweep
  - resultats
date: 2026-09-08
---

# Résultats par expérience

Retour à [[Sweep 2026-09-08]].

PSNR en dB à l'itération 10 000, SSIM dans le tableau qui suit chaque expérience. `Δ` = écart au PSNR de la mixture.

## Classement

| Rang | Expérience | moy. 5 jeux | Δ mixture | moy. SIR2 | moy. nature20+real20 | SSIM moy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **SM_big** | **23.21** | +1.41 | 24.25 | 21.66 | 0.864 |
| 2 | **SMnotone** | **23.18** | +1.38 | 24.33 | 21.46 | 0.860 |
| 3 | **SM** | **23.04** | +1.23 | 24.05 | 21.51 | 0.862 |
| 4 | **SMonly** | **22.80** | +0.99 | 23.87 | 21.18 | 0.861 |
| 5 | **BIAS2true** | **22.77** | +0.97 | 23.64 | 21.47 | 0.855 |
| 6 | **VOC** | **22.71** | +0.91 | 23.70 | 21.23 | 0.859 |
| 7 | **BIAS1** | **22.60** | +0.80 | 23.45 | 21.34 | 0.855 |
| 8 | **BIAS2** | **22.52** | +0.72 | 23.26 | 21.40 | 0.855 |
| 9 | **SMnoperc** | **22.26** | +0.46 | 23.06 | 21.08 | 0.848 |
| 10 | **NM** | **22.24** | +0.43 | 22.90 | 21.24 | 0.854 |
| 11 | **VOConly** | **21.66** | -0.15 | 22.97 | 19.69 | 0.842 |
| — | *mixture* | *21.80* | — | *23.20* | *19.70* | — |

### PSNR par jeu

| Expérience | postcard | solid | wild | nature20 | real20 |
| --- | --- | --- | --- | --- | --- |
| **SM_big** | 21.33 | 25.60 | 25.80 | 22.34 | 20.99 |
| **SMnotone** | 21.24 | 25.76 | 25.99 | 22.31 | 20.62 |
| **SM** | 20.69 | 25.55 | 25.91 | 22.25 | 20.78 |
| **SMonly** | 21.21 | 25.02 | 25.39 | 21.81 | 20.56 |
| **BIAS2true** | 19.76 | 25.66 | 25.50 | 22.18 | 20.76 |
| **VOC** | 20.24 | 25.47 | 25.39 | 22.02 | 20.45 |
| **BIAS1** | 19.24 | 25.44 | 25.66 | 22.26 | 20.42 |
| **BIAS2** | 18.68 | 25.43 | 25.68 | 22.20 | 20.61 |
| **SMnoperc** | 19.36 | 24.64 | 25.16 | 22.00 | 20.16 |
| **NM** | 19.53 | 24.29 | 24.89 | 21.91 | 20.57 |
| **VOConly** | 20.47 | 24.27 | 24.17 | 19.56 | 19.82 |
| *mixture* | *20.81* | *24.02* | *24.79* | *20.62* | *18.78* |

### Δ PSNR vs mixture

| Expérience | postcard | solid | wild | nature20 | real20 |
| --- | --- | --- | --- | --- | --- |
| **SM_big** | +0.53 | +1.58 | +1.02 | +1.72 | +2.21 |
| **SMnotone** | +0.43 | +1.74 | +1.21 | +1.69 | +1.84 |
| **SM** | -0.12 | +1.53 | +1.12 | +1.63 | +1.99 |
| **SMonly** | +0.40 | +1.00 | +0.61 | +1.19 | +1.78 |
| **BIAS2true** | -1.05 | +1.64 | +0.71 | +1.57 | +1.98 |
| **VOC** | -0.57 | +1.45 | +0.60 | +1.40 | +1.67 |
| **BIAS1** | -1.57 | +1.42 | +0.87 | +1.64 | +1.64 |
| **BIAS2** | -2.13 | +1.41 | +0.89 | +1.58 | +1.82 |
| **SMnoperc** | -1.45 | +0.62 | +0.38 | +1.38 | +1.38 |
| **NM** | -1.28 | +0.27 | +0.10 | +1.29 | +1.79 |
| **VOConly** | -0.34 | +0.25 | -0.62 | -1.06 | +1.04 |

> [!warning] `VOConly` est en dessous de l'image d'entrée
> 21.66 dB contre 21.80 pour la mixture non traitée. Il gagne +1.04 dB sur real20 mais perd -1.06 dB sur nature20 et -0.62 dB sur wild. Entraîné sur le seul blend VOC2012, le réseau abîme plus d'images qu'il n'en répare. C'est le seul groupe dans ce cas.

Les dix autres battent la mixture, de +0.44 dB (`NM`) à +1.41 dB (`SM_big`). Le gain se concentre sur `nature20` et `real20`, où tous sauf `VOConly` gagnent 1.2 à 2.2 dB. `postcard` est le seul jeu où la majorité des groupes **perdent** face à l'entrée.

## Détail par expérience

### SM

référence : simulé + nature + real89. Données : simulated (5 k), cible `t`, poids 0.8 / 0.1 / 0.1. Durée moyenne 17.7 min.

**PSNR (dB)**

| Run | seed | postcard | solid | wild | nature20 | real20 | moy. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `SMseed0_0` | 0 | 20.78 | 25.44 | 25.97 | 22.22 | 20.67 | 23.02 |
| `SMseed0_1` | 0 | 20.85 | 25.86 | 26.01 | 22.11 | 20.92 | 23.15 |
| `SMseed_1` | 1 | 19.97 | 25.51 | 25.83 | 22.28 | 20.69 | 22.86 |
| `SMseed_2` | 2 | 21.18 | 25.37 | 25.82 | 22.40 | 20.83 | 23.12 |
| **moyenne** | — | **20.69** ± 0.52 | **25.55** ± 0.22 | **25.91** ± 0.10 | **22.25** ± 0.12 | **20.78** ± 0.12 | **23.04** |
| Δ mixture | — | -0.12 | +1.53 | +1.12 | +1.63 | +1.99 | — |

**SSIM**

| Run | postcard | solid | wild | nature20 | real20 |
| --- | --- | --- | --- | --- | --- |
| `SMseed0_0` | 0.849 | 0.915 | 0.914 | 0.843 | 0.789 |
| `SMseed0_1` | 0.853 | 0.917 | 0.914 | 0.842 | 0.792 |
| `SMseed_1` | 0.838 | 0.912 | 0.913 | 0.844 | 0.791 |
| `SMseed_2` | 0.853 | 0.915 | 0.913 | 0.846 | 0.790 |
| **moyenne** | **0.848** | **0.915** | **0.913** | **0.844** | **0.791** |

### SM_big

3x plus d'exemples simulés, même distribution que `simulated`. Données : simulated_big_biaised2 (15 k), cible `t`, poids 0.8 / 0.1 / 0.1. Durée moyenne 17.4 min.

**PSNR (dB)**

| Run | seed | postcard | solid | wild | nature20 | real20 | moy. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `SM_big_seed_0` | 0 | 21.72 | 25.59 | 25.79 | 22.32 | 20.90 | 23.26 |
| `SM_big_seed_1` | 1 | 20.90 | 25.46 | 25.73 | 22.25 | 21.03 | 23.07 |
| `SM_big_seed_2` | 2 | 21.39 | 25.74 | 25.89 | 22.45 | 21.04 | 23.30 |
| **moyenne** | — | **21.33** ± 0.41 | **25.60** ± 0.14 | **25.80** ± 0.08 | **22.34** ± 0.10 | **20.99** ± 0.08 | **23.21** |
| Δ mixture | — | +0.53 | +1.58 | +1.02 | +1.72 | +2.21 | — |

**SSIM**

| Run | postcard | solid | wild | nature20 | real20 |
| --- | --- | --- | --- | --- | --- |
| `SM_big_seed_0` | 0.863 | 0.917 | 0.912 | 0.845 | 0.790 |
| `SM_big_seed_1` | 0.856 | 0.912 | 0.911 | 0.844 | 0.792 |
| `SM_big_seed_2` | 0.851 | 0.917 | 0.913 | 0.846 | 0.794 |
| **moyenne** | **0.856** | **0.915** | **0.912** | **0.845** | **0.792** |

### SMonly

simulé seul, sans photos réelles. Données : simulated (5 k), cible `t`, poids 1.0 / 0 / 0. Durée moyenne 18.3 min.

**PSNR (dB)**

| Run | seed | postcard | solid | wild | nature20 | real20 | moy. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `SMonly_seed0` | 0 | 20.87 | 24.94 | 25.63 | 21.66 | 20.64 | 22.75 |
| `SMonly_seed1` | 1 | 21.47 | 25.07 | 25.21 | 22.01 | 20.35 | 22.82 |
| `SMonly_seed2` | 2 | 21.29 | 25.05 | 25.34 | 21.74 | 20.69 | 22.82 |
| **moyenne** | — | **21.21** ± 0.31 | **25.02** ± 0.07 | **25.39** ± 0.21 | **21.81** ± 0.18 | **20.56** ± 0.18 | **22.80** |
| Δ mixture | — | +0.40 | +1.00 | +0.61 | +1.19 | +1.78 | — |

**SSIM**

| Run | postcard | solid | wild | nature20 | real20 |
| --- | --- | --- | --- | --- | --- |
| `SMonly_seed0` | 0.854 | 0.911 | 0.912 | 0.834 | 0.791 |
| `SMonly_seed1` | 0.861 | 0.913 | 0.910 | 0.838 | 0.791 |
| `SMonly_seed2` | 0.858 | 0.909 | 0.909 | 0.837 | 0.792 |
| **moyenne** | **0.857** | **0.911** | **0.910** | **0.836** | **0.792** |

### SMnotone

rendu gamma seul (`tone=False`), pas de courbe ACR. Données : simulated (5 k), cible `t`, poids 0.8 / 0.1 / 0.1. Durée moyenne 17.7 min.

**PSNR (dB)**

| Run | seed | postcard | solid | wild | nature20 | real20 | moy. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `SMnotone_seed0` | 0 | 21.49 | 25.91 | 26.01 | 22.13 | 20.53 | 23.22 |
| `SMnotone_seed1` | 1 | 21.76 | 25.70 | 25.98 | 22.31 | 20.70 | 23.29 |
| `SMnotone_seed2` | 2 | 20.47 | 25.66 | 25.99 | 22.47 | 20.63 | 23.04 |
| **moyenne** | — | **21.24** ± 0.68 | **25.76** ± 0.14 | **25.99** ± 0.01 | **22.31** ± 0.17 | **20.62** ± 0.09 | **23.18** |
| Δ mixture | — | +0.43 | +1.74 | +1.21 | +1.69 | +1.84 | — |

**SSIM**

| Run | postcard | solid | wild | nature20 | real20 |
| --- | --- | --- | --- | --- | --- |
| `SMnotone_seed0` | 0.851 | 0.915 | 0.911 | 0.842 | 0.784 |
| `SMnotone_seed1` | 0.853 | 0.913 | 0.912 | 0.844 | 0.785 |
| `SMnotone_seed2` | 0.839 | 0.913 | 0.912 | 0.844 | 0.788 |
| **moyenne** | **0.848** | **0.914** | **0.912** | **0.843** | **0.786** |

### SMnoperc

sans terme perceptuel (`perceptual=0`). Données : simulated (5 k), cible `t`, poids 0.8 / 0.1 / 0.1. Durée moyenne 12.6 min.

**PSNR (dB)**

| Run | seed | postcard | solid | wild | nature20 | real20 | moy. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `SMnoperc_seed0` | 0 | 19.53 | 25.00 | 25.24 | 21.99 | 20.07 | 22.37 |
| `SMnoperc_seed1` | 1 | 19.06 | 23.67 | 24.87 | 21.87 | 20.05 | 21.90 |
| `SMnoperc_seed2` | 2 | 19.49 | 25.26 | 25.38 | 22.12 | 20.37 | 22.53 |
| **moyenne** | — | **19.36** ± 0.26 | **24.64** ± 0.85 | **25.16** ± 0.26 | **22.00** ± 0.13 | **20.16** ± 0.18 | **22.26** |
| Δ mixture | — | -1.45 | +0.62 | +0.38 | +1.38 | +1.38 | — |

**SSIM**

| Run | postcard | solid | wild | nature20 | real20 |
| --- | --- | --- | --- | --- | --- |
| `SMnoperc_seed0` | 0.819 | 0.906 | 0.900 | 0.836 | 0.781 |
| `SMnoperc_seed1` | 0.816 | 0.898 | 0.897 | 0.836 | 0.784 |
| `SMnoperc_seed2` | 0.817 | 0.908 | 0.901 | 0.837 | 0.783 |
| **moyenne** | **0.817** | **0.904** | **0.899** | **0.836** | **0.782** |

### BIAS1

reflets plus forts (vis jusqu'à 0.59). Données : simulated_biaised1 (5 k), cible `t`, poids 0.8 / 0.1 / 0.1. Durée moyenne 19.5 min.

**PSNR (dB)**

| Run | seed | postcard | solid | wild | nature20 | real20 | moy. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `BIAS1_seed0` | 0 | 19.57 | 25.36 | 25.54 | 22.38 | 20.30 | 22.63 |
| `BIAS1_seed1` | 1 | 19.10 | 25.25 | 25.62 | 22.17 | 20.38 | 22.50 |
| `BIAS1_seed2` | 2 | 19.04 | 25.71 | 25.82 | 22.22 | 20.59 | 22.68 |
| **moyenne** | — | **19.24** ± 0.29 | **25.44** ± 0.24 | **25.66** ± 0.14 | **22.26** ± 0.11 | **20.42** ± 0.15 | **22.60** |
| Δ mixture | — | -1.57 | +1.42 | +0.87 | +1.64 | +1.64 | — |

**SSIM**

| Run | postcard | solid | wild | nature20 | real20 |
| --- | --- | --- | --- | --- | --- |
| `BIAS1_seed0` | 0.835 | 0.915 | 0.909 | 0.844 | 0.778 |
| `BIAS1_seed1` | 0.825 | 0.914 | 0.911 | 0.841 | 0.777 |
| `BIAS1_seed2` | 0.825 | 0.917 | 0.912 | 0.843 | 0.784 |
| **moyenne** | **0.828** | **0.915** | **0.911** | **0.843** | **0.780** |

### BIAS2

reflets plus forts, second tirage. Données : simulated_biaised2 (5 k), cible `t`, poids 0.8 / 0.1 / 0.1. Durée moyenne 20.0 min.

**PSNR (dB)**

| Run | seed | postcard | solid | wild | nature20 | real20 | moy. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `BIAS2_seed0` | 0 | 18.93 | 25.44 | 25.70 | 22.07 | 20.60 | 22.55 |
| `BIAS2_seed1` | 1 | 18.42 | 25.48 | 25.72 | 22.27 | 20.57 | 22.49 |
| `BIAS2_seed2` | 2 | 18.69 | 25.38 | 25.62 | 22.27 | 20.65 | 22.52 |
| **moyenne** | — | **18.68** ± 0.26 | **25.43** ± 0.05 | **25.68** ± 0.06 | **22.20** ± 0.11 | **20.61** ± 0.04 | **22.52** |
| Δ mixture | — | -2.13 | +1.41 | +0.89 | +1.58 | +1.82 | — |

**SSIM**

| Run | postcard | solid | wild | nature20 | real20 |
| --- | --- | --- | --- | --- | --- |
| `BIAS2_seed0` | 0.828 | 0.917 | 0.909 | 0.841 | 0.782 |
| `BIAS2_seed1` | 0.817 | 0.914 | 0.912 | 0.844 | 0.781 |
| `BIAS2_seed2` | 0.826 | 0.916 | 0.910 | 0.844 | 0.783 |
| **moyenne** | **0.823** | **0.916** | **0.910** | **0.843** | **0.782** |

### BIAS2true

même données, cible = transmission non atténuée. Données : simulated_biaised2 (5 k), cible `true_t`, poids 0.8 / 0.1 / 0.1. Durée moyenne 18.5 min.

**PSNR (dB)**

| Run | seed | postcard | solid | wild | nature20 | real20 | moy. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `BIAS2true_seed0` | 0 | 20.13 | 25.65 | 25.45 | 22.18 | 20.72 | 22.82 |
| `BIAS2true_seed1` | 1 | 19.65 | 25.68 | 25.44 | 22.16 | 20.86 | 22.76 |
| `BIAS2true_seed2` | 2 | 19.49 | 25.64 | 25.60 | 22.21 | 20.72 | 22.73 |
| **moyenne** | — | **19.76** ± 0.33 | **25.66** ± 0.02 | **25.50** ± 0.09 | **22.18** ± 0.03 | **20.76** ± 0.08 | **22.77** |
| Δ mixture | — | -1.05 | +1.64 | +0.71 | +1.57 | +1.98 | — |

**SSIM**

| Run | postcard | solid | wild | nature20 | real20 |
| --- | --- | --- | --- | --- | --- |
| `BIAS2true_seed0` | 0.832 | 0.912 | 0.910 | 0.844 | 0.782 |
| `BIAS2true_seed1` | 0.823 | 0.914 | 0.910 | 0.843 | 0.783 |
| `BIAS2true_seed2` | 0.824 | 0.916 | 0.909 | 0.843 | 0.783 |
| **moyenne** | **0.826** | **0.914** | **0.910** | **0.843** | **0.783** |

### NM

cible `true_t`, pas de clé `t` dans les fichiers. Données : simulated_not_modified (5 k), cible `true_t`, poids 0.8 / 0.1 / 0.1. Durée moyenne 17.6 min.

**PSNR (dB)**

| Run | seed | postcard | solid | wild | nature20 | real20 | moy. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `NM_seed0` | 0 | 19.46 | 24.32 | 25.12 | 22.09 | 20.38 | 22.27 |
| `NM_seed1` | 1 | 19.35 | 24.39 | 24.74 | 21.85 | 20.86 | 22.24 |
| `NM_seed2` | 2 | 19.77 | 24.16 | 24.79 | 21.78 | 20.47 | 22.19 |
| **moyenne** | — | **19.53** ± 0.22 | **24.29** ± 0.12 | **24.89** ± 0.21 | **21.91** ± 0.16 | **20.57** ± 0.25 | **22.24** |
| Δ mixture | — | -1.28 | +0.27 | +0.10 | +1.29 | +1.79 | — |

**SSIM**

| Run | postcard | solid | wild | nature20 | real20 |
| --- | --- | --- | --- | --- | --- |
| `NM_seed0` | 0.835 | 0.911 | 0.908 | 0.839 | 0.778 |
| `NM_seed1` | 0.832 | 0.907 | 0.905 | 0.839 | 0.784 |
| `NM_seed2` | 0.837 | 0.908 | 0.905 | 0.838 | 0.780 |
| **moyenne** | **0.835** | **0.909** | **0.906** | **0.839** | **0.781** |

### VOC

blend synthétique VOC + nature + real89. Données : VOC2012 (5 000 paires), cible `—`, poids 0.8 / 0.1 / 0.1. Durée moyenne 17.3 min.

**PSNR (dB)**

| Run | seed | postcard | solid | wild | nature20 | real20 | moy. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `VOCseed_0` | 0 | 20.38 | 25.46 | 25.56 | 21.87 | 20.61 | 22.78 |
| `VOCseed_1` | 1 | 19.68 | 25.57 | 25.25 | 22.04 | 20.53 | 22.61 |
| `VOCseed_2` | 2 | 20.74 | 25.52 | 25.50 | 22.11 | 20.22 | 22.82 |
| `VOCseed_3` | 3 | 20.17 | 25.33 | 25.24 | 22.07 | 20.44 | 22.65 |
| **moyenne** | — | **20.24** ± 0.44 | **25.47** ± 0.10 | **25.39** ± 0.17 | **22.02** ± 0.11 | **20.45** ± 0.17 | **22.71** |
| Δ mixture | — | -0.57 | +1.45 | +0.60 | +1.40 | +1.67 | — |

**SSIM**

| Run | postcard | solid | wild | nature20 | real20 |
| --- | --- | --- | --- | --- | --- |
| `VOCseed_0` | 0.856 | 0.911 | 0.905 | 0.839 | 0.785 |
| `VOCseed_1` | 0.851 | 0.913 | 0.903 | 0.840 | 0.786 |
| `VOCseed_2` | 0.860 | 0.912 | 0.907 | 0.840 | 0.782 |
| `VOCseed_3` | 0.854 | 0.911 | 0.905 | 0.840 | 0.781 |
| **moyenne** | **0.855** | **0.912** | **0.905** | **0.840** | **0.783** |

### VOConly

VOC seul, sans photos réelles. Données : VOC2012 (5 000 paires), cible `—`, poids 1.0 / 0 / 0. Durée moyenne 18.1 min.

**PSNR (dB)**

| Run | seed | postcard | solid | wild | nature20 | real20 | moy. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `VOConly_seed0` | 0 | 20.38 | 24.36 | 24.15 | 19.75 | 19.73 | 21.67 |
| `VOConly_seed1` | 1 | 20.30 | 24.18 | 23.95 | 19.45 | 19.89 | 21.55 |
| `VOConly_seed2` | 2 | 20.74 | 24.27 | 24.40 | 19.47 | 19.85 | 21.74 |
| **moyenne** | — | **20.47** ± 0.23 | **24.27** ± 0.09 | **24.17** ± 0.22 | **19.56** ± 0.17 | **19.82** ± 0.08 | **21.66** |
| Δ mixture | — | -0.34 | +0.25 | -0.62 | -1.06 | +1.04 | — |

**SSIM**

| Run | postcard | solid | wild | nature20 | real20 |
| --- | --- | --- | --- | --- | --- |
| `VOConly_seed0` | 0.856 | 0.902 | 0.891 | 0.782 | 0.769 |
| `VOConly_seed1` | 0.856 | 0.901 | 0.893 | 0.786 | 0.773 |
| `VOConly_seed2` | 0.860 | 0.901 | 0.898 | 0.782 | 0.774 |
| **moyenne** | **0.858** | **0.902** | **0.894** | **0.783** | **0.772** |
