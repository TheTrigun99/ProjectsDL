---
tags:
  - sirr
  - sweep
  - comparaisons
date: 2026-09-08
---

# Comparaisons

Retour à [[Sweep 2026-09-08]]. Chiffres bruts dans [[Sweep 2026-09-08 - Résultats]].

Écart de PSNR entre deux expériences, avec l'erreur-type de la différence. Un `+` marque les écarts supérieurs à 2 erreurs-types — avec 3 ou 4 seeds c'est **indicatif, pas un test**.

| Comparaison | postcard | solid | wild | nature20 | real20 | moy. |
| --- | --- | --- | --- | --- | --- | --- |
| **SM** − VOC | +0.45 | +0.07 | +0.52 + | +0.23 + | +0.33 + | **+0.32** |
| **SMonly** − VOConly | +0.74 + | +0.75 + | +1.23 + | +2.25 + | +0.74 + | **+1.14** |
| **SM** − SMonly | -0.51 | +0.53 + | +0.51 + | +0.45 + | +0.22 | **+0.24** |
| **VOC** − VOConly | -0.23 | +1.20 + | +1.22 + | +2.47 + | +0.62 + | **+1.06** |
| **SM_big** − SM | +0.64 | +0.05 | -0.10 | +0.09 | +0.21 + | **+0.18** |
| **BIAS1** − SM | -1.45 + | -0.10 | -0.25 + | +0.00 | -0.35 + | **-0.43** |
| **BIAS2** − SM | -2.02 + | -0.11 | -0.23 + | -0.05 | -0.17 + | **-0.52** |
| **BIAS1** − BIAS2 | +0.56 + | +0.01 | -0.02 | +0.05 | -0.18 + | **+0.08** |
| **BIAS2true** − BIAS2 | +1.08 + | +0.22 + | -0.18 + | -0.02 | +0.16 + | **+0.25** |
| **NM** − SM | -1.17 + | -1.25 + | -1.02 + | -0.34 + | -0.21 | **-0.80** |
| **SMnotone** − SM | +0.55 | +0.21 | +0.09 | +0.05 | -0.15 | **+0.15** |
| **SMnoperc** − SM | -1.34 + | -0.90 | -0.74 + | -0.26 + | -0.61 + | **-0.77** |

### Ce que chaque comparaison teste

| Comparaison | Question |
| --- | --- |
| **SM** − VOC | Simulation vs blend VOC, recette identique |
| **SMonly** − VOConly | Idem sans photos réelles, pour isoler la source |
| **SM** − SMonly | Apport des 20 % de photos réelles au simulé |
| **VOC** − VOConly | Le même apport, côté VOC |
| **SM_big** − SM | 5 000 → 15 000 exemples simulés |
| **BIAS1** − SM | Reflets plus forts (biaised1) |
| **BIAS2** − SM | Reflets plus forts (biaised2) |
| **BIAS1** − BIAS2 | Les deux tirages biaisés entre eux |
| **BIAS2true** − BIAS2 | Cible `true_t` vs `t`, mêmes données |
| **NM** − SM | Cible `true_t` de `simulated_not_modified` |
| **SMnotone** − SM | Rendu gamma seul vs courbe ACR |
| **SMnoperc** − SM | Retrait du terme perceptuel |

### Erreurs-types

| Comparaison | postcard | solid | wild | nature20 | real20 |
| --- | --- | --- | --- | --- | --- |
| **SM** − VOC | ± 0.34 | ± 0.12 | ± 0.10 | ± 0.08 | ± 0.10 |
| **SMonly** − VOConly | ± 0.22 | ± 0.07 | ± 0.18 | ± 0.14 | ± 0.11 |
| **SM** − SMonly | ± 0.31 | ± 0.12 | ± 0.13 | ± 0.12 | ± 0.12 |
| **VOC** − VOConly | ± 0.26 | ± 0.07 | ± 0.15 | ± 0.11 | ± 0.10 |
| **SM_big** − SM | ± 0.35 | ± 0.14 | ± 0.07 | ± 0.08 | ± 0.08 |
| **BIAS1** − SM | ± 0.31 | ± 0.18 | ± 0.10 | ± 0.09 | ± 0.10 |
| **BIAS2** − SM | ± 0.30 | ± 0.11 | ± 0.06 | ± 0.09 | ± 0.06 |
| **BIAS1** − BIAS2 | ± 0.22 | ± 0.14 | ± 0.09 | ± 0.09 | ± 0.09 |
| **BIAS2true** − BIAS2 | ± 0.24 | ± 0.03 | ± 0.06 | ± 0.07 | ± 0.05 |
| **NM** − SM | ± 0.29 | ± 0.13 | ± 0.13 | ± 0.11 | ± 0.16 |
| **SMnotone** − SM | ± 0.47 | ± 0.14 | ± 0.05 | ± 0.12 | ± 0.08 |
| **SMnoperc** − SM | ± 0.30 | ± 0.50 | ± 0.16 | ± 0.10 | ± 0.12 |

## Ce qui ressort

**1. La simulation bat le blend VOC**, surtout sur les jeux réels : `SM` − `VOC` = +0.33 dB sur real20 et +0.52 dB sur wild. Sans photos réelles l'écart explose : `SMonly` − `VOConly` = +1.14 dB en moyenne, dont +2.25 dB sur nature20.

**2. Les 20 % de photos réelles portent presque tout le transfert de VOC**, beaucoup moins celui du simulé : `VOC` − `VOConly` = +1.06 dB contre `SM` − `SMonly` = +0.24 dB. `VOConly` seul ne généralise pas ; `SMonly` tient debout tout seul.

**3. Tripler le volume simulé ne rapporte presque rien** : `SM_big` − `SM` = +0.18 dB, le seul écart net étant +0.21 dB sur real20. À 10 000 itérations et 1.3 M paramètres, ce n'est pas la taille du jeu qui limite.

**4. Entraîner sur des reflets plus forts dégrade le test**, surtout postcard : `BIAS2` − `SM` = -2.02 dB. Les jeux de test ont des reflets nettement plus faibles que les jeux biaisés — voir [[Sweep 2026-09-08 - Datasets]]. On entraîne à côté de la cible.

**5. `true_t` n'est pas une cible équivalente à `t`.** Sur les mêmes données, `BIAS2true` − `BIAS2` = +1.08 dB sur postcard mais -0.18 dB sur wild : la cible non atténuée déplace le compromis plus qu'elle ne l'améliore. `NM` est le pire groupe (-0.80 dB) mais pour une autre raison — son `true_t` est cassé, voir [[Sweep 2026-09-08 - Anomalies]].

**6. Le terme perceptuel gagne son coût.** `SMnoperc` − `SM` = -0.77 dB et -0.0143 de SSIM, pour 12.6 min au lieu de 17.7. PSNR **et** SSIM baissent : ce n'est pas un arbitrage netteté/fidélité.

**7. La courbe ACR n'apporte rien de mesurable** : `SMnotone` − `SM` = +0.15 dB, aucun écart au-delà du bruit inter-seed. Le rendu gamma seul suffit sur ces benchmarks.

## Bruit inter-seed

L'échelle à laquelle lire tout ce qui précède.

| Jeu | écart-type SM (n=4) | max − min (SM) | les deux runs seed 0 |
| --- | --- | --- | --- |
| postcard | 0.52 dB | 1.22 dB | 0.07 dB |
| solid | 0.22 dB | 0.50 dB | 0.43 dB |
| wild | 0.10 dB | 0.18 dB | 0.04 dB |
| nature20 | 0.12 dB | 0.30 dB | 0.11 dB |
| real20 | 0.12 dB | 0.25 dB | 0.25 dB |

> [!warning] Deux runs à seed identique ne donnent pas le même résultat
> `SMseed0_0` et `SMseed0_1` ont la même config et le même seed. Ils divergent de 0.43 dB sur solid et 0.25 dB sur real20 — les pertes divergent dès la 2e itération. Causes attendues : cuDNN benchmark, AMP, `torch.compile`, et le `random` global du dataset dont les workers ne sont pas seedés.
> 
> **Tout écart inférieur à ~0.3 dB sur solid/real20 et ~1 dB sur postcard n'est pas interprétable ici.**