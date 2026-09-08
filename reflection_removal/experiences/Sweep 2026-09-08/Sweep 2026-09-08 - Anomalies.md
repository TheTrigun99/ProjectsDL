---
tags:
  - sirr
  - sweep
  - anomalies
date: 2026-09-08
---

# Anomalies

Retour à [[Sweep 2026-09-08]]. Classées par impact sur l'interprétation.

| # | Anomalie | Gravité |
| --- | --- | --- |
| 1 | `simulated_big_biaised2` n'est pas biaisé | 🔴 |
| 2 | `true_t` cassé dans `biaised1` et `not_modified` | 🔴 |
| 3 | Fuite train vers test : une image de `real20` est dans `real89` | 🟠 |
| 4 | PSNR d'éval biaisé par le batching | 🟠 |
| 5 | `postcard` non exploitable pour sélectionner un modèle | 🟠 |
| 6 | Aucun poids sauvegardé | 🟡 |
| 7 | `SM` a 4 runs mais 3 seeds | 🟡 |
| 8 | Configs VOC trompeuses | 🟡 |
| 9 | Divers | 🟢 |

## 1. `simulated_big_biaised2` n'est pas biaisé

Malgré son nom, sa distribution de force de reflet est celle de `simulated`, pas celle de `simulated_biaised2`.

| Jeu | vis moy | vis max | part > 0.40 |
| --- | --- | --- | --- |
| `simulated` | 0.258 | 0.531 | 5.4 % |
| `simulated_big_biaised2` | 0.258 | 0.531 | 5.2 % |
| `simulated_biaised2` | 0.306 | 0.591 | 26.3 % |

Les deux `trials.parquet` le confirment : même critère d'acceptation, `vis` plafonné à 0.4000 exactement, mêmes taux de rejet à 0.1 % près (71.4 et 71.6 % rejetés sur le SSIM, 10.1 % acceptés dans les deux cas).

**Conséquence.** `SM_big` n'est pas « BIAS2 en plus gros », c'est « SM en plus gros ». La comparaison `SM_big` − `SM` est donc bien un test de volume à distribution constante, ce qui est utile — mais si l'intention était de tester « reflets forts + volume », ce run ne le fait pas. À renommer avant publication.

## 2. `true_t` a un gain incohérent dans `biaised1` et `not_modified`

Rapport moyenne(`true_t`) / moyenne(`t`) sur 400 exemples :

| Jeu | p10 | médiane | p90 | corr(t, true_t) médiane |
| --- | --- | --- | --- | --- |
| `simulated_biaised2` | 1.088 | 1.102 | 1.158 | 0.999 |
| `simulated_biaised1` | 0.438 | 1.015 | 2.023 | 0.998 |
| `simulated_not_modified` | 0.414 | 0.979 | 1.793 | 0.998 |

Dans `biaised2`, `true_t` est proprement la transmission **dé-atténuée** : un facteur serré autour de 1/(1−R) ≈ 1.10, exactement ce que `t = t * (1 - R_map)` retire dans `simulate_example()`.

Dans `biaised1` et `not_modified`, le même rapport varie d'un facteur 4 d'un exemple à l'autre alors que la **structure** reste parfaitement alignée (corrélation médiane 0.998). C'est donc un facteur d'échelle global aléatoire, pas un décalage de contenu.

**Cause la plus probable.** `true_t` y a été sauvé **avant** `compute_exposure()` (Func. S1). La ré-exposition normalise `m`, `t` et `r` ensemble pour amener moyenne(`m`) à `TAU = 0.1329` — et de fait moyenne(`m`) vaut 0.1329 partout — mais si `true_t` ne passe pas dans le même appel, il garde l'échelle de la scène d'origine, qui varie énormément.

> [!danger] Ce que ça invalide
> `NM` est le seul groupe entraîné sur ce `true_t`-là. On lui demande de deviner une exposition non déductible de l'entrée. C'est cohérent avec ce qu'on observe : `NM` a la plus grosse perte pixel du sweep (0.091 contre 0.050 pour `SM`) et le pire PSNR moyen (−0.80 dB).
> 
> **Le résultat de `NM` ne dit rien sur `true_t` comme cible** — il dit que ce fichier est cassé. Le vrai test de `true_t` est `BIAS2true`, et lui se tient (+0.25 dB vs `BIAS2`).

`BIAS1` n'est pas affecté : il utilise `t_key="t"`.

## 3. Fuite train vers test dans `real20`

`data_others/real89/real89/blended/77.jpg` est **identique octet à octet** à `robustsirr_test_dataset/real20/blended/58.jpg`, et pareil pour les deux `transmission_layer/`. Diff max = 0 sur les deux couches, 736x1133.

`real89` pèse 10 % de l'échantillonnage dans `SM`, `SM_big`, `SMnotone`, `SMnoperc`, `BIAS1`, `BIAS2`, `BIAS2true`, `NM` et `VOC`. Ces neuf groupes ont vu 1 des 20 paires de real20, soit **5 % du jeu de test**. `SMonly` et `VOConly` (poids 1/0/0) sont les seuls propres.

Aucun autre doublon : contrôle par hash sur toutes les paires train x test, aucun recouvrement de noms non plus.

## 4. Le PSNR d'éval est une moyenne de moyennes de batchs

`evaluate()` empile une moyenne par batch puis en fait la moyenne. Avec `batch_size=8`, un jeu dont la taille n'est pas multiple de 8 surpondère son dernier batch.

| Jeu | n | découpage | PSNR loggé | PSNR par image | biais |
| --- | --- | --- | --- | --- | --- |
| `nature20` | 20 | 8+8+**4** | 20.62 | 21.48 | **−0.86 dB** |
| `wild` | 101 | 12x8+**5** | 24.79 | 24.95 | −0.16 dB |
| `real20` | 20 | 8+8+**4** | 18.78 | 18.82 | −0.04 dB |
| `postcard` | 179 | 22x8+**3** | 20.81 | 20.82 | −0.01 dB |
| `solid` | 200 | 25x8 | 24.02 | 24.02 | 0 |

Le biais est **identique pour tous les runs**, donc les comparaisons de ces notes restent valides. En revanche les valeurs absolues, en particulier sur nature20, ne sont pas comparables aux chiffres publiés.

**Correctif** : accumuler somme et compte sur les images plutôt que de moyenner les moyennes de batchs, ou évaluer avec `batch_size=1`.

## 5. `postcard` n'est pas exploitable pour sélectionner un modèle

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

Sur `BIAS1` et `BIAS2`, le modèle est **sous la mixture dans 17 évals sur 21** : il abîme postcard plus souvent qu'il ne l'améliore. Partout, le meilleur point de la trajectoire dépasse le point final de 1.4 à 3.3 dB, et l'écart entre deux évals consécutives monte jusqu'à 4.62 dB (`BIAS1_seed0`, it 4000).

postcard a la fenêtre de reflets la plus étroite des cinq jeux (p10-p90 = 18.3 à 22.9 dB) et 179 images seulement : à prendre comme un diagnostic, pas comme un critère.

Les quatre autres jeux sont stables : sur `nature20` et `real20`, aucun groupe sauf `VOConly` ne passe sous la mixture.

## 6. Aucun poids sauvegardé

Les 35 runs ont `"ckpt": false`. `find runs -name ckpt.pt` ne renvoie rien : seuls `hist.json` et `log.txt` subsistent. Impossible de rejouer un modèle, de sortir des images, ou de refaire une éval en pleine résolution sans tout relancer — environ 10 h de GPU.

## 7. Le groupe `SM` a 4 runs mais 3 seeds

`SMseed0_0` et `SMseed0_1` sont deux exécutions du **même seed 0**. L'écart-type « inter-seed » de `SM` (n=4) mélange donc variance de seed et non-déterminisme d'exécution. Ce n'est pas grave, c'est même utile pour calibrer le bruit (voir [[Sweep 2026-09-08 - Comparaisons]]), mais l'écart-type de `SM` n'est pas homogène à celui des autres groupes.

## 8. Les configs VOC portent des paramètres qui ne servent pas

`VOCseed_*` et `VOConly_*` ont `simulated=false` mais gardent `npz_dir`, `t_key="t"`, `ev_jitter=[-1.9, 0.9]` et `tone=true` dans leur `config.json`. Dans `build_train_loader()`, la branche `simulated=False` n'utilise aucun de ces arguments : `**kw` est ignoré.

Ce n'est pas un bug d'exécution, mais deux différences réelles s'ajoutent à « simulé vs VOC » et se confondent avec :

- **pas de jitter d'exposition** côté VOC — le simulé en reçoit −1.9 / +0.9 EV, les photos réelles jamais ;
- **politique de crop différente** — VOC utilise `frac=(1.0, 1.0)`, champ complet, alors que `nature` et `real89` utilisent `frac=(0.4, 1.0)`.

À garder en tête : `SM` − `VOC` n'isole pas la seule source de données.

## 9. Divers, sans conséquence

- **Paires (t, r) dupliquées** dans les jeux simulés : 4 984 paires distinctes pour 5 000 fichiers, 14 823 pour 15 000. Les rendus diffèrent, seul le couple d'images source se répète.
- **`trials.parquet` compte plus d'acceptés que de fichiers** : 5 001 pour 5 000, 15 005 pour 15 000. Course sur le compteur dans la génération multi-thread ; les fichiers écrits sont corrects.
- **VOC contient 15 paires quasi dégénérées** sur 5 000, PSNR(M,T) supérieur à 30 dB et jusqu'à 70 dB : mixture ≈ transmission, reflet quasi nul. 0.3 %, négligeable.
- **`simulated` et `simulated_not_modified` sont le même tirage** : 1 487 mixtures identiques sur 1 500 comparées. `not_modified` remplace la clé `t` par `true_t` et n'a **pas** de clé `t`.
- **GPU 4 exclu de la file** dans `queue.sh` pour cause de *misaligned address / illegal memory access*. Tout le sweep a tourné sur le seul GPU 5, en séquentiel.

## Runs en cours, non inclus

Trois runs `dim=64` lancés à 11:34 sur les GPU 3, 5 et 6. Aucun terminé à l'écriture de ces notes. Ils changent **trois choses à la fois** par rapport à `SM_big` :

| Paramètre | SM_big | SM_big_*_64 |
| --- | --- | --- |
| `dim` | 32 | **64** |
| `iters` | 10 000 | **15 000** |
| poids | 0.8 / 0.1 / 0.1 | **0.96 / 0.02 / 0.02** |
| `eval_every` | 500 | 1 000 |

Le passage à 0.96 / 0.02 / 0.02 est le plus lourd de conséquence : il ramène presque à `SMonly`, dont on sait qu'il perd 0.2 à 0.5 dB sur solid, wild et nature20. Si le run gagne ou perd, on ne saura pas lequel des trois facteurs en est responsable.