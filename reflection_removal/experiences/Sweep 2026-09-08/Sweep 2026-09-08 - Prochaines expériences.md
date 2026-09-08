---
tags:
  - sirr
  - sweep
  - plan
date: 2026-09-08
runs: 8
---

# Prochaines expériences

Retour à [[Sweep 2026-09-08]]. 8 runs, 4 GPUs, environ 40 min de mur.

## Ce que le sweep précédent a déjà tranché

| Levier | Effet mesuré | Verdict |
| --- | --- | --- |
| Volume simulé (5 k → 15 k) | +0.18 dB | épuisé |
| Capacité + itérations (`dim` 32→64, 10 k→15 k it) | +0.01 dB | épuisé |
| Courbe ACR | +0.15 dB | épuisé |
| Source (simulé vs VOC) | +0.32 à +1.14 dB | tranché, le simulé gagne |
| Terme perceptuel (0 → 0.1) | +0.77 dB | **jamais réglé au-delà de on/off** |
| Force des reflets (plus forts) | −0.43 à −0.52 dB | **testé dans un seul sens** |

Les trois `SM_big_*_64` terminés à 12:12 sont le point décisif : `dim=64` + 15 000 itérations
donnent **+0.01 dB** sur `SM_big` en `dim=32` / 10 000 it. Ni la capacité, ni l'optimisation,
ni le volume ne sont le goulot. Ce qui reste, c'est la **distribution des données**.

## Hypothèse testée

Le simulateur produit des reflets bien plus forts que les benchmarks.

| | vis moy | PSNR(M,T) médian | part < 18 dB |
| --- | --- | --- | --- |
| `simulated` (jeu actuel) | 0.258 | 17.18 | 59.7 % |
| Les 5 benchmarks réunis | — | 22.06 | 7.3 % |

On a testé de rendre les reflets **plus** forts (`BIAS1`, `BIAS2`) : ça dégrade, jusqu'à −2 dB
sur postcard. Personne n'a testé l'autre sens.

## Plan : dose-réponse sur la force du reflet

Trois sous-ensembles de **5 000 exemples** tirés de `simulated_big_biaised2` (15 000), qui ne
diffèrent **que** par `vis` = moyenne(r) / moyenne(m) en linéaire. Même générateur, même N,
même seed de tirage : le seul facteur qui bouge est la force du reflet.

| Jeu | vis moy | plage | PSNR(M,T) médian | part < 18 dB |
| --- | --- | --- | --- | --- |
| `simvis_low` | 0.159 | 0.062 – 0.210 | 20.33 | 12.6 % |
| `simvis_rand` | 0.258 | 0.067 – 0.526 | 17.19 | 58.9 % |
| `simvis_high` | 0.361 | 0.304 – 0.531 | 14.70 | 99.0 % |
| *benchmarks* | — | — | *22.06* | *7.3 %* |

`simvis_low` est le plus proche des benchmarks. `simvis_high` sert de **contrôle positif** :
s'il n'est pas nettement pire, le protocole ne mesure rien et il faut chercher ailleurs.

Construits par symlinks vers les `.npz` d'origine — coût disque nul, rien n'est dupliqué.
Chaque dossier contient un `manifest.json` qui donne le fichier source et le `vis` de chaque lien.

## Plan : réglage du terme perceptuel

Le passage 0 → 0.1 vaut +0.77 dB, le plus gros levier du sweep, et il n'a jamais été réglé.
`perc03` teste 0.1 → 0.3 sur la recette `SM`, dont on a 4 seeds comme contrôle.

## Les 8 runs

| Config | npz_dir | perceptual | seed |
| --- | --- | --- | --- |
| `vislow_s0` | `simvis_low` | 0.1 | 0 |
| `visrand_s0` | `simvis_rand` | 0.1 | 0 |
| `vishigh_s0` | `simvis_high` | 0.1 | 0 |
| `perc03_s0` | `simulated` | **0.3** | 0 |
| `vislow_s1` | `simvis_low` | 0.1 | 1 |
| `visrand_s1` | `simvis_rand` | 0.1 | 1 |
| `vishigh_s1` | `simvis_high` | 0.1 | 1 |
| `perc03_s1` | `simulated` | **0.3** | 1 |

Tout le reste est la recette `SM` : `dim=32`, `dim_mults=1,2,4`, `lr=1e-3`, 10 000 itérations,
batch 16, poids `0.8,0.1,0.1`, `tone=true`, `t_key=t`, `pixel=1.0`, `gradient=0.1`.

Une seule différence avec le sweep précédent : **`ckpt: true`**, qui corrige l'anomalie 6.
Environ 5 Mo par run.

## Lancement

```bash
cd ~/ProjectsDL/reflection_removal
tmux new -s sweep2
GPUS="0 1 2 3" ./queue.sh \
  sweep/vislow_s0.json sweep/visrand_s0.json sweep/vishigh_s0.json sweep/perc03_s0.json \
  sweep/vislow_s1.json sweep/visrand_s1.json sweep/vishigh_s1.json sweep/perc03_s1.json
```

L'ordre est délibéré : la première vague de 4 donne un seed complet sur chaque bras, donc
un résultat lisible même si on interrompt à mi-parcours.

`queue.sh` accepte maintenant `GPUS` depuis l'environnement, plus besoin d'éditer le fichier.
Le défaut est `0 1 2 3`. Le GPU 4 reste exclu, le 7 était occupé.

## Comment lire les résultats

Avec 2 seeds par bras, l'erreur-type d'une différence entre deux bras vaut environ
0.22 dB sur solid et 0.15 dB sur wild, nature20 et real20 — voir le bruit inter-seed dans
[[Sweep 2026-09-08 - Comparaisons]].

**Juger sur la moyenne de solid, wild, nature20 et real20. Ignorer postcard**, dont
l'écart-type inter-seed est de 0.52 dB (anomalie 5 dans [[Sweep 2026-09-08 - Anomalies]]).

Trois issues possibles :

- **`low` > `rand` > `high`** — le simulateur tape trop fort. Action : redescendre la fenêtre
  `vis` dans `measure()` (`vis_min, vis_max`) et régénérer, plutôt que d'ajouter des exemples.
- **`low` ≈ `rand`, `high` nettement pire** — la force du reflet ne limite que dans l'excès.
  Action : garder la recette, chercher ailleurs (réalisme géométrique, défocus, fantômes).
- **les trois se valent** — le protocole ne mesure rien à cette échelle. Action : suspecter que
  le plafond de 23.2 dB vient du modèle ou de la loss, pas des données.

## À corriger avant de publier quoi que ce soit

Indépendamment de ces 8 runs, voir [[Sweep 2026-09-08 - Anomalies]] :

1. La moyenne de moyennes de batchs dans `evaluate()` — coûte 0.86 dB sur nature20.
2. `real89/77.jpg` = `real20/58.jpg`, à retirer de `real89`.
3. Renommer `simulated_big_biaised2`, qui n'est pas biaisé.
