#!/usr/bin/env bash
# File d'attente : autant de configs qu'on veut, un run à la fois par GPU.
# Contrairement à sweep.sh (une vague, une config par GPU), on peut en empiler
# vingt : chacune part sur le premier GPU qui se libère.
#
#     tmux new -s nuit
#     ./queue.sh sweep/*.json
#     Ctrl-B puis D          détache ; on peut fermer le terminal et éteindre son PC
#     tmux attach -t nuit    pour revenir voir où ça en est
#
# Un run qui plante ne bloque pas la file : le worker enchaîne sur la suivante.
# EXTRA="--force" ./queue.sh ...  pour écraser des runs/<name>/ existants.

# Pas de -e : un run en échec ne doit pas tuer la file.
set -uo pipefail

# Surchargeable sans éditer le fichier :  GPUS="1 2 5 6" ./queue.sh sweep/*.json
# Le 4 sort des « misaligned address » / « illegal memory access » : ne pas le mettre.
read -ra GPUS <<< "${GPUS:-0 1 2 3}"

(( $# )) || { echo "usage: $0 sweep/*.json" >&2; exit 1; }

mkdir -p runs
QUEUE=$(mktemp /tmp/queue.XXXXXX)
LOCK="$QUEUE.lock"
: > "$LOCK"
printf '%s\n' "$@" > "$QUEUE"
trap 'rm -f "$QUEUE" "$LOCK"' EXIT

# Retire et renvoie la première ligne de la file. Sous verrou : les workers
# tournent en parallèle et doivent tirer des configs différentes.
pop() {
  exec 9>"$LOCK"
  flock 9
  local line=""
  if [ -s "$QUEUE" ]; then
    line=$(head -n1 "$QUEUE")
    sed -i 1d "$QUEUE"
  fi
  flock -u 9
  printf '%s' "$line"
}

worker() {
  local gpu=$1 cfg name code try seed_arg
  while cfg=$(pop); [ -n "$cfg" ]; do
    name=$(basename "$cfg" .json)

    # 3 tentatives. La 1re avec la config telle quelle ; les suivantes après
    # destruction du dossier du run raté, avec une seed tirée au hasard (un
    # crash CUDA n'a rien à voir avec la seed, et pour une étude multi-seeds
    # n'importe quelle seed fait l'affaire) et sans torch.compile, d'où
    # venaient les « misaligned address » / « illegal memory access ».
    # La seed réellement utilisée est enregistrée dans runs/<name>/config.json.
    for try in 1 2 3; do
      seed_arg=""
      if [ $try -gt 1 ]; then
        seed_arg="--seed $(( (RANDOM << 15) | RANDOM )) --no-compile --force"
        rm -rf "runs/$name"
        mv -f "runs/$name.launch.log" "runs/$name.crash$((try-1)).log" 2>/dev/null
        echo "[$(date +%H:%M)] GPU $gpu  reprise  $name (essai $try) $seed_arg"
      else
        echo "[$(date +%H:%M)] GPU $gpu  démarre  $name"
      fi

      # Un cache inductor par run : deux torch.compile simultanés dans le même
      # /tmp/torchinductor_$USER se corrompent mutuellement.
      TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${USER}_${name}" \
        python3 train.py --gpu "$gpu" --cfg "$cfg" --name "$name" ${EXTRA:-} $seed_arg \
        > "runs/$name.launch.log" 2>&1
      code=$?
      [ $code -eq 0 ] && break
    done

    if [ $code -eq 0 ]; then
      echo "[$(date +%H:%M)] GPU $gpu  fini     $name"
    else
      # Un run à moitié fait fausserait toute analyse qui balaie runs/*/hist.json :
      # on ne garde que les logs de crash.
      rm -rf "runs/$name"
      echo "[$(date +%H:%M)] GPU $gpu  ABANDON  $name après 3 essais -> runs/$name.crash*.log"
    fi
  done
  echo "[$(date +%H:%M)] GPU $gpu  file vide, worker terminé"
}

echo "$# configs à passer sur ${#GPUS[@]} GPUs (${GPUS[*]})"
for gpu in "${GPUS[@]}"; do worker "$gpu" & done
wait
echo
echo "[$(date +%H:%M)] terminé. Résultats dans runs/<name>/{log.txt,hist.json}"
