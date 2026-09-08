#!/usr/bin/env bash
# Un run par fichier de config, un GPU par run, tout en parallèle.
#
#     ./sweep.sh sweep/*.json
#
# Le nom du run = le nom du fichier. Les logs vont dans runs/<name>/log.txt ;
# ce script ne garde que ce qui sortirait avant l'ouverture de ce fichier
# (crash à l'import, config invalide) dans runs/<name>.launch.log.
set -euo pipefail

GPUS=(5)          # les autres tournent pour daehyun / ksi (nvidia-smi)

(( $# )) || { echo "usage: $0 sweep/*.json" >&2; exit 1; }
(( $# <= ${#GPUS[@]} )) || {
  echo "$# configs pour ${#GPUS[@]} GPUs : lance-les en deux vagues" >&2; exit 1; }

mkdir -p runs
i=0
for cfg in "$@"; do
  name=$(basename "$cfg" .json)
  gpu=${GPUS[$i]}
  echo "GPU $gpu  <-  $name"
  # Un cache inductor par run : torch.compile écrit sinon dans le même
  # /tmp/torchinductor_$USER, et deux compilations simultanées s'y marchent
  # dessus (au mieux une erreur de compilation, au pire un kernel corrompu
  # qui part en « illegal memory access » au premier backward).
  TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${USER}_${name}" \
  nohup python3 train.py --gpu "$gpu" --cfg "$cfg" --name "$name" ${EXTRA:-} \
    > "runs/$name.launch.log" 2>&1 &
  i=$((i + 1))
done

echo
echo "lancé. suivi :  tail -f runs/*/log.txt   |   nvidia-smi -l 5"
wait
