#!/bin/bash
# Headline test runs under ONE globally selected setting (plan step 2).
#
#   scripts/run_headline.sh <weighting> <alpha> <tau> [out-dir]
#   SEEDS="3 4" scripts/run_headline.sh ...   # extra seeds into a separate dir
#
# The setting comes from scripts/select_settings.py, which chooses it on D1
# *validation* data. It is applied unchanged to every draft source and both
# corpora. Test is decoded at tau* only, so no other threshold's test result
# exists to be picked from later.
set -euo pipefail
cd "$(dirname "$0")/.."
W=$1; A=$2; T=$3; OUT=${4:-results/refiner/headline}
mkdir -p "$OUT"
SEEDS=${SEEDS:-"0 1 2"}
# shellcheck disable=SC2206
seed_list=($SEEDS)
common=(--alphas 0 "$A" --seeds "${seed_list[@]}" --head-type binary --weighting "$W"
        --per-melody 3 --gated-thresholds "$T" --eval-splits test --save-artifacts)

run() {  # corpus source
  local corpus=$1 src=$2 tag
  tag=$(echo "$src" | tr -d '-' | tr 'A-Z' 'a-z')
  if [ "$corpus" = mtc ]; then
    .venv/bin/python scripts/train_refiner.py --sources "$src" "${common[@]}" \
      --out "$OUT/${corpus}_${tag}.json"
  else
    .venv/bin/python scripts/train_refiner.py --sources "$src" "${common[@]}" \
      --essen-test 0.2 --references essen_test --generator models/generator_ctrl.pth \
      --out "$OUT/${corpus}_${tag}.json"
  fi
}

# One lane per corpus/source pair would oversubscribe the GPU; three lanes fit.
( run mtc D1;   run essen D1 ) &
( run mtc D2-infill; run essen D2-infill ) &
( run mtc D3;   run essen D3 ) &
wait
echo "headline runs complete -> $OUT"
