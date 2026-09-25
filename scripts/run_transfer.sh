#!/bin/bash
# Cross-source transfer under the validation-selected setting (plan step 5c).
#
#   scripts/run_transfer.sh <weighting> <alpha> <tau> [out-dir]
#
# A head trained on one corruption source is tested on another. If D3's
# advantage comes from learning our injection rules, a D3-trained head should
# lose it on D1, while a D1-trained head finds D3's errors about as well as its
# own. In-domain baselines are the headline runs (results/refiner/headline_r2).
set -euo pipefail
cd "$(dirname "$0")/.."
W=$1; A=$2; T=$3; OUT=${4:-results/refiner/transfer}
mkdir -p "$OUT"
common=(--alphas "$A" --seeds 0 1 2 --head-type binary --weighting "$W"
        --per-melody 3 --gated-thresholds "$T" --eval-splits test --save-artifacts)

run() {  # train-source test-source
  local tag
  tag="$(echo "$1" | tr -d '-' | tr 'A-Z' 'a-z')_to_$(echo "$2" | tr -d '-' | tr 'A-Z' 'a-z')"
  .venv/bin/python scripts/train_refiner.py --sources "$1" --test-sources "$2" \
    "${common[@]}" --out "$OUT/mtc_${tag}.json"
}

( run D1 D3; run D1 D2-infill ) &
( run D3 D1; run D3 D2-infill ) &
wait
echo "transfer runs complete -> $OUT"
