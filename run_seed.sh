#!/usr/bin/env bash
# Train + eval all adapter configs for one seed at one catalog size.
#   ./run_seed.sh <seed> [catalog_size=4] [model=openai/gpt-oss-20b]
# Run one instance per seed in parallel:
#   ./run_seed.sh 1 4  > logs/cat4_s1.log 2>&1 &
#   ./run_seed.sh 1 16 > logs/cat16_s1.log 2>&1 &
# Query dirs are shared across catalog sizes (catalogs are nested; files are per tool).
set -euo pipefail
SEED="${1:?usage: run_seed.sh <seed> [catalog_size] [model]}"
N="${2:-4}"
MODEL="${3:-openai/gpt-oss-20b}"
CAT="${CAT:-tools/catalog_64.json}"
RANK="${RANK:-32}"
QN="${QN:-queries_valid}"                       # natural (validated)
QS="${QS:-queries_strong_valid}"                # strong, seen paraphrases
QU="${QU:-queries_strong_unseen_valid}"         # strong, unseen paraphrases (held only)
OUT="${OUT:-results}"
COMMON="--catalog $CAT --catalog-size $N --queries $QN --model $MODEL"

# name : arm : trainer flags   (override with CONFIGS_FILE=path, one entry per line)
if [[ -n "${CONFIGS_FILE:-}" ]]; then
  mapfile -t CONFIGS < "$CONFIGS_FILE"
else
  CONFIGS=(
    "ex100_C:C:--examples-per-tool 100"
    "ex100_D:D:--examples-per-tool 100"
    "mix50_C:C:--examples-per-tool 50 --queries-strong $QS --strong-per-tool 50"
    "mix100_C:C:--examples-per-tool 100 --queries-strong $QS --strong-per-tool 100"
  )
fi

for cfg in "${CONFIGS[@]}"; do
  [[ -z "$cfg" || "$cfg" == \#* ]] && continue
  IFS=: read -r name arm flags <<< "$cfg"
  MTAG=""
  if [[ "$MODEL" != "openai/gpt-oss-20b" ]]; then
    MTAG="_$(basename "$MODEL" | tr -c 'A-Za-z0-9\n' '_')"     # e.g. _Inkling_Small
  fi
  tag="cat${N}_${name}${MTAG}_s${SEED}"
  if [[ -f "runs/$tag.json" ]]; then
    echo "=== $tag already trained, reusing checkpoint ==="
  else
    echo "=== [$(date +%H:%M:%S)] train $tag ==="
    # shellcheck disable=SC2086
    python train_sft.py $COMMON --rank "$RANK" --seed "$SEED" --tag "$tag" --arm "$arm" $flags
  fi
  ckpt=$(python3 -c "import json; print(json.load(open('runs/$tag.json'))['checkpoint'])")

  for pair in "seen:$QS:$tag" "unseen:$QU:${tag}_unseen"; do
    IFS=: read -r label qdir etag <<< "$pair"
    if ls "$OUT"/*"__${etag}_p"*.summary.json >/dev/null 2>&1; then
      echo "=== $etag ($label) already evaluated, skipping ==="; continue
    fi
    echo "=== [$(date +%H:%M:%S)] eval $etag ($label strong) ==="
    python eval_arms.py $COMMON --queries-strong "$qdir" --checkpoint "$ckpt" --arm "$arm" --tag "$etag" --out "$OUT"
  done
done
echo "=== [$(date +%H:%M:%S)] seed $SEED cat$N done ==="