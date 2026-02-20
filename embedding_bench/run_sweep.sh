#!/usr/bin/env bash
# Orchestrate the full embedding benchmark sweep.
# Reads embedding-master.yaml and calls run_bench.sh for every model entry
# sequentially, using sweep parameters from the YAML.
#
# Usage:
#   HF_TOKEN=<tok> HARDWARE=h100 bash embedding_bench/run_sweep.sh
#
# Override the master config:
#   MASTER_CONFIG=path/to/embedding-master.yaml HF_TOKEN=<tok> HARDWARE=h100 bash run_sweep.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAILED_MODELS=()

MASTER_CONFIG="${MASTER_CONFIG:-"$SCRIPT_DIR/embedding-master.yaml"}"

HF_TOKEN=${HF_TOKEN:?HF_TOKEN is required}
HARDWARE=${HARDWARE:-unknown}
FORCE=${FORCE:-false}

# ── Parse models + sweep params from YAML ────────────────────────────────────
# Emits one TSV line per model: hf_model <TAB> framework <TAB> chunk_sizes <TAB> batch_sizes <TAB> concurrencies
mapfile -t MODEL_LINES < <(python3 - "$MASTER_CONFIG" <<'PYEOF'
import sys, yaml

with open(sys.argv[1]) as f:
    config = yaml.safe_load(f)

sweep = config.get("sweep", {})
global_chunk_sizes   = sweep.get("chunk-sizes",   [256, 512])
global_batch_sizes   = sweep.get("batch-sizes",   [1, 4, 16, 64, 256])
global_concurrencies = sweep.get("concurrencies", [1, 4, 16, 64])
num_requests         = str(sweep.get("num-requests", 200))

for key, model_cfg in config.get("models", {}).items():
    hf_model  = model_cfg["hf-model"]
    framework = model_cfg.get("framework", "vllm")
    # Per-model overrides; fall back to global sweep values
    chunk_sizes   = ",".join(str(x) for x in model_cfg.get("chunk-sizes",   global_chunk_sizes))
    batch_sizes   = ",".join(str(x) for x in model_cfg.get("batch-sizes",   global_batch_sizes))
    concurrencies = ",".join(str(x) for x in model_cfg.get("concurrencies", global_concurrencies))
    print(f"{hf_model}\t{framework}\t{chunk_sizes}\t{batch_sizes}\t{concurrencies}\t{num_requests}")
PYEOF
)

echo "Config:    $MASTER_CONFIG"
echo "Hardware:  $HARDWARE"
echo "Models:    ${#MODEL_LINES[@]}"
echo ""

# ── Phase 1: Validation sweep (first config for each model) ──────────────────
echo "=== Phase 1: Validation sweep (first config per model) ==="
VALIDATION_FAILED=()

for line in "${MODEL_LINES[@]}"; do
    IFS=$'\t' read -r hf_model framework chunk_sizes batch_sizes concurrencies num_requests <<< "$line"

    # Extract first value from each comma-separated list
    first_chunk="${chunk_sizes%%,*}"
    first_batch="${batch_sizes%%,*}"
    first_conc="${concurrencies%%,*}"

    # Skip if first-combo result file already exists
    model_slug="${hf_model//\//_}"
    result_dir="${RESULT_DIR:-"$SCRIPT_DIR/../results/${model_slug}__${HARDWARE}__${framework}"}"
    first_result="${result_dir}/${model_slug}__chunk${first_chunk}__bs${first_batch}__conc${first_conc}.json"

    if [[ -f "$first_result" && "$FORCE" != "true" ]]; then
        echo ">>> Validation skip: $hf_model — first config already present"
        continue
    fi

    echo ">>> Validating: $hf_model (chunk=$first_chunk batch=$first_batch conc=$first_conc)"
    if ! MODEL="$hf_model" \
        FRAMEWORK="$framework" \
        HF_TOKEN="$HF_TOKEN" \
        HARDWARE="$HARDWARE" \
        CHUNK_SIZES="$first_chunk" \
        BATCH_SIZES="$first_batch" \
        CONCURRENCIES="$first_conc" \
        NUM_REQUESTS="$num_requests" \
        FORCE="$FORCE" \
        bash "$SCRIPT_DIR/run_bench.sh"; then
        echo ">>> VALIDATION FAILED: $hf_model ($framework)"
        VALIDATION_FAILED+=("$hf_model ($framework)")
    fi
    echo ""
done

if [[ ${#VALIDATION_FAILED[@]} -gt 0 ]]; then
    echo "=== Validation failed — aborting full sweep ==="
    for m in "${VALIDATION_FAILED[@]}"; do
        echo "  - $m"
    done
    exit 1
fi

echo "=== All models passed validation ==="
echo ""

# ── Phase 2: Full sweep ───────────────────────────────────────────────────────
echo "=== Phase 2: Full sweep ==="

for line in "${MODEL_LINES[@]}"; do
    IFS=$'\t' read -r hf_model framework chunk_sizes batch_sizes concurrencies num_requests <<< "$line"

    echo ">>> Model=$hf_model  Framework=$framework"

    # Check if all expected result files already exist; skip server startup if so
    model_slug="${hf_model//\//_}"
    result_dir="${RESULT_DIR:-"$SCRIPT_DIR/../results/${model_slug}__${HARDWARE}__${framework}"}"
    all_done=$(python3 - "$result_dir" "$chunk_sizes" "$batch_sizes" "$concurrencies" <<'PYEOF'
import sys, itertools
from pathlib import Path

result_dir    = Path(sys.argv[1])
chunk_sizes   = sys.argv[2].split(",")
batch_sizes   = sys.argv[3].split(",")
concurrencies = sys.argv[4].split(",")
model_slug    = result_dir.name.split("__")[0]

missing = []
for cs, bs, c in itertools.product(chunk_sizes, batch_sizes, concurrencies):
    fname = f"{model_slug}__chunk{cs}__bs{bs}__conc{c}.json"
    if not (result_dir / fname).exists():
        missing.append(fname)

print("false" if missing else "true")
PYEOF
)

    if [[ "$all_done" == "true" && "$FORCE" != "true" ]]; then
        echo ">>> Skipping Model=$hf_model Framework=$framework (all results present)"
        echo ""
        continue
    fi

    if ! MODEL="$hf_model" \
        FRAMEWORK="$framework" \
        HF_TOKEN="$HF_TOKEN" \
        HARDWARE="$HARDWARE" \
        CHUNK_SIZES="$chunk_sizes" \
        BATCH_SIZES="$batch_sizes" \
        CONCURRENCIES="$concurrencies" \
        NUM_REQUESTS="$num_requests" \
        FORCE="$FORCE" \
        bash "$SCRIPT_DIR/run_bench.sh"; then
        echo ">>> FAILED: Model=$hf_model Framework=$framework"
        FAILED_MODELS+=("$hf_model ($framework)")
    fi

    echo ""
done

echo "=== Sweep complete ==="
if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
    echo ""
    echo "Failed models:"
    for m in "${FAILED_MODELS[@]}"; do
        echo "  - $m"
    done
    exit 1
fi
