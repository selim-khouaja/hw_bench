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
chunk_sizes    = ",".join(str(x) for x in sweep.get("chunk-sizes",    [256, 512]))
batch_sizes    = ",".join(str(x) for x in sweep.get("batch-sizes",    [1, 4, 16, 64, 256]))
concurrencies  = ",".join(str(x) for x in sweep.get("concurrencies",  [1, 4, 16, 64]))
num_requests   = str(sweep.get("num-requests", 200))

for key, model_cfg in config.get("models", {}).items():
    hf_model  = model_cfg["hf-model"]
    framework = model_cfg.get("framework", "vllm")
    print(f"{hf_model}\t{framework}\t{chunk_sizes}\t{batch_sizes}\t{concurrencies}\t{num_requests}")
PYEOF
)

echo "=== Embedding Sweep ==="
echo "Config:    $MASTER_CONFIG"
echo "Hardware:  $HARDWARE"
echo "Models:    ${#MODEL_LINES[@]}"
echo ""

# ── Run bench for each model ──────────────────────────────────────────────────
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
