#!/usr/bin/env bash
# Run vLLM embedding server and benchmark sweep for one model-hardware combo.
# Usage: MODEL=BAAI/bge-m3 HF_TOKEN=<token> HARDWARE=h100 bash run_bench.sh
set -e

# Resolve script directory so this script can be run from any cwd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../InferenceX/benchmarks/benchmark_lib.sh"

# ── Required env vars ────────────────────────────────────────────────────────
MODEL=${MODEL:?MODEL is required (e.g. BAAI/bge-m3)}
HF_TOKEN=${HF_TOKEN:?HF_TOKEN is required}

# ── Optional env vars with defaults ─────────────────────────────────────────
HARDWARE=${HARDWARE:-unknown}
CHUNK_SIZES=${CHUNK_SIZES:-"256,512"}
BATCH_SIZES=${BATCH_SIZES:-"1,4,16,64,256"}
CONCURRENCIES=${CONCURRENCIES:-"1,4,16,64"}
NUM_REQUESTS=${NUM_REQUESTS:-200}
PORT=${PORT:-8000}
EXTRA_VLLM_ARGS=${EXTRA_VLLM_ARGS:-""}

MODEL_SLUG="${MODEL//\//_}"
RESULT_DIR="${RESULT_DIR:-"$SCRIPT_DIR/../results/${MODEL_SLUG}__${HARDWARE}"}"
SERVER_LOG="$RESULT_DIR/server.log"

mkdir -p "$RESULT_DIR"

echo "=== Embedding Benchmark ==="
echo "Model:       $MODEL"
echo "Hardware:    $HARDWARE"
echo "Result dir:  $RESULT_DIR"
echo ""

# ── Start vLLM embedding server ──────────────────────────────────────────────
echo "Starting vLLM server on port $PORT ..."
HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
vllm serve "$MODEL" \
    --task embed \
    --port "$PORT" \
    --trust-remote-code \
    $EXTRA_VLLM_ARGS \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# ── Wait for server ready (reused from InferenceX/benchmarks/benchmark_lib.sh) ──
wait_for_server_ready \
    --port "$PORT" \
    --server-log "$SERVER_LOG" \
    --server-pid "$SERVER_PID"

echo "Server ready (PID $SERVER_PID)."

# ── Run benchmark sweep for each chunk size ──────────────────────────────────
for CHUNK_SIZE in ${CHUNK_SIZES//,/ }; do
    echo ""
    echo "--- chunk_size=$CHUNK_SIZE ---"
    python "$SCRIPT_DIR/benchmark_embedding.py" \
        --model "$MODEL" \
        --base-url "http://localhost:$PORT" \
        --chunk-size "$CHUNK_SIZE" \
        --batch-sizes "$BATCH_SIZES" \
        --concurrencies "$CONCURRENCIES" \
        --num-requests "$NUM_REQUESTS" \
        --result-dir "$RESULT_DIR"
done

# ── Shutdown server ───────────────────────────────────────────────────────────
echo ""
echo "Stopping server (PID $SERVER_PID) ..."
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

echo ""
echo "Done. Results in $RESULT_DIR"
