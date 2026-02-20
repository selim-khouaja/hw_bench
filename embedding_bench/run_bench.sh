#!/usr/bin/env bash
# Run embedding server and benchmark sweep for one model-hardware-framework combo.
# Usage: MODEL=BAAI/bge-m3 HF_TOKEN=<token> HARDWARE=h100 bash run_bench.sh
#        FRAMEWORK=sglang MODEL=BAAI/bge-m3 HF_TOKEN=<token> HARDWARE=h100 bash run_bench.sh
# Resolve script directory so this script can be run from any cwd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Wait for server health endpoint ───────────────────────────────────────────
wait_for_server_ready() {
    local health_url="" server_log="" server_pid="" sleep_interval=5
    while [[ $# -gt 0 ]]; do
        case $1 in
            --health-url)    health_url="$2";   shift 2 ;;
            --server-log)    server_log="$2";   shift 2 ;;
            --server-pid)    server_pid="$2";   shift 2 ;;
            --sleep-interval) sleep_interval="$2"; shift 2 ;;
            *) echo "Unknown param: $1"; return 1 ;;
        esac
    done
    tail -f -n +1 "$server_log" &
    local TAIL_PID=$!
    until curl --output /dev/null --silent --fail "$health_url"; do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "Server died before becoming healthy."
            kill $TAIL_PID 2>/dev/null || true
            return 1
        fi
        sleep "$sleep_interval"
    done
    kill $TAIL_PID
}

# ── Required env vars ────────────────────────────────────────────────────────
MODEL=${MODEL:?MODEL is required (e.g. BAAI/bge-m3)}
HF_TOKEN=${HF_TOKEN:?HF_TOKEN is required}

# ── Optional env vars with defaults ─────────────────────────────────────────
HARDWARE=${HARDWARE:-unknown}
FRAMEWORK=${FRAMEWORK:-"vllm"}          # vllm | sglang
FORCE=${FORCE:-false}
CHUNK_SIZES=${CHUNK_SIZES:-"256,512"}
BATCH_SIZES=${BATCH_SIZES:-"1,4,16,64,256,512,1024,2048,4096,8192,16384,32768"}
CONCURRENCIES=${CONCURRENCIES:-"1,4"}
NUM_REQUESTS=${NUM_REQUESTS:-50}
PORT=${PORT:-8000}
# Backward-compat alias: EXTRA_VLLM_ARGS is honoured if EXTRA_SERVER_ARGS is not set
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-${EXTRA_VLLM_ARGS:-""}}"

MODEL_SLUG="${MODEL//\//_}"
RESULT_DIR="${RESULT_DIR:-"$SCRIPT_DIR/../results/${MODEL_SLUG}__${HARDWARE}__${FRAMEWORK}"}"
SERVER_LOG="$RESULT_DIR/server.log"

mkdir -p "$RESULT_DIR"

echo "=== Embedding Benchmark ==="
echo "Model:       $MODEL"
echo "Hardware:    $HARDWARE"
echo "Framework:   $FRAMEWORK"
echo "Result dir:  $RESULT_DIR"
echo ""

# ── Start embedding server ───────────────────────────────────────────────────
echo "Starting $FRAMEWORK embedding server on port $PORT ..."
case "$FRAMEWORK" in
  vllm)
    HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
    vllm serve "$MODEL" \
        --port "$PORT" \
        --trust-remote-code \
        $EXTRA_SERVER_ARGS \
        > "$SERVER_LOG" 2>&1 &
    HEALTH_URL="http://0.0.0.0:${PORT}/health"
    ;;
  sglang)
    HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
    python -m sglang.launch_server \
        --model-path "$MODEL" \
        --port "$PORT" \
        --is-embedding \
        --trust-remote-code \
        $EXTRA_SERVER_ARGS \
        > "$SERVER_LOG" 2>&1 &
    HEALTH_URL="http://0.0.0.0:${PORT}/health"
    ;;
  *)
    echo "Unknown FRAMEWORK: $FRAMEWORK (supported: vllm, sglang)" >&2
    exit 1
    ;;
esac
SERVER_PID=$!

# ── Wait for server ready ────────────────────────────────────────────────────
if ! wait_for_server_ready \
    --health-url "$HEALTH_URL" \
    --server-log "$SERVER_LOG" \
    --server-pid "$SERVER_PID"; then
    echo "ERROR: Server failed to start for $MODEL" | tee -a "$RESULT_DIR/errors.log"
    exit 1
fi

echo "Server ready (PID $SERVER_PID)."

# ── Run benchmark sweep for each chunk size ──────────────────────────────────
BENCH_ERRORS=0
for CHUNK_SIZE in ${CHUNK_SIZES//,/ }; do
    echo ""
    echo "--- chunk_size=$CHUNK_SIZE ---"
    if ! python "$SCRIPT_DIR/benchmark_embedding.py" \
        --model "$MODEL" \
        --base-url "http://localhost:$PORT" \
        --chunk-size "$CHUNK_SIZE" \
        --batch-sizes "$BATCH_SIZES" \
        --concurrencies "$CONCURRENCIES" \
        --num-requests "$NUM_REQUESTS" \
        --framework "$FRAMEWORK" \
        --result-dir "$RESULT_DIR" \
        $( [[ "$FORCE" == "true" ]] && echo "--force" ); then
        echo "ERROR: Benchmark failed for $MODEL chunk_size=$CHUNK_SIZE" | tee -a "$RESULT_DIR/errors.log"
        BENCH_ERRORS=$((BENCH_ERRORS + 1))
    fi
done

# ── Shutdown server ───────────────────────────────────────────────────────────
echo ""
echo "Stopping server (PID $SERVER_PID) ..."
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

echo ""
if [[ $BENCH_ERRORS -gt 0 ]]; then
    echo "Done with $BENCH_ERRORS error(s). Results in $RESULT_DIR"
    exit 1
else
    echo "Done. Results in $RESULT_DIR"
fi
