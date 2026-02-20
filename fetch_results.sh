#!/usr/bin/env bash
# Watch a RunPod instance running run_sweep.sh and continuously rsync new
# results to the local results/ directory.
#
# Usage:
#   POD_HOST=<ip-or-hostname> POD_PORT=<ssh-port> bash fetch_results.sh
#
# Optional:
#   POD_USER       SSH user (default: root)
#   REMOTE_REPO    Path to the cloned repo on the pod (default: ~/hw_bench)
#   LOCAL_RESULTS  Local destination directory (default: ./results)
#   POLL_INTERVAL  Seconds between rsync polls (default: 30)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

POD_HOST=${POD_HOST:?POD_HOST is required}
POD_PORT=${POD_PORT:?POD_PORT is required}
POD_USER=${POD_USER:-root}
REMOTE_REPO=${REMOTE_REPO:-"~/hw_bench"}
LOCAL_RESULTS="${LOCAL_RESULTS:-"$SCRIPT_DIR/results"}"
POLL_INTERVAL=${POLL_INTERVAL:-30}

SSH_OPTS="-e ssh -p ${POD_PORT}"
REMOTE="${POD_USER}@${POD_HOST}:${REMOTE_REPO}/results/"

mkdir -p "$LOCAL_RESULTS"

echo "Watching results on ${POD_USER}@${POD_HOST}:${POD_PORT}"
echo "  Remote: ${REMOTE_REPO}/results/"
echo "  Local:  ${LOCAL_RESULTS}/"
echo "  Poll:   every ${POLL_INTERVAL}s"
echo ""

do_rsync() {
    rsync -avz --ignore-existing \
        -e "ssh -p ${POD_PORT}" \
        "$REMOTE" "$LOCAL_RESULTS/"
}

# Check that the remote is reachable and results/ exists
if ! ssh -p "$POD_PORT" "${POD_USER}@${POD_HOST}" "test -d ${REMOTE_REPO}/results" 2>/dev/null; then
    echo "Remote results/ dir not found yet — waiting for sweep to start..."
fi

while true; do
    # Check if run_sweep.sh (or run_bench.sh) is still running on the pod
    sweep_alive=$(ssh -p "$POD_PORT" "${POD_USER}@${POD_HOST}" \
        "pgrep -f 'run_sweep.sh|run_bench.sh|benchmark_embedding.py' > /dev/null 2>&1 && echo yes || echo no")

    do_rsync || true

    if [[ "$sweep_alive" == "no" ]]; then
        echo ""
        echo "Sweep finished on pod. Running final sync..."
        do_rsync || true
        echo "Done. All results saved to $LOCAL_RESULTS"
        exit 0
    fi

    sleep "$POLL_INTERVAL"
done
