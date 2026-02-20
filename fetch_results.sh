#!/usr/bin/env bash
# Watch a RunPod instance running run_sweep.sh and continuously rsync new
# results to the local results/ directory.
#
# Usage (RunPod SSH proxy):
#   POD_ID=doz07vedn2p7nh-64410a71 bash fetch_results.sh
#
# Usage (direct IP):
#   POD_HOST=216.81.245.29 POD_PORT=19508 bash fetch_results.sh
#
# Optional:
#   SSH_KEY        Path to SSH key (default: ~/.ssh/id_ed25519)
#   REMOTE_REPO    Path to the cloned repo on the pod (default: /hw_bench)
#   LOCAL_RESULTS  Local destination directory (default: ./results)
#   POLL_INTERVAL  Seconds between rsync polls (default: 30)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SSH_KEY=${SSH_KEY:-"$HOME/.ssh/id_ed25519"}
REMOTE_REPO=${REMOTE_REPO:-"/hw_bench"}
LOCAL_RESULTS="${LOCAL_RESULTS:-"$SCRIPT_DIR/results"}"
POLL_INTERVAL=${POLL_INTERVAL:-30}

# Build SSH command based on whether POD_ID or POD_HOST is provided
if [[ -n "${POD_ID:-}" ]]; then
    SSH_CMD="ssh -i $SSH_KEY ${POD_ID}@ssh.runpod.io"
    SSH_RSYNC="-e \"ssh -i $SSH_KEY\""
    REMOTE="${POD_ID}@ssh.runpod.io:${REMOTE_REPO}/results/"
    DISPLAY_TARGET="${POD_ID}@ssh.runpod.io"
elif [[ -n "${POD_HOST:-}" ]]; then
    POD_PORT=${POD_PORT:?POD_PORT is required when using POD_HOST}
    POD_USER=${POD_USER:-root}
    SSH_CMD="ssh -i $SSH_KEY -p $POD_PORT ${POD_USER}@${POD_HOST}"
    REMOTE="${POD_USER}@${POD_HOST}:${REMOTE_REPO}/results/"
    DISPLAY_TARGET="${POD_USER}@${POD_HOST}:${POD_PORT}"
else
    echo "Error: set POD_ID or POD_HOST" >&2
    exit 1
fi

mkdir -p "$LOCAL_RESULTS"

echo "Watching results on ${DISPLAY_TARGET}"
echo "  Remote: ${REMOTE_REPO}/results/"
echo "  Local:  ${LOCAL_RESULTS}/"
echo "  Poll:   every ${POLL_INTERVAL}s"
echo ""

do_rsync() {
    if [[ -n "${POD_ID:-}" ]]; then
        rsync -avz --ignore-existing \
            -e "ssh -i $SSH_KEY" \
            "$REMOTE" "$LOCAL_RESULTS/"
    else
        rsync -avz --ignore-existing \
            -e "ssh -i $SSH_KEY -p $POD_PORT" \
            "$REMOTE" "$LOCAL_RESULTS/"
    fi
}

# Check that the remote is reachable and results/ exists
if ! $SSH_CMD "test -d ${REMOTE_REPO}/results" 2>/dev/null; then
    echo "Remote results/ dir not found yet — waiting for sweep to start..."
fi

while true; do
    # Check if run_sweep.sh (or run_bench.sh) is still running on the pod
    sweep_alive=$($SSH_CMD \
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