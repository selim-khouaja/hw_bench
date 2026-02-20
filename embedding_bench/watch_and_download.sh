# Poll a RunPod pod until run_sweep.sh finishes, then rsync results locally.
#
# Usage:
#   POD_HOST=216.81.151.48 POD_PORT=12396 bash embedding_bench/watch_and_download.sh
#
# Optional overrides:
#   POD_USER=root REMOTE_DIR=/hw_bench/results LOCAL_DIR=./results POLL_INTERVAL=60

set -e

# Load pod.env if present and vars not already set
ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pod.env"
if [[ -f "$ENV_FILE" ]]; then
    echo "Loading $ENV_FILE"
    set -o allexport; source "$ENV_FILE"; set +o allexport
fi

POD_HOST=${POD_HOST:?POD_HOST is required (e.g. 216.81.151.48)}
POD_PORT=${POD_PORT:?POD_PORT is required (e.g. 12396)}
POD_USER=${POD_USER:-root}
REMOTE_DIR=${REMOTE_DIR:-/hw_bench/results}
LOCAL_DIR=${LOCAL_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../results"}
POLL_INTERVAL=${POLL_INTERVAL:-60}

SSH="ssh -o StrictHostKeyChecking=no -p $POD_PORT $POD_USER@$POD_HOST"

echo "=== Watch & Download ==="
echo "Pod:         $POD_USER@$POD_HOST:$POD_PORT"
echo "Remote dir:  $REMOTE_DIR"
echo "Local dir:   $LOCAL_DIR"
echo "Poll every:  ${POLL_INTERVAL}s"
echo ""

mkdir -p "$LOCAL_DIR"

echo "Ensuring rsync is installed on pod..."
$SSH "command -v rsync > /dev/null || (apt-get update -q && apt-get install -y rsync -q)"

echo "Syncing results continuously (Ctrl+C to stop)..."
while true; do
    rsync -avz --ignore-existing \
        -e "ssh -o StrictHostKeyChecking=no -p $POD_PORT" \
        "$POD_USER@$POD_HOST:$REMOTE_DIR/" \
        "$LOCAL_DIR/" \
        2>&1 | grep -v "^sending\|^sent\|^total\|^$"

    if ! $SSH "pgrep -f run_sweep.sh > /dev/null 2>&1"; then
        echo ""
        echo "Sweep finished. Running final sync..."
        rsync -avz --ignore-existing \
            -e "ssh -o StrictHostKeyChecking=no -p $POD_PORT" \
            "$POD_USER@$POD_HOST:$REMOTE_DIR/" \
            "$LOCAL_DIR/"
        echo ""
        echo "Done. Results in $LOCAL_DIR"
        break
    fi

    sleep "$POLL_INTERVAL"
done
