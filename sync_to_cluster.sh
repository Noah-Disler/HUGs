#!/usr/bin/env bash

set -Eeuo pipefail

# Sync this checkout to the MS cluster. The default mode performs an initial
# sync and then watches for changes. Use --once for a single sync.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
LOCAL_DIR="${SCRIPT_DIR}/"
REMOTE_HOST="mscluster-login"
REMOTE_CODE_DIR="/home-mscluster/ndisler/projects/HUGs"
REMOTE_DATA_DIR="/datasets/ndisler/HUGs"
WATCH_INTERVAL="${WATCH_INTERVAL:-3}"
MODE="watch"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: ./sync_to_cluster.sh [--once | --watch] [--dry-run]

  --once     Sync once and exit.
  --watch    Sync once, then sync after each local change (default).
  --dry-run  Show what rsync would change without modifying the cluster.
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

while (($#)); do
  case "$1" in
    --once) MODE="once" ;;
    --watch) MODE="watch" ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown option: $1" ;;
  esac
  shift
done

command -v ssh >/dev/null || die "ssh is not installed"
command -v rsync >/dev/null || die "rsync is not installed"
[[ "$REMOTE_CODE_DIR" == "/home-mscluster/ndisler/projects/HUGs" ]] || \
  die "refusing to sync to unexpected path: $REMOTE_CODE_DIR"

EXCLUDES=(
  ".git/"
  ".DS_Store"
  "**/__pycache__/"
  "*.pyc"
  "*.pyo"
  ".pytest_cache/"
  ".mypy_cache/"
  ".ruff_cache/"
  "venv"
  ".venv"
  "logdir"
  "slurm_logs"
  "datasets"
  "wandb"
  "dist"
  "*.egg-info"
  "MUJOCO_LOG.TXT"
  ".hugs-sync-root"
)

RSYNC_ARGS=(-az --delete --itemize-changes)
for pattern in "${EXCLUDES[@]}"; do
  RSYNC_ARGS+=(--exclude "$pattern")
done
if ((DRY_RUN)); then
  RSYNC_ARGS+=(--dry-run)
fi

prepare_remote() {
  if ((DRY_RUN)); then
    ssh -o BatchMode=yes "$REMOTE_HOST" \
      "test -f '$REMOTE_CODE_DIR/.hugs-sync-root'" || \
      die "remote sync root is not initialized; run once without --dry-run"
    return
  fi

  ssh -o BatchMode=yes "$REMOTE_HOST" bash -s -- \
    "$REMOTE_CODE_DIR" "$REMOTE_DATA_DIR" <<'REMOTE'
set -Eeuo pipefail

code_dir="$1"
data_dir="$2"

mkdir -p \
  "$code_dir" \
  "$data_dir/logdir" \
  "$data_dir/slurm_logs" \
  "$data_dir/venv" \
  "$data_dir/.cache/pip" \
  "$data_dir/.cache/conda"

link_dataset_dir() {
  local name="$1"
  local target="$data_dir/$name"
  local link="$code_dir/$name"

  if [[ -e "$link" && ! -L "$link" ]]; then
    echo "Refusing to replace non-symlink: $link" >&2
    exit 1
  fi
  ln -sfn "$target" "$link"
}

link_dataset_dir logdir
link_dataset_dir slurm_logs
link_dataset_dir venv
touch "$code_dir/.hugs-sync-root"
REMOTE
}

do_sync() {
  local output
  if ! output="$(rsync "${RSYNC_ARGS[@]}" -e "ssh -o BatchMode=yes" \
      "$LOCAL_DIR" "$REMOTE_HOST:$REMOTE_CODE_DIR/" 2>&1)"; then
    echo "$output" >&2
    return 1
  fi

  if [[ -n "$output" ]]; then
    echo "$output"
  fi
  echo "Synced at $(date '+%H:%M:%S')"
}

echo "HUGs -> MS cluster"
echo "Local:  $LOCAL_DIR"
echo "Remote: $REMOTE_HOST:$REMOTE_CODE_DIR/"
echo "Data:   $REMOTE_DATA_DIR/"

prepare_remote
do_sync
prepare_remote

[[ "$MODE" == "once" ]] && exit 0
((DRY_RUN)) && die "--dry-run cannot be combined with continuous watch mode"

echo "Watching for changes; press Ctrl+C to stop."
if command -v fswatch >/dev/null; then
  fswatch -o -l 1 \
    --exclude '/\.git/' \
    --exclude '/__pycache__/' \
    --exclude '\.py[co]$' \
    --exclude '/venv/' \
    --exclude '/\.venv/' \
    --exclude '/logdir/' \
    --exclude '/slurm_logs/' \
    --exclude '/wandb/' \
    "$LOCAL_DIR" | while read -r _; do
      do_sync || echo "Sync failed; waiting for the next change." >&2
    done
else
  echo "fswatch is not installed; using ${WATCH_INTERVAL}s polling."
  while sleep "$WATCH_INTERVAL"; do
    do_sync || echo "Sync failed; retrying in ${WATCH_INTERVAL}s." >&2
  done
fi
