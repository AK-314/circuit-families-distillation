#!/usr/bin/env bash
# One-command bootstrap, execution and evidence packaging for a fresh clone.

set -u

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
RESULT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/stage14-symbolica-probe.XXXXXX")"
BOOTSTRAP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/stage14-symbolica-bootstrap.XXXXXX")"
ARCHIVE="${PWD}/stage14-symbolica-probe-output.tar.gz"
LOG="${RESULT_DIR}/complete-terminal-output.txt"

cleanup() {
  rm -rf "$RESULT_DIR" "$BOOTSTRAP_DIR"
}
trap cleanup EXIT

run_probe() {
  set -e
  echo "===== STAGE 14 SYMBOLICA PRACTICE-NODE PROBE ====="
  test -n "${STAGE14_EXPECTED_SHA:-}" || {
    echo "FAIL: STAGE14_EXPECTED_SHA is required"
    return 2
  }
  test -n "$ROOT" || {
    echo "FAIL: run this command inside the cloned repository"
    return 2
  }
  cd "$ROOT"
  ACTUAL_SHA="$(git rev-parse HEAD)"
  echo "expected_sha=${STAGE14_EXPECTED_SHA}"
  echo "actual_sha=${ACTUAL_SHA}"
  test "$ACTUAL_SHA" = "$STAGE14_EXPECTED_SHA" || {
    echo "FAIL: checkout does not match the required probe SHA"
    return 2
  }
  test -z "$(git status --porcelain --untracked-files=no)" || {
    echo "FAIL: tracked checkout is not clean"
    git status --short --untracked-files=no
    return 2
  }
  echo "repository_guard=PASS"
  echo "===== BASE HARDWARE ====="
  uname -srm
  command -v nvidia-smi >/dev/null || {
    echo "FAIL: nvidia-smi is unavailable; this is not a CUDA practice node"
    return 2
  }
  nvidia-smi \
    --query-gpu=index,name,driver_version,memory.total \
    --format=csv,noheader
  df -h "$ROOT"
  python3 --version
  echo "===== LOCKED ENVIRONMENT ====="
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  if command -v uv >/dev/null; then
    UV_BIN="$(command -v uv)"
  else
    python3 -m venv "$BOOTSTRAP_DIR/uv-venv"
    "$BOOTSTRAP_DIR/uv-venv/bin/python" -m pip install --disable-pip-version-check "uv==0.11.28"
    UV_BIN="$BOOTSTRAP_DIR/uv-venv/bin/uv"
  fi
  "$UV_BIN" --version
  "$UV_BIN" sync --frozen --no-dev
  .venv/bin/python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("FAIL: locked PyTorch environment cannot see CUDA")
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    print(f"cuda_device_{index}={properties.name} vram_bytes={properties.total_memory}")
PY
  echo "===== SYNTHETIC REPRESENTATIVE PROBE ====="
  PYTHONPATH="$ROOT/src" .venv/bin/python scripts/run_stage14_symbolica_probe.py \
    --output-root "$RESULT_DIR/results"
  test -z "$(git status --porcelain --untracked-files=no)" || {
    echo "FAIL: probe modified tracked repository files"
    git status --short --untracked-files=no
    return 2
  }
  echo "final_tracked_checkout_clean=PASS"
  echo "===== BOUNDARY ====="
  echo "registered_or_private_artifacts_accessed=false"
  echo "scientific_data=false"
  echo "production_eligible=false"
  echo "definitive_execution_started=false"
  echo "stage15_started=false"
}

set +e
run_probe 2>&1 | tee "$LOG"
STATUS="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$STATUS" >"$RESULT_DIR/exit-status.txt"
printf '%s\n' "${STAGE14_EXPECTED_SHA:-missing}" >"$RESULT_DIR/expected-source-sha.txt"

rm -f "$ARCHIVE"
tar -czf "$ARCHIVE" -C "$RESULT_DIR" .
if command -v sha256sum >/dev/null; then
  ARCHIVE_SHA="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
else
  ARCHIVE_SHA="$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')"
fi

echo "===== RETURN THIS ONE FILE TO ALEX ====="
echo "PROBE_EXIT_STATUS=$STATUS"
echo "OUTPUT_ARCHIVE=$ARCHIVE"
echo "OUTPUT_ARCHIVE_SHA256=$ARCHIVE_SHA"
echo "OUTPUT_ARCHIVE_BYTES=$(wc -c <"$ARCHIVE" | tr -d ' ')"
exit "$STATUS"
