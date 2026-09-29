#!/usr/bin/env bash
# Execute Protocol V2 unchanged on a clean 24 GiB CUDA host.
#
# Usage (after reviewing the frozen protocol):
#   V2_CONFIRM_RUN=YES bash scripts/run_v2_cuda24.sh
#
# This runner intentionally refuses to substitute quantization, CPU offload,
# smaller cohorts, checkpoints, or decoding values.  It does not copy raw
# artifacts outside the ignored artifacts/v2 directory.

set -euo pipefail

if [[ "${V2_CONFIRM_RUN:-}" != "YES" ]]; then
  echo "Set V2_CONFIRM_RUN=YES after reviewing the frozen V2 protocol." >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "V2 runner requires a clean committed checkout." >&2
  exit 2
fi
git rev-parse --verify HEAD >/dev/null

python_bin="${V2_RUNNER_PYTHON:-python3}"
"$python_bin" - <<'PY'
import sys
try:
    import torch
except Exception as error:
    raise SystemExit(f"V2 requires the pinned torch runtime: {error}")
if not torch.cuda.is_available():
    raise SystemExit("V2 requires one available CUDA device")
if torch.cuda.device_count() != 1:
    raise SystemExit("V2 runner requires exactly one visible CUDA device")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("V2 requires BF16 support; no lower-precision fallback is allowed")
total = torch.cuda.get_device_properties(0).total_memory
minimum = 24 * 1024**3
if total < minimum:
    raise SystemExit(f"V2 runner requires >=24 GiB GPU memory; observed {total} bytes")
print(f"CUDA precheck passed: {torch.cuda.get_device_name(0)}, {total} bytes")
PY

config="configs/recovery_v2_frozen.yaml"
manifest="artifacts/v2/manifests/v2_deepmath_selection.json"
split="artifacts/v2/manifests/v2_deepmath_grouped_split.json"

"$python_bin" -m reasoning_recovery.v2_cli prepare-manifest --config "$config" --output "$manifest"
"$python_bin" -m reasoning_recovery.v2_cli prepare-split --config "$config" --manifest "$manifest" --output "$split"
"$python_bin" -m reasoning_recovery.v2_cli qualify-runtime --config "$config"
"$python_bin" -m reasoning_recovery.v2_cli preflight --config "$config" --confirm-run

"$python_bin" - <<'PY'
import json
from pathlib import Path
files = sorted(Path("artifacts/v2/qualification").glob("qualification-*.json"), key=lambda p: p.stat().st_mtime)
if not files:
    raise SystemExit("no V2 synthetic qualification receipt was created")
payload = json.loads(files[-1].read_text(encoding="utf-8"))
gate = payload.get("hardware_gate", {})
if gate.get("passed") is not True:
    raise SystemExit("synthetic qualification did not pass; preserving hardware block and stopping")
PY

"$python_bin" -m reasoning_recovery.v2_cli collect-train-validation --config "$config" --manifest "$manifest" --split "$split" --confirm-run
"$python_bin" -m reasoning_recovery.v2_cli select-model --config "$config" --manifest "$manifest" --split "$split"
"$python_bin" -m reasoning_recovery.v2_cli collect-test --config "$config" --manifest "$manifest" --split "$split" --confirm-run
"$python_bin" -m reasoning_recovery.v2_cli analyze-test --config "$config" --manifest "$manifest" --split "$split"
