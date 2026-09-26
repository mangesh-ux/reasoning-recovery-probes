# Reasoning Recovery Probes

This is a small, auditable implementation of **Pilot P0** for the Reasoning
Recovery Probes feasibility study. P0 asks whether the required data can be
collected reliably on a local 8 GB GPU. It is not a hidden-state experiment,
a probe-training experiment, or an early-stopping method.

## What P0 records

For each selected MATH-500 problem and stochastic rollout, P0 saves the
original generated token IDs once, evaluates the original final answer, and
creates fixed-token checkpoint prefixes from those saved IDs. It forces a
deterministic answer from each available prefix, evaluates it, and records one
of the four transition labels:

| Checkpoint answer | Original final answer | Label |
| --- | --- | --- |
| Wrong | Correct | \`W_TO_C\` |
| Wrong | Wrong | \`W_TO_W\` |
| Correct | Correct | \`C_TO_C\` |
| Correct | Wrong | \`C_TO_W\` |

Failures and non-evaluable answers remain explicit records; they are never
converted to incorrect answers or replaced. P0 reports operational evidence
only: completion counts, token counts, transition counts, runtime, peak VRAM,
failure statuses, and local artifact size.

## Deliberate exclusions

- No hidden-state collection during normal generation.
- No activation-prediction metrics, logistic regression, classifiers, or
  probe training.
- No semantic checkpointing based on words such as “wait” or “actually”.
- No automatic full \`30 problems x 4 rollouts\` execution.
- No reuse of P0 examples as confirmatory data without a new approved protocol.

## Review before any model-backed run

Read these files first:

1. [docs/research_question.md](docs/research_question.md)
2. [docs/pilot_p0_protocol.md](docs/pilot_p0_protocol.md)
3. [docs/decision_log.md](docs/decision_log.md)
4. [docs/artifact_schema.md](docs/artifact_schema.md)
5. [configs/pilot_p0.yaml](configs/pilot_p0.yaml)

The configuration is a proposed P0 contract, not a silently selected final
methodology. In particular, pin model and dataset revisions after the smoke
review and before a broader feasibility run.

## Installation

Create an isolated environment. Install a CUDA-enabled PyTorch build chosen
from the official PyTorch selector for the local driver before installing the
project extras; this project intentionally does not select a CUDA wheel for
you.

\`\`\`powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install the CUDA-enabled PyTorch command supplied by pytorch.org/get-started/locally.
python -m pip install -e ".[runtime,dev]"
\`\`\`

The model's published generation configuration uses sampling with temperature
0.6, top-p 0.95, and top-k 20; P0 records its configured values in every
artifact set. See the [Qwen3-1.7B model card](https://huggingface.co/Qwen/Qwen3-1.7B)
and its [generation configuration](https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/generation_config.json).

## Offline verification and controlled execution

These commands are supplied for review after implementation. They do not run
automatically:

\`\`\`powershell
# Offline synthetic/unit tests: no model or dataset download.
python -m unittest discover -s tests -v

# Create a deterministic 1-problem feasibility manifest (downloads MATH-500).
rrp prepare-manifest --config configs/pilot_p0.yaml --output artifacts/manifests/smoke.json --limit 1

# Manually run exactly one stochastic trajectory after reviewing the manifest.
rrp run --config configs/pilot_p0.yaml --manifest artifacts/manifests/smoke.json --max-problems 1 --max-rollouts 1 --confirm-run

# After review, create a small local feasibility subset.
rrp prepare-manifest --config configs/pilot_p0.yaml --output artifacts/manifests/small.json --limit 5
rrp run --config configs/pilot_p0.yaml --manifest artifacts/manifests/small.json --confirm-run

# Only after smoke/small review: prepare and manually run the configured P0 scope.
rrp prepare-manifest --config configs/pilot_p0.yaml --output artifacts/manifests/pilot_p0.json
rrp run --config configs/pilot_p0.yaml --manifest artifacts/manifests/pilot_p0.json --confirm-run
\`\`\`

The final two commands are intentionally manual. They preserve the declared
default scope rather than silently reducing it after observing runtime.

## Layout

\`\`\`text
configs/                 Reviewed P0 configuration
docs/                    Protocol, decisions, and artifact contract
src/reasoning_recovery/  Small runtime modules
tests/                   Offline synthetic/unit tests
scripts/                 Thin command wrapper
artifacts/               Local-only raw records and summaries (ignored by Git)
\`\`\`

Raw prompts, outputs, token IDs, dataset rows, and local runtime artifacts are
intentionally ignored by Git. Do not copy them into a public release.
