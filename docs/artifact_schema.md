# Artifact schema

All runtime artifacts live under the configured local \`artifacts/\` directory,
which is ignored by Git. The schema intentionally separates raw private
records from derived summaries.

## Run layout

\`\`\`text
artifacts/
  manifests/             Immutable selection manifests
  runs/<run_id>/
    run_manifest.json    Config, provenance, and execution scope
    ledger.jsonl         Append-only request-intent and terminal events
    raw/rollouts/        One receipt per original stochastic trajectory
    raw/checkpoints/     One receipt per fixed checkpoint
    summary/summary.json Derived operational aggregation
\`\`\`

## Stable identities

| Object | Stable inputs |
| --- | --- |
| \`problem_id\` | Dataset's configured stable ID plus source index in the manifest |
| \`rollout_id\` | \`problem_id\`, \`rollout_seed\`, and manifest hash |
| \`checkpoint_id\` | \`rollout_id\` and fixed \`checkpoint_token\` coordinate |
| \`run_id\` | Manifest hash, configuration hash, and declared execution scope |

The filesystem filename is a safe hash-based rendering of the logical ID; the
full logical ID stays inside the JSON record.

## Base rollout receipt

Each receipt contains the stable IDs, problem metadata, prompt text, reference
answer, seed, exact prompt and generated token IDs, their SHA-256 hashes,
generation status, decoded completion, extracted final answer and evaluation
record, runtime, peak allocated/reserved VRAM, error status, model/tokenizer
provenance, config hash, source Git commit, and timestamp.

## Checkpoint receipt

Each record references its original rollout and contains the fixed token
coordinate, availability status, exact prefix token count and SHA-256, forced
cue token IDs/hash, forced generated token IDs/hash, forced completion,
extracted answer/evaluation, transition label when evaluable, runtime, peak
VRAM, and failure status. Prefix token IDs can always be reconstructed from
the referenced base receipt; P0 does not duplicate them by default.

## Evaluation states

\`\`\`text
CORRECT
INCORRECT
NON_EVALUABLE
ERROR
\`\`\`

Only \`CORRECT\` and \`INCORRECT\` produce a Boolean correctness value. The other
states carry a reason and are not coerced to a false result.

## Immutability and recovery

Records are created with exclusive file creation. Existing content is never
overwritten. The ledger records intent before a request and a terminal state
after it. If a process stops after an intent starts but before a terminal
receipt, the next invocation preserves it as \`INTERRUPTED_UNKNOWN\` rather than
quietly reissuing the request.

## Public-release rule

Do not add raw prompts, MATH rows, completions, token IDs, model weights,
caches, machine paths, secrets, or local logs to a public release. Any later
public report must use reviewed aggregate statistics only.
