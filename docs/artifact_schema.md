# Artifact schema

All runtime artifacts live under the configured local \`artifacts/\` directory,
which is ignored by Git. The schema intentionally separates raw private
records from derived summaries.

## Run layout

\`\`\`text
artifacts/
  manifests/             Immutable selection manifests
  runs/<run_id>/
    run_manifest.json    Config, provenance, and immutable full campaign scope
    ledger.jsonl         Append-only request-intent and terminal events
    raw/rollouts/        One receipt per original stochastic trajectory
    raw/checkpoints/     One receipt per fixed checkpoint
    summary/summary-<hash>.json
                         Immutable, content-addressed operational aggregation
    runtime_contract.json
                         Resolved runtime and forced-cue contract
    writer.lock.json     Ephemeral exclusive-writer lease during an invocation
    setup/               Immutable qualification or orchestration failures
\`\`\`

## Stable identities

| Object | Stable inputs |
| --- | --- |
| \`problem_id\` | Dataset's configured stable ID plus source index in the manifest |
| \`rollout_id\` | \`problem_id\`, \`rollout_seed\`, and manifest hash |
| \`checkpoint_id\` | \`rollout_id\` and fixed \`checkpoint_token\` coordinate |
| \`run_id\` | Manifest hash, configuration hash, and immutable full campaign scope |

The filesystem filename is a safe hash-based rendering of the logical ID; the
full logical ID stays inside the JSON record.

## Base rollout receipt

Each receipt contains the stable IDs, problem metadata, prompt text, reference
answer, seed, exact prompt and generated token IDs, their SHA-256 hashes,
generation status, decoded completion, extracted final answer and evaluation
record, runtime, peak allocated/reserved VRAM, error status, model/tokenizer
provenance, config hash, source Git commit, and timestamp.

The checkpoint coordinate is explicitly the raw generated-token sequence after
the native chat-template prompt. It is not a content-only reasoning count: an
emitted `<think>` opening token is part of the coordinate.

## Checkpoint receipt

Each record references its original rollout and contains the fixed token
coordinate, availability status, exact prefix token count and SHA-256, forced
cue token IDs/hash, forced generated token IDs/hash, forced completion,
extracted answer/evaluation, `observed_transition_label` when evaluable, the
more conservative primary `transition_label` when eligible, an
`analytic_eligibility` decision and reason codes, runtime, peak VRAM, and
failure status. Prefix token IDs can always be reconstructed from
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

Answer extraction normally uses `EXTRACTED`, `MISSING_BOXED_ANSWER`,
`MALFORMED_BOXED_ANSWER`, or `EMPTY_BOXED_ANSWER`. An unexpected extraction
implementation fault is recorded as `ERROR` alongside an evaluation `ERROR`;
it does not erase the completed generation.

## Immutability and recovery

Records are created with exclusive file creation. Existing content is never
overwritten. The ledger records intent before a request and a terminal state
after it. A writer lease permits one process per campaign; a stale lease is
preserved for explicit inspection rather than removed automatically. If a
process stops after an intent starts but before a terminal receipt, the next
invocation preserves it as \`INTERRUPTED_UNKNOWN\` rather than quietly reissuing
the request.

Before a model request, the runner writes a runtime contract containing the
resolved Hub revisions, a local SHA-256 manifest of the cached model/tokenizer
assets, tokenizer chat-template hash, special IDs, effective generation EOS
IDs, CUDA/PyTorch/GPU facts, and the raw token IDs and hashes of the
close-think marker and forced cue. A resume must match this contract.

## Public-release rule

Do not add raw prompts, MATH rows, completions, token IDs, model weights,
caches, machine paths, secrets, or local logs to a public release. Any later
public report must use reviewed aggregate statistics only.
