# Pilot P0 protocol

**Status:** implementation-ready draft; review the decision log before a
model-backed run.

## Population and fixed scope

- Model: \`Qwen/Qwen3-1.7B\` using native Transformers and the configured
  thinking template.
- Dataset: \`HuggingFaceH4/MATH-500\`, \`test\` split.
- Default feasibility scope: 30 deterministically selected problems and four
  stochastic rollouts per selected problem.
- Candidate checkpoint coordinates: 256, 512, 1024, 2048, and 3072 generated
  reasoning-token positions when they exist.

P0 is a feasibility pilot. The selection manifest is generated before model
requests, is immutable once written, and is separate from any future study
population.

## Selection and provenance

\`prepare-manifest\` validates the expected dataset fields and ranks all source
rows by SHA-256 over the configured selection seed, stable problem ID, and
source index. It writes the selected rows' IDs and content hashes before a
model call. No answer, model output, confidence, runtime, or correctness value
participates in selection.

The configuration permits an unpinned Hub revision only for a smoke review. It
records the observed dataset fingerprint and resolved model/tokenizer revision.
Before a larger P0 run, review those values and pin them in the configuration
or preserve the exact reviewed manifest. A changed manifest or configuration
starts a distinct artifact set.

## Base trajectory

For every \`(problem_id, rollout_seed)\` unit, P0 constructs the configured
thinking-mode prompt and samples exactly one trajectory. It saves the prompt
token IDs and the generated-token IDs on CPU. It does not request hidden
states, scores, or attentions. It records whether generation reached EOS,
reached the configured cap, raised CUDA OOM, or otherwise failed.

The default sampling values are Qwen3's published thinking-mode values:
temperature 0.6, top-p 0.95, and top-k 20. Sampling remains stochastic; P0
records seeds and provenance but does not claim bitwise reproducibility across
arbitrary software or hardware.

## Fixed checkpoint prefixes

The coordinate is a count of generated model tokens after the chat-template
prompt token IDs. A checkpoint exists only when the original saved generation
has at least that many tokens and its prefix is still before the first detected
\`</think>\` token sequence. For every candidate coordinate, P0 writes an
explicit available or unavailable record and an SHA-256 integrity hash of the
exact prefix.

No semantic marker chooses a checkpoint. A later final-answer phase, an early
EOS, a cap, or malformed thinking boundary is retained as a structured status,
not repaired into a different coordinate.

## Forced answer

For an available checkpoint, P0 reconstructs:

\`\`\`text
saved prompt token IDs + saved generated prefix token IDs + configured close-think cue token IDs
\`\`\`

The original trajectory is not generated again. P0 appends the configured
\`</think>\\n\\n\` cue and greedily generates a bounded final-answer continuation.
The cue text, token IDs, decoding settings, and output IDs are saved. This
operational forced-answer semantics is intentionally explicit in the
configuration; changing it defines a new P0 configuration.

## Answer extraction and evaluation

The primary extractor retains the last balanced \`\\boxed{...}\` expression in a
completion. Missing, empty, or malformed boxed answers are non-evaluable;
there is no fallback that guesses an answer from arbitrary prose. The primary
evaluator is \`math-verify\`, with its installed version and backend errors
recorded. An evaluator error or unavailable backend never becomes
\`INCORRECT\`.

P0 evaluates the original final completion and each forced completion. It
creates a transition label only if both have an evaluated Boolean correctness
value.

## Memory, failures, and resumption

Generation runs one request at a time under inference mode. Input/output token
tensors are moved to CPU and released promptly; normal generation does not
retain full hidden-state tensors. CUDA allocated/reserved peak memory is
captured for each request.

Each request has an append-only intent event and immutable receipt. Completed
records are never regenerated. Failed and interrupted-unknown records are kept
and are not silently retried. A later invocation can continue units that were
never started while preserving all prior evidence.

## P0 outputs

The summary reports attempted/completed problems and trajectories, base and
forced token totals, checkpoint availability by position, all transition
counts, non-evaluable/error counts, failures, trajectory runtime mean/p50,
peak VRAM, and local artifact disk use. It must not report activation metrics,
classifier results, or a scientific recovery claim.
