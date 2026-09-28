# Pilot P0 protocol

**Status:** frozen for the authorized staged P0 campaign on 2026-09-28. The
executable contract is \`configs/pilot_p0_v1_frozen.yaml\`.

## Population and fixed scope

- Model: \`Qwen/Qwen3-1.7B\` revision
  \`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e\`, using native Transformers and
  the configured thinking template.
- Dataset: \`HuggingFaceH4/MATH-500\` revision
  \`6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be\`, \`test\` split.
- Full feasibility campaign: 30 deterministically selected problems and four
  stochastic rollouts per selected problem.
- Candidate checkpoint coordinates: 256, 512, 1024, 2048, and 3072 raw
  generated-token positions when they exist.

P0 is a feasibility pilot. The selection manifest is generated before model
requests, is immutable once written, and is separate from any future study
population. The first 10 problems x 4 seeds are a staged portion of the same
full 30-problem x 4-seed campaign, not a separate sample.

## Selection and provenance

\`prepare-manifest\` validates the expected dataset fields and ranks all source
rows by SHA-256 over the configured selection seed, stable problem ID, and
source index. It writes the selected rows' IDs and content hashes before a
model call. No answer, model output, confidence, runtime, or correctness value
participates in selection.

The frozen configuration records the reviewed immutable Hub revisions. The
selection manifest additionally records the observed dataset fingerprint; a
changed manifest or configuration starts a distinct artifact set.

## Base trajectory

For every \`(problem_id, rollout_seed)\` unit, P0 constructs the configured
thinking-mode prompt and samples exactly one trajectory. It saves the prompt
token IDs and the generated-token IDs on CPU. It does not request hidden
states, scores, or attentions. It records whether generation reached EOS,
reached the configured cap, raised CUDA OOM, or otherwise failed.

The sampling values are temperature 0.6, top-p 0.95, and top-k 20 with a
4,096-token cap. Sampling remains stochastic; P0 records seeds and provenance
but does not claim bitwise reproducibility across arbitrary software or
hardware.

## Fixed checkpoint prefixes

The coordinate is a count of raw generated model tokens after the chat-template
prompt token IDs. It includes an emitted \`<think>\` opening token when the
native template/model produces one; it is not a content-only reasoning-token
count. A checkpoint at the first token of the first detected \`</think>\` token
sequence is valid because its prefix excludes that marker. Any later coordinate
that includes any part of the marker is unavailable. For every candidate
coordinate, P0 writes an explicit available or unavailable record and an
SHA-256 integrity hash of the exact prefix.

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
retains an \`observed_transition_label\` when both answers are evaluable. Its
primary \`transition_label\` is more conservative: it additionally requires an
available checkpoint, an EOS-terminated base trajectory that contains the
close-think marker, and an EOS-terminated forced completion. Capped, malformed,
non-evaluable, evaluator-error, unavailable, and other terminal outcomes stay
in the artifacts with explicit exclusion reasons; they are not relabelled.

## Memory, failures, and resumption

Generation runs one request at a time under inference mode. Input/output token
tensors are moved to CPU and released promptly; normal generation does not
retain full hidden-state tensors. CUDA allocated/reserved peak memory is
captured for each request.

Each request has an append-only intent event and immutable receipt. Completed
records are never regenerated. Failed and interrupted-unknown records are kept
and are not silently retried. An atomic campaign writer lease prevents two
invocations from issuing the same request. A later bounded-to-full invocation
can continue units that were never started while preserving all prior evidence.

## P0 outputs

The derived report includes final-answer accuracy, trajectory lengths,
checkpoint availability, observed and primary transition rates,
non-evaluable/error/exclusion counts, generation throughput, peak VRAM,
failure types, same-problem contrast counts, and artifact disk use. It must
not report activation metrics, classifier results, or a scientific recovery
claim.

## Source and runtime qualification

The model-backed command requires a clean committed source tree. It resolves
the model and tokenizer Hub references to immutable commits before loading
either asset, then records the resulting runtime contract. A changed resolved
revision, tokenizer template, forced cue tokenization, or static runtime
identity does not silently continue an existing run directory.
