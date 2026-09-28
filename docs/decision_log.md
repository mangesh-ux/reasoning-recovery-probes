# P0 decision log

This log distinguishes explicit implementation proposals from scientific
decisions that require review before model-backed execution.

| ID | Topic | Current proposal | Status before first smoke run |
| --- | --- | --- | --- |
| D1 | Study boundary | P0 is a separate feasibility-only repository; it cannot modify or inherit claims from the confidence-audit project. | Fixed by scope |
| D2 | Target model/dataset | Qwen3-1.7B and HuggingFaceH4/MATH-500, as requested. | Fixed by scope |
| D3 | Dataset/model revisions | Pin MATH-500 at \`6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be\` and model/tokenizer at \`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e\`. | Frozen 2026-09-28 |
| D4 | Pilot selection | Hash rank \`(selection seed, problem ID, source index)\`; write one immutable 30-problem manifest before model calls. | Frozen 2026-09-28 |
| D5 | Base decoding | Thinking enabled; temperature 0.6, top-p 0.95, top-k 20, 4,096-token cap, and seeds 101–104. | Frozen 2026-09-28 |
| D6 | Checkpoint coordinate | Generated-token positions 256, 512, 1024, 2048, and 3072 after the chat-template prompt, only before detected \`</think>\`. | Frozen 2026-09-28 |
| D7 | Forced-answer semantics | Append tokenized \`</think>\\n\\n\` to the exact saved prefix, then greedy decode up to 512 tokens. | Frozen 2026-09-28 |
| D8 | Answer semantics | Last balanced \`\\boxed{...}\` only; missing/malformed boxes remain non-evaluable; use versioned \`math-verify\`. | Frozen 2026-09-28 |
| D9 | Failure policy | Retain OOM, cap, malformed, evaluator, and interruption statuses; no silent retry or replacement. | Fixed by scope |
| D10 | Activation interface | No activation collection or predictive analysis in P0. | Fixed by scope |
| D11 | Raw checkpoint coordinate | Count all generated token IDs after the initial native chat-template prompt, including any emitted `<think>` opening token. | Fixed by implementation |
| D12 | Resume behavior | A started request without a matching immutable receipt becomes `INTERRUPTED_UNKNOWN`; it is not regenerated. | Fixed by implementation |
| D13 | Staged campaign | The 10 x 4 first stage and 30 x 4 extension share the same full-manifest campaign identity; completed receipts are reused. | Frozen 2026-09-28 |
| D14 | Primary-label censoring | A primary transition requires available checkpoint, base and forced EOS termination, a base close-think marker, and evaluable answers. All other raw outcomes remain recorded as exclusions. | Frozen 2026-09-28 |

## How to review a smoke run

Before invoking \`rrp run\`, inspect the generated manifest, this log, the exact
configuration hash, and the run command. The command's \`--confirm-run\` flag is
deliberate: it records that a human initiated the model-backed request. Use
only \`configs/pilot_p0_v1_frozen.yaml\` for the authorized campaign. Do not
change a scientific field after observing a result in the same artifact set;
create a new reviewed configuration and preserve the original evidence.

## Post-campaign outcome

This section records the completed P0 outcome. It does not amend any frozen
pre-run decision above.

| ID | Topic | Recorded outcome | Status |
| --- | --- | --- | --- |
| D15 | Operational extension gate | The first 10 x 4 stage met all four predeclared criteria. Its immutable aggregate analysis receipt is summary-f6ffab5cf81ed60ec79ca888a3a355d982f92f801b4eecd50c750dd25d3e921e--9f55bafcf58f6e30.json: 40 completed bases, 153 available forced completions, zero failed/interrupted/OOM receipts, a 4,273,995,776 B peak reservation, and a proportional remaining-generation projection of 9.07 h against the 12 h gate. The remaining units of the same immutable 30 x 4 campaign then completed. | Passed |
| D16 | P0 statistical viability | The strict primary population contains two W_TO_C events from one problem and zero W_TO_W events. No same-problem/same-checkpoint group contains both labels. The observed-before-censoring sensitivity view also has zero W_TO_W events. | Scientific block |
| D17 | Later activation study | A final activation protocol, activation collection, probe fitting, and A/B/C evaluation are not authorized from this P0 outcome. P0 remains feasibility-only evidence. | Not frozen / not run |
