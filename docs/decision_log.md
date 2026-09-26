# P0 decision log

This log distinguishes explicit implementation proposals from scientific
decisions that require review before model-backed execution.

| ID | Topic | Current proposal | Status before first smoke run |
| --- | --- | --- | --- |
| D1 | Study boundary | P0 is a separate feasibility-only repository; it cannot modify or inherit claims from the confidence-audit project. | Fixed by scope |
| D2 | Target model/dataset | Qwen3-1.7B and HuggingFaceH4/MATH-500, as requested. | Fixed by scope |
| D3 | Dataset/model revisions | Resolve and record Hub revisions; pin reviewed revisions before a broader run. | Requires review |
| D4 | Pilot selection | Hash rank \`(selection seed, problem ID, source index)\` and freeze a manifest before model calls. | Requires review |
| D5 | Base decoding | Thinking enabled; Qwen sampling defaults; 4,096-token proposed cap; four listed seeds. | Requires review |
| D6 | Checkpoint coordinate | Generated-token positions after the chat-template prompt, only while before detected \`</think>\`. | Requires review |
| D7 | Forced-answer semantics | Append tokenized \`</think>\\n\\n\` to the exact saved prefix, then greedy decode up to 512 tokens. | Requires review |
| D8 | Answer semantics | Last balanced \`\\boxed{...}\` only; missing/malformed boxes remain non-evaluable; use versioned \`math-verify\`. | Requires review |
| D9 | Failure policy | Retain OOM, cap, malformed, evaluator, and interruption statuses; no silent retry or replacement. | Fixed by scope |
| D10 | Activation interface | No activation collection or predictive analysis in P0. | Fixed by scope |
| D11 | Raw checkpoint coordinate | Count all generated token IDs after the initial native chat-template prompt, including any emitted `<think>` opening token. | Fixed by implementation |
| D12 | Resume behavior | A started request without a matching immutable receipt becomes `INTERRUPTED_UNKNOWN`; it is not regenerated. | Fixed by implementation |

## How to review a smoke run

Before invoking \`rrp run\`, inspect the generated manifest, this log, the exact
configuration hash, and the run command. The command's \`--confirm-run\` flag is
deliberate: it records that a human initiated the model-backed request. Do not
change a scientific field after observing a result in the same artifact set;
create a new reviewed configuration and preserve the original evidence.
