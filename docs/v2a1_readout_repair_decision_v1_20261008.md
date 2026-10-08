# V2-A1 answer-readout diagnosis and repair decision — v1

Date: 2026-10-08. Status: read-only diagnosis; prospective proposal, not execution authorization.

## Decision

Do not restart the original experiment. Its train/validation collection completed: 768 base trajectories, 768 terminal readouts, 3,840 checkpoint receipts and 3,793 activation matrices; 9,122 collection operations completed with no failed/interrupted collection operations. The selector then failed because there were **zero eligible wrong→correct or wrong→wrong rows**. No probe selection or test collection followed. This is a non-evaluable recovery endpoint, not evidence against activation predictiveness, and not an abruptly interrupted collection.

Recommend a **CPU-first reference-parsability check, followed conditionally by a small paired train-only readout diagnostic**. The preferred repair candidate is a fixed boxed-answer prefill on the exact saved prefixes; a cap-only arm isolates whether additional readout length is sufficient. Neither has been tested. Preserve the original operational failure, labels and conclusions unchanged.

## Diagnosis

This audit verified 6,882 existing train-only input payloads against the recovered inventory and inspected 15 representative readout texts across eight termination/extraction categories. A supplemental audit verified 3,456 train readout receipts for recorded evaluator reasons and timing. It did not run a model, evaluator or probe, load activation tensors, or inspect validation/test result bodies. Examples were descriptive category coverage, not a representative prevalence sample.

| TRAIN evidence | Finding |
|---|---|
| Terminal readouts | 574/576 reached the 64-token cap; only 2 reached EOS. |
| Available checkpoint readouts | 2,774/2,848 capped; 74 reached EOS. Another 32 coordinates were unavailable. |
| Checkpoint extraction | 85 complete boxes, 5 malformed boxes, 2,758 missing boxes. Fourteen complete boxes still ended at the cap. |
| Evaluation | Only 22 checkpoint answers were evaluable, including 20 with EOS. Of 85 extracted answers, 63 failed reference parsing; 51 of those had EOS. |
| Paired endpoint | One EOS/EOS pair was C→C; no W→C or W→W pair existed. |

**Cap and continued explanation are the strongest observed failure mechanisms.** Inspected capped outputs included unfinished derivations, boxes cut at their opening, and complete answers followed by more explanation. One terminal example emitted another close-thinking marker and began a proof after a box. Simply extending the cap could allow further reasoning rather than reveal an already-held answer. Absence of a literal `<think>` tag does not establish absence of reasoning.

**Reference parsing is a separate blocker.** The selected-answer preflight checks nonempty strings, not mathematical parsability. Recorded failures say the reference did not parse; they do not establish why its syntax failed or whether an adapter would be sound. The 63/85 count is conditioned on extracted outputs and must not be generalized to all reference answers. Do not alter gold answers, the evaluator or the cohort to remove this attrition.

**No template, prefix, EOS or boxed-extractor defect was demonstrated.** Every audited train prompt ended at the correct assistant boundary; all 576 base generations began with the opening thinking token. The readout inputs matched `saved prompt IDs + exact saved prefix IDs + [151668, 271]`, the frozen close-thinking cue. All 2,848 available train activation inputs matched `prompt + prefix`, without the cue. Independent balanced-box parsing agreed for all 3,424 completed train readouts. EOS classification matched saved IDs; EOS wins even if it occurs at token 64.

**Decoding was actually greedy.** Receipts record `do_sample=False`, cap 64; code sets one beam and resolved EOS IDs `[151645, 151643]`. The hash-matched pinned model generation configuration has sampling defaults but no non-neutral repetition, minimum-length or suppression settings; explicit greedy settings override sampling. The library's corresponding defaults are neutral. The archive does not record every resolved processor field, so this is a provenance gap, not proof that an inherited penalty caused the failure. All audited first-step entropies were finite. Qwen warns against greedy *thinking* generation; that supports caution about continued-reasoning readouts, not a demonstrated causal diagnosis here. [Pinned generation configuration](https://huggingface.co/Qwen/Qwen3-1.7B/raw/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/generation_config.json), [Transformers 4.52.4 defaults](https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/generation/configuration_utils.py), [Qwen model guidance](https://huggingface.co/Qwen/Qwen3-1.7B).

## Exact reusable artifacts

Frozen run: `v2-run-498f9d84f5b45d26d732`, under the verified recovered archive's `reasoning-recovery-probes/artifacts/v2a1/`. The scientific inventory identifies every file by full SHA256 and size; logical rollout/checkpoint IDs and prefix hashes supply the join keys.

| Artifact family | Reuse without model recomputation |
|---|---|
| `manifests/` and `runs/<run-id>/run_manifest.json`, `runtime_contract.json` | Same 160 identities, 96/32/32 split, provenance and fixed coordinates. No reselection or replacement. |
| `raw/rollouts/` | All 768 saved prompt/generated-ID sequences and terminal-prefix boundaries; 576 are TRAIN. Reconstruct prefixes by slicing saved IDs, never by decoding/re-tokenizing. |
| `raw/activation_receipts/`, `raw/activation_tensors/` | All 3,793 archived BF16 `[28, 2048]` matrices, including 2,848 TRAIN matrices, remain reusable for their exact pre-cue inputs. This audit verified train receipt/hash linkage, not new tensor-value validity. |
| `raw/terminal_readouts/`, `raw/checkpoint_readouts/` | Existing 64-token outputs, statuses, scores and failures are immutable reference measurements. They cannot supply the ungenerated continuation of a longer/new readout. |
| Structural A features | Difficulty, topic and original checkpoint coordinates/normalization remain reusable. |
| Observable B features | Preserve original values for the original instrument only. Recapture all B′ values for a revised instrument, including first-step entropy/margin, answer likelihoods/count and previous-checkpoint agreement. Full vocabulary logits were not archived, so alternative answer confidence cannot be recovered offline. |

There are no naturally emitted post-thinking terminal answers to reuse: base generation stopped at the first close marker or its reasoning cap. In TRAIN, 425/576 bases used cap-prefix terminal boundaries. Do not extend those base trajectories to manufacture terminal outcomes. A later full revised measurement would need all five readout coordinates to construct the original immediate-prior-available-checkpoint feature; the proposed two-coordinate pilot cannot substitute for that feature.

Identity anchors: source `926d076f3baf7b5d31328f8865bd70cefe8396c1`; canonical configuration hash `bbab807d3ead00739dcab7482eecb6df763e72fa0308c19b12f0a6e06f30a6b3`; canonical manifest hash `8eb97c603f9731a8f482cfc855ba048d03613de420c846eba5a11f28fa80dc22`; canonical split hash `bad1088f2857f421326c2ce4d7ff94d0f913fef2df4ddc3d8ab5b1edd2052685`. Physical archive SHA256: `213cb4055e3381a6cdff043d305b63ad0f36e735b32b043a8a7b61926065032d`.

## Bounded prospective diagnostic

Use a new `v2a1-readout-measurement-diagnostic-v1` namespace, not the original controller or its expired approval. Before any new outputs, lock six TRAIN problems per difficulty by ranking `SHA256("v2a1-readout-measurement-diagnostic-v1|" + problem_id)` within each difficulty. Retain every failure/unavailable case; no replacement.

**CPU-first gate:** under the unchanged pinned evaluator, at least 10/12 references must parse, including at least 5/6 in each difficulty. Otherwise stop this panel as a reference–evaluator representation block: no GPU spend, new hash seed/panel or replacement. A separate evaluator-representation proposal would be required. Parsing is not semantic validation; failed cases remain in the denominator, without implying global dataset corruption.

Use all six existing rollouts, checkpoint 512, checkpoint 2048 and the saved terminal prefix: **216 intended prefix slots per fresh arm**.

| Protocol | Exact prospective change | Interpretation |
|---|---|---|
| R0 reference | Reuse saved greedy-64 outputs; zero new calls. | Original failed instrument; no relabeling. |
| R1 cap isolation | Original cue/prompt/prefix/EOS/greedy decoder, cap 512. | Same checkpoint activation input; more readout computation may add reasoning. |
| R2 preferred candidate | Original prompt/prefix and close-cue IDs, then fixed `\boxed{` prefill; greedy cap 128. | Same pre-cue state, new answer-format intervention; an induced answer, not direct access to latent belief. |

Freeze the suffix token IDs and full resolved generation/processor configuration before launch. Use the same operator at checkpoint and terminal. R2 parsing combines the fixed opening box with model-generated completion, excludes the final EOS token, and accepts exactly one nonempty balanced box plus whitespace, followed by EOS. No prose, new thinking tags, candidate constraints, forced answer values or forced EOS. Do not score the fixed prefill as generated answer tokens; version its token-span/confidence policy separately. R1 also reports the original-style last-box criterion, but progression uses the common strict answer-only criterion.

Select by format/faithfulness only: prefer R2 if it passes; otherwise consider R1 only if it passes those same gates. Mask CORRECT versus INCORRECT as EVALUABLE until the operator-selection ledger is immutable; do not choose by correctness, recovery yield or probe performance. A base-template rewrite, earlier `/no_think` instruction, sampled reasoning extension, changed evaluator or relaxed cap-success rule is outside scope. The revised hypothesis concerns association with a later **same-operator induced terminal readout**, not natural recovery, latent belief or the original V2 label.

**Budget:** at most 432 new requests and 138,240 generated tokens. Historical TRAIN capped readouts at these coordinates pooled 35.18 tokens/s, including generation prefill but excluding loading/I/O/evaluation. Linear projection is about 1.09 GPU-hours; a planning allowance of twice that plus 0.5 hours is 2.68 hours. Set hard stops at **3 paid GPU-hours and USD1.60 incremental spend**, whichever is reached first, including startup/loading and backup. Historical compute plus both retained-volume charges would be approximately USD1.54 for three hours; this is not a live price or completion guarantee. Longer outputs/prefills may exceed the allowance. Account-wide actual spend must also remain below the USD29 authority and above the USD1.02 operational floor; continued stopped storage still bills. Require fresh pricing/funding and new dated execution authorization. Stop incomplete at the cap; no retry, extra arm or automatic top-up.

## Go/no-go and next boundary

These are **pilot feasibility gates**, not relaxed full-study criteria:

1. **Integrity:** 100% saved prompt/prefix/activation-input hash agreement. For R1, compare its first up-to-64 generated IDs with R0; any divergence blocks a claim that length alone was isolated until resolved. Do not silently replace the old reference.
2. **Measurement:** ≥80% strict EOS + answer-only box + EVALUABLE success separately for issued checkpoint and terminal requests; all errors count as failures. Report intended-slot availability by coordinate/difficulty. Require paired successful coverage for ≥9/12 problems, including ≥4/6 per difficulty, and ≤10% visible new derivation across **every issued output**, including caps/rejections. Freeze a two-reviewer rubric: new calculation/inference steps or proof/explanation count as derivation; a bare expression or answer restatement does not; disagreements count conservatively as derivation. Accepted outputs must also have zero visible derivation inside the box; grammar alone cannot establish that, or exclude hidden computation.
3. **State compatibility:** require ≥90% normalized-answer agreement with old evaluable EOS anchors if ≥10 anchors across ≥3 independent problems exist. A lower rate blocks continuity with the original instrument; scarcity is INCONCLUSIVE, not automatic abandonment. Either permits only reporting distinct-instrument feasibility, not claiming a compatible V2 repair. For stated-answer fidelity, preselect two of the 18 intended prefix slots per problem by a fixed hash rank before new outputs; never resample. Reviewers, blinded to arm/gold/evaluator/transition, mark explicit target-answer declarations (not intermediate equations or hypothetical candidates), otherwise NO_EXPLICIT_ANSWER. Require ≥90% agreement with those declarations and no extra derivation, with ≥10 assessable cases; otherwise fail or mark INCONCLUSIVE as appropriate. Hash identity proves unchanged inputs, not unchanged beliefs.
4. **Endpoint diversity, checked only after instrument selection:** ≥3 W→C and ≥3 W→W rows, each spanning ≥3 independent duplicate-cluster problems; no problem contributes over half of either class. Require ≥2 same-problem/same-coordinate groups across ≥2 problems containing both labels. Missing/evaluator-failed/capped cases are never wrong labels. Failure blocks progression; do not switch arms to improve class yield.

Twelve problems cannot meet the original TRAIN floor of 50 rows and 15 independent problems per class. A passing pilot authorizes no probe or test: it supports only proposing separately approved train-scale remeasurement and confirmation. Archived-trajectory remeasurement remains exploratory; locking a new cue does not make it prospective confirmation. Freeze the validation confirmation protocol before any new validation readout, with unchanged full-study event/diversity floors. Validation confirms only the locked operator; failure ends that namespace without another cue/cap/parser/cohort. Test remains ungenerated/uninspected until a new immutable selection/gate and explicit authorization. Version every new config, parser, runtime, receipt, B′ feature, label and analysis; reference original matrices read-only.

Continue a compatible-repair proposal only if format, parsability, compatibility and independent event yield pass within the cap. Inconclusive/disagreeing old anchors limit the result to distinct-instrument feasibility and require explicit reframing before further work. Abandon the state-readout endpoint if high extraction yields no diverse W→C/W→W cases, visible derivation dominates, stated-answer fidelity fails, or reference repair requires unjustified label changes. Preserve failures; do not search for easier problems, another model, more reasoning or a publishable activation effect.

Private evidence identities: train readout diagnosis `3bc9b02246ba47a61b4987b13e9e7b9cf380106388b0ceef83e8cfaf80d500c0`; evaluator/timing diagnosis `78fd34e0ab4f9f011916a0ca622d64b755a03bf82f8bb45a1191bd6301ea58de`. Raw examples and benchmark identities are not reproduced here.
