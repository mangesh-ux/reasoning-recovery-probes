# V2 Results

**Protocol ID:** v2-recovery-activation-probe-20260929  
**Status:** no V2 scientific model-backed execution has been run at this document's creation.

## Result discipline

This document will record only aggregate-safe results from the frozen V2 pipeline. It will never contain problem statements, answers, raw completions, token IDs, hidden-state tensors, private artifact paths, or model files.

The report structure is fixed before V2 results:

| Required result | Pre-execution state |
|---|---|
| Manifest and split integrity | Not yet measured |
| Synthetic hardware qualification | Not yet measured |
| Train/validation W_TO_C and W_TO_W counts | Not yet measured |
| Validation-selected regularization and layer | Not yet measured |
| Test B versus C comparison | Not yet measured |
| Within-problem matched result | Not yet measured |
| Complete layerwise curve | Not yet measured |
| Runtime, VRAM, GPU-hours, and activation storage | Not yet measured |
| Final scientific classification | Not yet measured |

## Required final reporting

When V2 reaches a terminal decision, the result record must include:

1. primary scientific classification: positive, null, confounded, underpowered, non-evaluable, or hardware block;
2. A, B, and validation-selected C metrics on their correct paired held-out cohort;
3. B-to-C deltas and clustered bootstrap intervals;
4. complete test-layer curve rather than a test-selected best layer;
5. W_TO_C and W_TO_W counts, denominators, attrition, and unique-problem counts by split;
6. matched problem-by-checkpoint group count, unique-problem count, concordance, and uncertainty;
7. runtime contract, throughput, allocator telemetry, estimated and observed storage, and total GPU-hours;
8. limitations and non-claims.

If a frozen gate fails, this document must preserve the failure classification and explain which prerequisite was absent. It must not substitute a different cohort, metric, layer, feature set, or protocol.
