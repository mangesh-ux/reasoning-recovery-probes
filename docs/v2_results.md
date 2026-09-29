# V2 Results

**Protocol ID:** v2-recovery-activation-probe-20260929  
**Status:** local hardware block recorded after synthetic-only qualification; no V2 scientific trajectory, DeepMath row, evaluator call, activation artifact, probe fit, or test result was collected.

## Result discipline

This document will record only aggregate-safe results from the frozen V2 pipeline. It will never contain problem statements, answers, raw completions, token IDs, hidden-state tensors, private artifact paths, or model files.

## Local hardware qualification outcome

The pinned runtime passed load-only qualification and every required
synthetic-only operation. The result is nevertheless a **local hardware
block** because the predeclared conservative full-campaign projection was
54.37 GPU hours, above the frozen 48-hour ceiling. This is an operational
result, not a recovery-representation result.

| Measurement | Observed aggregate-safe result |
|---|---:|
| Synthetic reasoning throughput | 22.46 generated tokens/s |
| Synthetic greedy-readout throughput | 20.70 generated tokens/s |
| Activation forward at 512 tokens | 0.0850 s/forward |
| Activation forward at 1,024 tokens | 0.1435 s/forward |
| Activation forward at 2,048 tokens | 0.2839 s/forward |
| Activation forward at 4,096 tokens | 0.5872 s/forward |
| Required synthetic measured iterations | 3/3 completed for every workload |
| CUDA OOM or integrity failure | None observed |
| Largest observed CUDA reserved peak | 3,776,970,752 bytes |
| Smallest observed allocator headroom | 4,812,439,552 bytes |
| Frozen headroom requirement | 1,073,741,824 bytes |
| Projected maximum activation payload | 550,502,400 bytes |
| Activation payload with 25% reserve | 688,128,000 bytes |
| Free local disk at qualification | 93,099,585,536 bytes |
| Predeclared local time ceiling | 48.00 GPU hours |
| Projected full V2 workload | 54.37 GPU hours |

The sole failed gate was wall-clock projection. The pipeline did not use Q4,
FP8, CPU offload, a shorter cap, fewer rollouts, fewer checkpoints, a changed
cue, or a changed evaluator to evade that gate. The immutable qualification
receipt is identified publicly by SHA-256
`9a46e9694329f481a3abc94ec1ff533eef810619ef773d904d8dad5abc9fdd3f`.

The separate portable 24 GiB CUDA runner is ready, but it has not been invoked
from this local machine. It must rerun synthetic qualification in its own
runtime namespace before collecting any scientific receipt.

## Scientific-result status

| Required result | Current state |
|---|---|
| Manifest and split integrity | Not run; no DeepMath selection was needed after the local hardware block |
| Synthetic hardware qualification | Completed; local time gate failed |
| Train/validation W_TO_C and W_TO_W counts | Not measured, not zero |
| Validation-selected regularization and layer | Not selected |
| Test B versus C comparison | Not run |
| Within-problem matched result | Not run |
| Complete layerwise curve | Not run |
| Observed primary scientific classification | Not evaluable locally because no scientific cohort was collected |
| Operational classification | Hardware block on the local 8 GiB RTX 3070 Ti Laptop GPU |

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
