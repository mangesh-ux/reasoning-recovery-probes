# Authorized P0 execution

**Status:** frozen before any model-backed generation on 2026-09-28.

This document records the owner-authorized, staged feasibility campaign. It
does not authorize activation collection, probe fitting, or any confirmatory
claim.

## Immutable campaign

- Configuration: `configs/pilot_p0_v1_frozen.yaml`.
- Manifest: one deterministic 30-problem MATH-500 selection manifest.
- Full campaign: 30 problems x 4 stochastic seeds (120 base trajectories).
- First stage: the canonical first 10 manifest problems x all four seeds
  (40 base trajectories).
- Extension: invoke the same manifest without `--max-problems`; the immutable
  campaign ID ensures receipts from the first stage are reused rather than
  regenerated.

## Predeclared extension gate

After the first stage, write an immutable aggregate-only RR-03 report and
extend automatically only when all of the following are true:

1. Model loading, cue validation, and all attempted base requests completed
   without CUDA OOM, setup failure, or interrupted-unknown status.
2. No base generation failed, and forced-generation failures excluding
   non-evaluable mathematical answers are at most 5% of attempted available
   checkpoints.
3. Peak CUDA reserved memory for generation remains at least 512 MiB below the
   device's reported total memory.
4. The stage-derived projection for the remaining 80 base trajectories and
   their forced requests is at most 12 hours of generation time.

Generation caps, malformed boxes, and non-evaluable answers never disappear
from the report. They do not constitute an execution error by themselves, but
they remain central to the later feasibility decision.

If this operational gate fails, preserve the 10 x 4 evidence, report the
failed criterion, and do not alter precision, checkpoints, caps, data, or
decoding to force an extension.
