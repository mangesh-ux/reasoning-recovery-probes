# V2 Hardware and Runtime Plan

**Protocol ID:** v2-recovery-activation-probe-20260929  
**Status:** synthetic-only qualification completed on the local GPU; local full-workload gate failed solely on the frozen 48-hour projection.

## Current local evidence

The local environment is a native Windows CUDA system with an NVIDIA GeForce RTX 3070 Ti Laptop GPU and 8 GiB reported VRAM. The installed stack reports BF16 support:

| Field | Observed pre-inference value |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 Ti Laptop GPU |
| GPU memory | 8,589,410,304 bytes, 8 GiB |
| CUDA compute capability | 8.6 |
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu128 |
| Transformers | 4.52.4 |
| Datasets | 4.3.0 |
| math-verify | 0.9.0 |
| scikit-learn | 1.5.1 |
| Qwen model topology | 28 transformer layers, width 2,048 |

P0 already demonstrated BF16 generation on this machine. Its evidence is useful but limited: P0 did not capture readout scores or execute all-layer activation forwards. Therefore P0 does not qualify the V2 workload.

## V2 memory-sensitive implementation

V2 must avoid two avoidable allocations:

1. A CausalLM forward that retains full-sequence logits at a long prefix.
2. An output-hidden-states request that retains every layer's full sequence representation.

V2 uses the base transformer with post-block hooks, no attention outputs, no cache, and immediate CPU transfer of only each final-token vector. It does not retain a layer-by-sequence activation collection.

One stored V2 activation matrix requires:

~~~text
28 layers x 2,048 values x 2 BF16 bytes = 114,688 bytes
~~~

At the maximum 4,800 available fixed checkpoint forwards, raw matrix payload is about 525 MiB before private-artifact metadata and filesystem overhead. This is an estimate, not a measured storage result.

## Synthetic-only qualification

Before any V2 scientific trajectory or DeepMath readout, the V2 runtime qualification runs in a separate private qualification namespace. It may load the pinned model and tokenizer but must not load DeepMath, create a V2 rollout ID, invoke the evaluator, save a scientific completion, or write scientific checkpoint receipts.

It uses a fixed non-benchmark synthetic chat input and synthetic token-length shapes. It records aggregate measurements only:

| Workload | Required configuration |
|---|---|
| Load-only | Pinned model/tokenizer, BF16, template and cue validation |
| Synthetic reasoning generation | V2 sampling settings, at most 128 generated tokens |
| Synthetic short readout | Frozen cue, greedy decoding, at most 64 generated tokens |
| Synthetic activation forward | Hook implementation at 512, 1,024, 2,048, and 4,096 input tokens |

Each measured workload has one warm-up and three timed iterations with CUDA synchronization. The qualification records tokens per second or forward passes per second, peak allocated/reserved CUDA memory, layer geometry, activation storage bytes, and failure state.

## Local pass/fail criteria

The local RTX 3070 Ti Laptop execution is permitted only when all criteria hold:

1. CUDA and BF16 are available.
2. The model, tokenizer, template, close marker, and readout cue match the pinned runtime contract.
3. All synthetic activation shapes succeed with the expected [28, 2048] BF16 representation.
4. No required synthetic workload raises CUDA OOM or a runtime integrity error.
5. Each mandatory activation forward retains at least 1 GiB reported allocator headroom.
6. The projected full V2 campaign requires no more than 48 wall-clock GPU hours, using the conservative maximum request count and measured rates.
7. The artifact volume plus a 25 percent overhead reserve fits inside the configured free-disk requirement.

Failure is a local hardware block. It does not authorize Q4, FP8, CPU offload, device-map auto placement, smaller checkpoints, reduced rollouts, shortened reasoning, altered cue, or modified decoder.

## Conservative workload estimate formula

After synthetic qualification, the runner calculates:

~~~text
base generation upper bound
  = 160 problems x 6 rollouts x 4,096 / measured base tokens per second

readout upper bound
  = (960 terminal readouts + 4,800 checkpoint readouts)
    x 64 / measured readout tokens per second

activation-forward bound
  = measured time at conservative prefix length
    x 4,800 checkpoint activation forwards

projected total
  = base generation bound + readout bound + activation-forward bound
~~~

The report labels this an upper-bound operational projection, not a scientific finding. Observed availability can only lower the actual readout and activation count; it is not used to weaken a pre-execution hardware gate.

## Portable 24 GiB CUDA path

If the laptop gate fails or the projection exceeds 48 hours, the supported next environment is a 24 GiB CUDA GPU, such as an RTX 3090, RTX 4090, RTX A5000, A10, or L4. The portable runner must:

1. use the same frozen model, tokenizer, dataset revision, BF16 dtype, prompt, sampling, checkpoints, readout, activations, and code commit;
2. assert one CUDA device, BF16 support, at least 24 GiB total memory, enough free disk, and the pinned package versions;
3. execute synthetic qualification before any scientific receipt;
4. store the cloud runtime contract separately and never mix its receipts with a local V2 run identity;
5. preserve all raw records only in the ignored artifact directory;
6. emit aggregate-safe reports only.

The 24 GiB runner is a portability and throughput route, not a relaxed scientific protocol.

The portable launcher is `scripts/run_v2_cuda24.sh`. It requires an explicit
`V2_CONFIRM_RUN=YES` acknowledgement, a clean committed checkout, exactly one
visible CUDA device with at least 24 GiB and BF16 support, runs the synthetic
qualification first, and stops without collection if the hardware gate fails.

## Measured local qualification

The local load-only runtime and all required synthetic workloads completed
without CUDA OOM. The all-layer hook returned the frozen [28, 2,048] BF16
matrix at every required input length. The worst observed allocator reservation
was 3,776,970,752 bytes, leaving 4,812,439,552 bytes of headroom, comfortably
above the 1 GiB minimum.

The conservative workload projection was 54.37 GPU hours:

| Component | Projected seconds |
|---|---:|
| 960 base trajectories at the 4,096-token cap | 175,106.75 |
| 5,760 deterministic readouts at the 64-token cap | 17,805.88 |
| 4,800 activation forwards at the 4,096-token measured rate | 2,818.66 |
| Total | 195,731.29 (54.37 hours) |

Because 54.37 hours exceeds the frozen local 48-hour maximum, the local V2
campaign is blocked. This is not an authorization to alter the protocol. The
portable 24 GiB path is the only prepared continuation route.
