# Completion-forecasting v1: CPU reproduction

This reproduces a separately versioned exploratory **thinking-completion**
question, not the non-evaluable original answer-recovery experiment. It requires
the private, verified original archive. No benchmark text or activations are
distributed in the public report. No cloud connection, language-model weights,
GPU allocation or test collection is required.

Use Python 3.12.3, NumPy 2.2.6, SciPy 1.14.0, scikit-learn 1.5.1, and the recorded
PyTorch 2.11.0+cu128 build solely for CPU tensor deserialization. The adapter sets
`CUDA_VISIBLE_DEVICES` empty, loads with `weights_only=True, map_location="cpu"`,
and validates the original BF16 content hash before lossless float32 conversion.
The analysis does not import PyTorch. New-runtime analysis is not claimed to
reproduce the original GPU runtime.

## Fixed identities and tests

- Protocol SHA256: `c9c63d608f1456b453ffccf79885c021c80acd18ee0401c68a1ae7220873f0eb`.
- Configuration SHA256: `b4b2cd75a5983ba04ab8a4d0e40f9f86253f656f4676def8334d5099a2006d4d`.
- Data-adapter SHA256: `fb2f48616a67582a10799757e6b4ed42caded86348e850af907c8a416c809261`.
- Analysis SHA256: `9a43b2c416cd7094745892b857804706c27c960e4f68c18e458835efacb723ea`.
- Original scientific-inventory SHA256: `b7782d22c207e8463fd4b41ecf772282a69f305ddc517c44239805df86198775`.

Run synthetic tests without accessing the private archive:

```powershell
python -B -m unittest discover -s tests -p test_completion_forecasting_data_v1.py -v
python -B -m unittest discover -s tests -p test_completion_forecasting_analysis_v1.py -v
```

The reviewed implementation passed 11 data-contract and 14 analysis-contract
tests before any completion-target validation enrollment or predictive fit.

## Archive audit and CPU export

From the repository root, provide the authenticated extraction directory, then
use a **fresh derivative directory** below `artifacts/completion-forecasting/v1/`.
Existing intents/results are write-once: do not delete them or rerun an uncertain
attempt. Audit and prepare each split separately; never select `test`.

```text
python -B scripts/completion_forecasting_data_v1.py
  --mode audit
  --extraction-root <private verified extraction>
  --output-dir artifacts/completion-forecasting/v1/<fresh-run>/train
  --split train
  --inventory-sha <inventory SHA above>
  --protocol-path docs/completion_forecasting_protocol_v1.md
  --protocol-sha256 <protocol SHA above>
  --helper-sha256 <adapter SHA above>
```

The displayed command is a parameter template, not a multiline shell command.
The audit prints only aggregate support and its full report hash. With that hash,
repeat the same parameters using `--mode prepare --audit-sha256 <audit hash>`.
Repeat audit/prepare for `validation` in its own directory. Preparation stops
without a modeling dataset if class/diversity gates fail. Every original payload
is size/hash checked before parsing; tensor shape, dtype, CPU device, content
hash and finiteness are rechecked. Neither phase accesses terminal readouts,
gold-answer values or original test bodies. Original checkpoint receipts contain
evaluator fields, but those fields are not indexed, copied or used as features.

Expected primary-checkpoint support, before and after tensor verification:

| Split | Rows | Close / cap | Problems with close / cap | Mixed-outcome problems |
|---|---:|---:|---:|---:|
| TRAIN | 576 | 151 / 425 | 52 / 90 | 46 |
| VALIDATION | 192 | 67 / 125 | 20 / 29 | 17 |

There was no activation attrition in this coordinate-512 analysis. All other
original coordinates and their unavailable cases remain preserved in the archive.

## Selection, locked models, then evaluation

The prepared report identifies `data.npz` by full SHA256. Run the analysis's
`select` command with only TRAIN data:

```text
python -B scripts/completion_forecasting_analysis_v1.py select
  --train <fresh-run>/train/data.npz --train-sha256 <train NPZ hash>
  --protocol docs/completion_forecasting_protocol_v1.md
  --protocol-sha256 <protocol SHA above>
  --output artifacts/completion-forecasting/v1/<fresh-run>/analysis
```

Five original-problem-grouped folds tune the fixed regularization grid. All 28
layers are evaluated on TRAIN only for Clex; secondary C uses that same selected
layer. Fold preprocessing is isolated. The immutable selection includes every
candidate score and a hash of the final TRAIN-only model/preprocessing file.
Convergence/integrity failures stop the namespace; no surviving-subset selection.

Only after the selection is complete and verified, run `evaluate` with the same
TRAIN, protocol and output parameters plus:

```text
--validation <fresh-run>/validation/data.npz
--validation-sha256 <validation NPZ hash>
--selection-sha256 <selection JSON hash>
```

Evaluation restores frozen linear weights without any fitting or preprocessing
updates. It produces private predictions and an aggregate `heldout-result.json`.
The original 32 TEST problems are not generated, inspected or relabeled.

Primary uncertainty is a 2,000-resample, duplicate-preserving problem-cluster
bootstrap for paired Blex-minus-Clex log loss. Within-problem concordance uses
only problems with both outcomes; ties receive half credit. Read the protocol
for support gates, secondary comparisons, fixed-model uncertainty limitations
and the precise meaning of a positive or null association.

Raw derivative NPZ/model/row-level files are private. Public results must contain
reviewed aggregates, provenance hashes and limitations only. File hashes include
serialization metadata; reproduce numerical predictions and cohort membership,
not a promise of byte-identical timestamps across fresh replications.

## Original measurement audit and independent result verification

The separate standard-library-only
[`v2a1_measurement_audit_v1.py`](../scripts/v2a1_measurement_audit_v1.py) reproduces
original train/validation termination, extraction, evaluation and primary
eligibility counters from stored receipt metadata. It neither recomputes labels
nor reads original test receipts, answer text fields or activation tensors.
Every input is size/hash authenticated before parsing. Its source SHA256 is
`2fd038050bfea7cce612003bc0c9d330198d1c6019684cc7a4c9ec2e669f8274`.

Parameter template, using a fresh private output:

```text
python -B scripts/v2a1_measurement_audit_v1.py
  --extraction-root <private verified extraction>
  --private-output artifacts/measurement-reproduction/v1/<new-result>.json
  --inventory-sha <inventory SHA above>
```

The executed reproduction consumed 4,613 authenticated inputs, including the
inventory and recovery-verification bootstrap files. It reproduced zero
wrong→correct and wrong→wrong rows in each original train/validation partition.
Its private report SHA256 is
`e26d45e4e02a53e48ec6b16eaa17db3f3dab90ec491c50b45e02a9ea24d779c7`.
Ten synthetic auditor tests also passed; together the three new suites pass 35.

An independent calculation using saved derivative predictions, without fitting,
reproduced primary/secondary paired log-loss effects and the matched
concordance effects and intervals within `1e-12`. It also verified all 128 CV
receipts, the immutable source/model chain and the pre-evaluation model lock.
The private independent-verification receipt SHA256 is
`cfb5a07e1f66e8811ad532bd4483b165dcbbcd1f38b5c79ab877a1e34b100603`.
These checks do not convert the exploratory holdout into confirmatory evidence.
