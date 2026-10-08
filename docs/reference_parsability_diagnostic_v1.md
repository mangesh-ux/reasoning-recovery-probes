# TRAIN reference-parsability diagnostic v1

Status: CPU-only diagnostic; separately versioned measurement audit, not a new
scientific run or permission for paid inference. Original V2-A1 remains frozen.

Before reading reference values, freeze the twelve TRAIN problems specified by
the existing repair decision: six per difficulty, ordered by
`SHA256("v2a1-readout-measurement-diagnostic-v1|" + problem_id)`. No replacement,
new seed, candidate answers, validation references or test references are allowed.

Recover only those source rows from the pinned local Parquet snapshot. Verify
the original source-provenance hash, each consumed physical shard hash, exact
question/reference/metadata hashes, source indices and frozen split membership.
Slice selected rows before conversion to Python objects; never display raw
questions, gold answers or parsed expressions. The existing Arrow 25.0.1 data
reader and exact symbolic stack are used in separate CPU processes.

Call the unchanged original `MathVerifyBackend.parse` once per reference, using
math-verify 0.9.0, latex2sympy2_extended 1.11.0, ANTLR 4.13.2, SymPy 1.14.0 and
mpmath 1.3.0. Keep `parsing_timeout=None, raise_on_error=True`; an external
30-second process watchdog protects the diagnostic, records a timeout as a
failure and never retries it. No equivalence evaluation or model is invoked.

The previously proposed original-backend gate requires nonempty parser output
for at least 10/12 references and 5/6 in each difficulty. It is a representation
feasibility gate, **not semantic validation**. Record symbolic and fallback
string counts separately: default parsing can return both, or a partial numeric
match. An undelimited LaTeX expression with only numeric parsed objects receives
a potential-partial-extraction flag; this flag is not a proven parser error.
Whole-expression coverage remains unverified and cannot be inferred from list
length or successful self-comparison.

Failure ends this fixed gate without GPU work. Any literal-wrapper or strict
parser study must be frozen separately before executing it; never alter the
original answers, evaluator or labels. Store new immutable panel, intent,
per-reference metadata receipts, consumed-input ledger and aggregate result in
the private `artifacts/readout-diagnostics/v1/` namespace. No fitting or held-out
evaluation is authorized by this diagnostic.
