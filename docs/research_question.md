# P0 research question and claim boundary

## Eventual question

The future, separately designed research question is:

> Given that a reasoning model is wrong at an intermediate checkpoint, do its
> hidden activations contain information about whether continued reasoning will
> later recover to a correct answer beyond observable checkpoint position and
> confidence?

P0 does not answer that question. It neither saves hidden activations nor fits
any predictive model.

## P0 feasibility question

P0 asks whether Qwen3-1.7B can reliably produce and preserve the data required
for a later study on the stated local hardware:

1. Can a model-backed stochastic reasoning trajectory be generated and saved?
2. Can fixed positions in its exact saved generated-token sequence be rebuilt
   without regenerating the original trajectory?
3. Can a deterministic answer be forced from each available saved prefix?
4. Can the forced checkpoint answer and the original trajectory's final answer
   be evaluated without hiding malformed or non-evaluable cases?
5. How often do the four observed checkpoint-to-original-final transitions
   occur, especially \`W_TO_C\` and \`W_TO_W\`?
6. What operational cost and failure evidence is observed?

## Transition semantics

For one problem and one sampled original trajectory, let \`checkpoint\` be the
correctness of an answer forced from an exact saved prefix. Let \`final\` be the
correctness of the final answer extracted from that same original saved
trajectory. The labels are:

\`\`\`text
checkpoint wrong + final correct -> W_TO_C
checkpoint wrong + final wrong   -> W_TO_W
checkpoint correct + final correct -> C_TO_C
checkpoint correct + final wrong -> C_TO_W
\`\`\`

The original final answer is never supplied by a new continuation sampled at
the checkpoint. A non-evaluable or error outcome has no transition label and
remains separately counted.

## Prohibited P0 claims

P0 cannot establish that activations predict recovery, that continued reasoning
is generally valuable, that a stopping rule is safe, that a checkpoint is
causal, or that any performance/efficiency method is superior. Its artifacts
are feasibility-only and must remain separate from any later confirmatory data.
