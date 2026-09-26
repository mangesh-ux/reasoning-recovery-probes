"""Construction of deterministic forced-answer inputs from saved prefixes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .checkpointing import TokenCheckpoint
from .provenance import sha256_token_ids


@dataclass(frozen=True)
class ForcedAnswerInput:
    """Exact token input supplied to the deterministic forced-answer request."""

    prompt_token_ids: tuple[int, ...]
    saved_prefix_token_ids: tuple[int, ...]
    close_think_cue_token_ids: tuple[int, ...]
    input_token_ids: tuple[int, ...]
    prompt_sha256: str
    saved_prefix_sha256: str
    cue_sha256: str
    input_sha256: str


def build_forced_answer_input(
    prompt_token_ids: Iterable[int],
    checkpoint: TokenCheckpoint,
    close_think_cue_token_ids: Iterable[int],
) -> ForcedAnswerInput:
    """Append a frozen cue to the exact saved prefix without resampling it."""

    if not checkpoint.available or checkpoint.prefix_generated_token_ids is None:
        raise ValueError("an available fixed checkpoint is required")

    prompt = tuple(int(token_id) for token_id in prompt_token_ids)
    prefix = checkpoint.prefix_generated_token_ids
    cue = tuple(int(token_id) for token_id in close_think_cue_token_ids)
    if not prompt:
        raise ValueError("prompt_token_ids cannot be empty")
    if not cue:
        raise ValueError("close_think_cue_token_ids cannot be empty")

    combined = prompt + prefix + cue
    return ForcedAnswerInput(
        prompt_token_ids=prompt,
        saved_prefix_token_ids=prefix,
        close_think_cue_token_ids=cue,
        input_token_ids=combined,
        prompt_sha256=sha256_token_ids(prompt),
        saved_prefix_sha256=sha256_token_ids(prefix),
        cue_sha256=sha256_token_ids(cue),
        input_sha256=sha256_token_ids(combined),
    )
