"""Exact fixed-token checkpoint construction from saved generated token IDs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .provenance import sha256_token_ids


class CheckpointStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    TRAJECTORY_TOO_SHORT = "TRAJECTORY_TOO_SHORT"
    AFTER_THINK_CLOSE = "AFTER_THINK_CLOSE"
    AFTER_TERMINAL_TOKEN = "AFTER_TERMINAL_TOKEN"


@dataclass(frozen=True)
class TokenCheckpoint:
    """One requested coordinate over an exact saved generated-token sequence."""

    token_position: int
    status: CheckpointStatus
    prefix_generated_token_ids: tuple[int, ...] | None
    prefix_sha256: str | None
    reason: str | None = None

    @property
    def available(self) -> bool:
        return self.status is CheckpointStatus.AVAILABLE


def build_fixed_checkpoints(
    generated_token_ids: Iterable[int],
    checkpoint_positions: Iterable[int],
    *,
    close_think_token_ids: Iterable[int],
    eos_token_ids: Iterable[int] = (),
) -> tuple[TokenCheckpoint, ...]:
    """Construct every requested fixed position without re-tokenizing text.

    The coordinate is one-based as a generated-token count: position 256 means
    the prefix contains the first 256 saved generated IDs.
    """

    generated = tuple(int(token_id) for token_id in generated_token_ids)
    close_think = tuple(int(token_id) for token_id in close_think_token_ids)
    if not close_think:
        raise ValueError("close_think_token_ids cannot be empty")
    eos_ids = frozenset(int(token_id) for token_id in eos_token_ids)

    close_start = find_subsequence(generated, close_think)
    checkpoints: list[TokenCheckpoint] = []
    for position in checkpoint_positions:
        if isinstance(position, bool) or not isinstance(position, int) or position <= 0:
            raise ValueError("checkpoint positions must be positive integers")
        if position > len(generated):
            checkpoints.append(
                TokenCheckpoint(
                    token_position=position,
                    status=CheckpointStatus.TRAJECTORY_TOO_SHORT,
                    prefix_generated_token_ids=None,
                    prefix_sha256=None,
                    reason=f"saved trajectory has {len(generated)} generated tokens",
                )
            )
            continue

        prefix = generated[:position]
        if close_start is not None and close_start < position:
            checkpoints.append(
                TokenCheckpoint(
                    token_position=position,
                    status=CheckpointStatus.AFTER_THINK_CLOSE,
                    prefix_generated_token_ids=None,
                    prefix_sha256=None,
                    reason="requested prefix reaches the original final-answer phase",
                )
            )
            continue
        if eos_ids and any(token_id in eos_ids for token_id in prefix):
            checkpoints.append(
                TokenCheckpoint(
                    token_position=position,
                    status=CheckpointStatus.AFTER_TERMINAL_TOKEN,
                    prefix_generated_token_ids=None,
                    prefix_sha256=None,
                    reason="requested prefix contains a terminal token",
                )
            )
            continue

        checkpoints.append(
            TokenCheckpoint(
                token_position=position,
                status=CheckpointStatus.AVAILABLE,
                prefix_generated_token_ids=prefix,
                prefix_sha256=sha256_token_ids(prefix),
            )
        )
    return tuple(checkpoints)


def find_subsequence(tokens: Iterable[int], target: Iterable[int]) -> int | None:
    """Return the first start index of an exact token subsequence, if present."""

    haystack = tuple(int(token_id) for token_id in tokens)
    needle = tuple(int(token_id) for token_id in target)
    if not needle:
        raise ValueError("target token sequence cannot be empty")
    last_start = len(haystack) - len(needle)
    for start in range(last_start + 1):
        if haystack[start : start + len(needle)] == needle:
            return start
    return None
