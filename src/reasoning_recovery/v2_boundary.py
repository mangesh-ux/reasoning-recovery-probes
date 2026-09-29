"""Exact V2 token-coordinate reasoning and readout-boundary primitives.

This module is deliberately independent of P0 receipts and text-derived
checkpointing.  V2 reasoning boundaries are constructed only from the saved
generated token IDs and the frozen close-thinking token-ID sequence.  The
readout boundary is separately reconstructed from the generated readout token
sequence so scalar score features never invent a character-level likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Sequence

from .provenance import sha256_token_ids


class BoundaryIntegrityError(ValueError):
    """Raised when saved IDs cannot satisfy the frozen V2 boundary contract."""


class ReasoningBoundaryStatus(str, Enum):
    """Terminal state of a base V2 reasoning trajectory."""

    CLOSE_MARKER_FOUND = "CLOSE_MARKER_FOUND"
    CAP_PREFIX = "CAP_PREFIX"
    EARLY_NO_CLOSE = "EARLY_NO_CLOSE"


class V2CheckpointStatus(str, Enum):
    """Availability is evidence; unavailable coordinates are never filled in."""

    AVAILABLE = "AVAILABLE"
    AFTER_THINK_CLOSE = "AFTER_THINK_CLOSE"
    TRAJECTORY_TOO_SHORT = "TRAJECTORY_TOO_SHORT"
    EARLY_NO_CLOSE = "EARLY_NO_CLOSE"


@dataclass(frozen=True)
class ReasoningBoundary:
    """The V2 terminal reasoning prefix derived directly from saved token IDs."""

    generated_token_ids: tuple[int, ...]
    close_think_token_ids: tuple[int, ...]
    termination_status: str
    status: ReasoningBoundaryStatus
    close_marker_start: int | None
    terminal_reasoning_prefix_token_ids: tuple[int, ...] | None
    terminal_prefix_sha256: str | None
    reason: str | None

    @property
    def has_terminal_reasoning_prefix(self) -> bool:
        """Whether a symmetric terminal readout can be constructed."""

        return self.terminal_reasoning_prefix_token_ids is not None

    @property
    def terminal_reasoning_token_count(self) -> int | None:
        prefix = self.terminal_reasoning_prefix_token_ids
        return len(prefix) if prefix is not None else None

    def safe_summary(self) -> dict[str, object]:
        """Return boundary metadata without raw generated token IDs."""

        return {
            "status": self.status.value,
            "termination_status": self.termination_status,
            "generated_token_count": len(self.generated_token_ids),
            "generated_token_ids_sha256": sha256_token_ids(self.generated_token_ids),
            "close_marker_start": self.close_marker_start,
            "terminal_reasoning_token_count": self.terminal_reasoning_token_count,
            "terminal_prefix_sha256": self.terminal_prefix_sha256,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class V2Checkpoint:
    """One exact raw generated-token checkpoint under the frozen V2 geometry."""

    token_position: int
    status: V2CheckpointStatus
    prefix_generated_token_ids: tuple[int, ...] | None
    prefix_sha256: str | None
    reason: str | None = None

    @property
    def available(self) -> bool:
        return self.status is V2CheckpointStatus.AVAILABLE

    def safe_summary(self) -> dict[str, object]:
        return {
            "token_position": self.token_position,
            "status": self.status.value,
            "prefix_token_count": (
                len(self.prefix_generated_token_ids)
                if self.prefix_generated_token_ids is not None
                else None
            ),
            "prefix_sha256": self.prefix_sha256,
            "reason": self.reason,
        }


def first_token_subsequence_start(
    token_ids: Sequence[int], target_token_ids: Sequence[int]
) -> int | None:
    """Return the first exact target-token start without decoding either side."""

    target = tuple(int(token_id) for token_id in target_token_ids)
    if not target:
        raise BoundaryIntegrityError("target token sequence cannot be empty")
    tokens = tuple(int(token_id) for token_id in token_ids)
    for start in range(len(tokens) - len(target) + 1):
        if tokens[start : start + len(target)] == target:
            return start
    return None


def construct_reasoning_boundary(
    generated_token_ids: Iterable[int],
    *,
    close_think_token_ids: Iterable[int],
    eos_token_ids: Iterable[int],
    max_new_tokens: int,
    termination_status: str,
) -> ReasoningBoundary:
    """Construct the frozen V2 terminal-prefix rule from exact emitted IDs.

    A first close-thinking marker yields the prefix immediately before it.
    Only a genuine max-new-token cap with no marker yields the all-generated
    cap prefix.  EOS or any other terminal event before the marker is an
    explicit early-no-close failure; it is never reclassified as a cap prefix.
    """

    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise BoundaryIntegrityError("max_new_tokens must be a positive integer")
    if max_new_tokens <= 0:
        raise BoundaryIntegrityError("max_new_tokens must be positive")
    if not isinstance(termination_status, str) or not termination_status:
        raise BoundaryIntegrityError("termination_status must be a non-empty string")

    generated = tuple(int(token_id) for token_id in generated_token_ids)
    marker = tuple(int(token_id) for token_id in close_think_token_ids)
    eos_ids = frozenset(int(token_id) for token_id in eos_token_ids)
    if not marker:
        raise BoundaryIntegrityError("close_think_token_ids cannot be empty")
    if eos_ids.intersection(marker):
        raise BoundaryIntegrityError("close-thinking marker cannot contain an EOS token")
    if len(generated) > max_new_tokens:
        raise BoundaryIntegrityError(
            "saved generated token count exceeds the frozen max_new_tokens"
        )

    close_start = first_token_subsequence_start(generated, marker)
    first_eos = next(
        (index for index, token_id in enumerate(generated) if token_id in eos_ids), None
    )

    if close_start is not None and (first_eos is None or first_eos >= close_start):
        prefix = generated[:close_start]
        return ReasoningBoundary(
            generated_token_ids=generated,
            close_think_token_ids=marker,
            termination_status=termination_status,
            status=ReasoningBoundaryStatus.CLOSE_MARKER_FOUND,
            close_marker_start=close_start,
            terminal_reasoning_prefix_token_ids=prefix,
            terminal_prefix_sha256=sha256_token_ids(prefix),
            reason=None,
        )

    if close_start is None and termination_status == "MAX_NEW_TOKENS" and len(generated) == max_new_tokens:
        return ReasoningBoundary(
            generated_token_ids=generated,
            close_think_token_ids=marker,
            termination_status=termination_status,
            status=ReasoningBoundaryStatus.CAP_PREFIX,
            close_marker_start=None,
            terminal_reasoning_prefix_token_ids=generated,
            terminal_prefix_sha256=sha256_token_ids(generated),
            reason="no_close_think_marker_at_terminal_cap",
        )

    if close_start is not None and first_eos is not None and first_eos < close_start:
        reason = "eos_precedes_first_close_think_marker"
    elif close_start is None and termination_status == "MAX_NEW_TOKENS":
        reason = "max_new_tokens_status_has_incomplete_saved_sequence"
    elif close_start is None and first_eos is not None:
        reason = "eos_before_close_think_marker"
    else:
        reason = "terminal_event_before_close_think_marker"
    return ReasoningBoundary(
        generated_token_ids=generated,
        close_think_token_ids=marker,
        termination_status=termination_status,
        status=ReasoningBoundaryStatus.EARLY_NO_CLOSE,
        close_marker_start=close_start,
        terminal_reasoning_prefix_token_ids=None,
        terminal_prefix_sha256=None,
        reason=reason,
    )


def construct_fixed_checkpoints(
    boundary: ReasoningBoundary, checkpoint_token_positions: Iterable[int]
) -> tuple[V2Checkpoint, ...]:
    """Build exact V2 checkpoints from the terminal prefix, never from text."""

    positions = tuple(checkpoint_token_positions)
    if not positions:
        raise BoundaryIntegrityError("at least one checkpoint position is required")
    if any(
        isinstance(position, bool) or not isinstance(position, int) or position <= 0
        for position in positions
    ):
        raise BoundaryIntegrityError("checkpoint positions must be positive integers")
    if positions != tuple(sorted(positions)) or len(set(positions)) != len(positions):
        raise BoundaryIntegrityError("checkpoint positions must be strictly increasing")

    prefix = boundary.terminal_reasoning_prefix_token_ids
    checkpoints: list[V2Checkpoint] = []
    for position in positions:
        if prefix is None:
            checkpoints.append(
                V2Checkpoint(
                    token_position=position,
                    status=V2CheckpointStatus.EARLY_NO_CLOSE,
                    prefix_generated_token_ids=None,
                    prefix_sha256=None,
                    reason=boundary.reason,
                )
            )
            continue
        if position <= len(prefix):
            exact_prefix = prefix[:position]
            checkpoints.append(
                V2Checkpoint(
                    token_position=position,
                    status=V2CheckpointStatus.AVAILABLE,
                    prefix_generated_token_ids=exact_prefix,
                    prefix_sha256=sha256_token_ids(exact_prefix),
                    reason=None,
                )
            )
            continue
        unavailable_status = (
            V2CheckpointStatus.AFTER_THINK_CLOSE
            if boundary.status is ReasoningBoundaryStatus.CLOSE_MARKER_FOUND
            else V2CheckpointStatus.TRAJECTORY_TOO_SHORT
        )
        checkpoints.append(
            V2Checkpoint(
                token_position=position,
                status=unavailable_status,
                prefix_generated_token_ids=None,
                prefix_sha256=None,
                reason=(
                    "requested coordinate reaches or follows the natural close-thinking marker"
                    if unavailable_status is V2CheckpointStatus.AFTER_THINK_CLOSE
                    else "terminal cap prefix is shorter than requested coordinate"
                ),
            )
        )
    return tuple(checkpoints)


class ReadoutBoxStatus(str, Enum):
    """Status of the frozen last-balanced-boxed extraction rule."""

    EXTRACTED = "EXTRACTED"
    MISSING_BOXED_ANSWER = "MISSING_BOXED_ANSWER"
    MALFORMED_BOXED_ANSWER = "MALFORMED_BOXED_ANSWER"
    EMPTY_BOXED_ANSWER = "EMPTY_BOXED_ANSWER"


@dataclass(frozen=True)
class ReadoutBoxBoundary:
    """Character boundary for the selected last balanced boxed answer."""

    raw_text: str
    status: ReadoutBoxStatus
    marker_char_index: int | None
    opening_brace_char_index: int | None
    closing_brace_char_index: int | None
    reason: str | None = None

    @property
    def completed(self) -> bool:
        return self.status in {ReadoutBoxStatus.EXTRACTED, ReadoutBoxStatus.EMPTY_BOXED_ANSWER}

    @property
    def extracted_text(self) -> str | None:
        if not self.completed or self.marker_char_index is None or self.closing_brace_char_index is None:
            return None
        return self.raw_text[self.marker_char_index : self.closing_brace_char_index + 1]

    @property
    def candidate_text(self) -> str | None:
        if (
            not self.completed
            or self.opening_brace_char_index is None
            or self.closing_brace_char_index is None
        ):
            return None
        return self.raw_text[self.opening_brace_char_index + 1 : self.closing_brace_char_index]


def find_last_balanced_boxed_boundary(text: str) -> ReadoutBoxBoundary:
    """Apply V2's last-balanced-boxed-only parser without score assumptions."""

    if not isinstance(text, str):
        raise TypeError("readout text must be a string")

    marker = r"\boxed"
    cursor = 0
    last_balanced: ReadoutBoxBoundary | None = None
    last_malformed: ReadoutBoxBoundary | None = None
    saw_marker = False
    while True:
        marker_start = text.find(marker, cursor)
        if marker_start < 0:
            break
        after_marker = marker_start + len(marker)
        # Do not parse a longer alphabetic command such as \boxedness.
        if after_marker < len(text) and text[after_marker].isalpha():
            cursor = after_marker
            continue
        saw_marker = True
        opening = after_marker
        while opening < len(text) and text[opening].isspace():
            opening += 1
        if opening >= len(text) or text[opening] != "{":
            last_malformed = ReadoutBoxBoundary(
                raw_text=text,
                status=ReadoutBoxStatus.MALFORMED_BOXED_ANSWER,
                marker_char_index=marker_start,
                opening_brace_char_index=None,
                closing_brace_char_index=None,
                reason="boxed marker is not followed by an opening brace",
            )
            cursor = after_marker
            continue
        closing = _matching_unescaped_brace(text, opening)
        if closing is None:
            last_malformed = ReadoutBoxBoundary(
                raw_text=text,
                status=ReadoutBoxStatus.MALFORMED_BOXED_ANSWER,
                marker_char_index=marker_start,
                opening_brace_char_index=opening,
                closing_brace_char_index=None,
                reason="boxed expression has no matching closing brace",
            )
            # Continue after the marker so a later complete box remains eligible.
            cursor = after_marker
            continue
        candidate = text[opening + 1 : closing]
        last_balanced = ReadoutBoxBoundary(
            raw_text=text,
            status=(
                ReadoutBoxStatus.EXTRACTED
                if candidate.strip()
                else ReadoutBoxStatus.EMPTY_BOXED_ANSWER
            ),
            marker_char_index=marker_start,
            opening_brace_char_index=opening,
            closing_brace_char_index=closing,
            reason="boxed expression is empty" if not candidate.strip() else None,
        )
        cursor = after_marker

    if last_balanced is not None:
        return last_balanced
    if last_malformed is not None:
        return last_malformed
    return ReadoutBoxBoundary(
        raw_text=text,
        status=(
            ReadoutBoxStatus.MALFORMED_BOXED_ANSWER
            if saw_marker
            else ReadoutBoxStatus.MISSING_BOXED_ANSWER
        ),
        marker_char_index=None,
        opening_brace_char_index=None,
        closing_brace_char_index=None,
        reason=(
            "no complete boxed expression was found"
            if saw_marker
            else "completion contains no boxed expression"
        ),
    )


@dataclass(frozen=True)
class TokenTextReconstruction:
    """Exact prefix-decoded character spans for a generated token sequence."""

    token_ids: tuple[int, ...]
    decoded_text: str | None
    token_pieces: tuple[str, ...] | None
    token_char_starts: tuple[int, ...] | None
    token_char_ends: tuple[int, ...] | None
    reason: str | None

    @property
    def available(self) -> bool:
        return (
            self.decoded_text is not None
            and self.token_pieces is not None
            and self.token_char_starts is not None
            and self.token_char_ends is not None
        )


def reconstruct_token_text(
    token_ids: Iterable[int], *, decode: Callable[[Sequence[int]], str]
) -> TokenTextReconstruction:
    """Map decoded characters to exact token positions by prefix reconstruction.

    Decoding token IDs one at a time is not a valid universal tokenizer
    alignment strategy.  Instead, every prefix is decoded with the same
    no-cleanup settings as the whole sequence.  If that decoder is not prefix
    monotonic, the boundary is explicitly unavailable rather than guessed.
    """

    ids = tuple(int(token_id) for token_id in token_ids)
    try:
        full_text = decode(ids)
    except Exception as error:
        return TokenTextReconstruction(
            token_ids=ids,
            decoded_text=None,
            token_pieces=None,
            token_char_starts=None,
            token_char_ends=None,
            reason=f"full_token_decode_failed:{type(error).__name__}",
        )
    if not isinstance(full_text, str):
        return TokenTextReconstruction(
            token_ids=ids,
            decoded_text=None,
            token_pieces=None,
            token_char_starts=None,
            token_char_ends=None,
            reason="decoder_did_not_return_text",
        )

    pieces: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    previous = ""
    for end in range(1, len(ids) + 1):
        try:
            decoded_prefix = decode(ids[:end])
        except Exception as error:
            return TokenTextReconstruction(
                token_ids=ids,
                decoded_text=None,
                token_pieces=None,
                token_char_starts=None,
                token_char_ends=None,
                reason=f"prefix_token_decode_failed:{type(error).__name__}",
            )
        if not isinstance(decoded_prefix, str):
            return TokenTextReconstruction(
                token_ids=ids,
                decoded_text=None,
                token_pieces=None,
                token_char_starts=None,
                token_char_ends=None,
                reason="prefix_decoder_did_not_return_text",
            )
        if not decoded_prefix.startswith(previous):
            return TokenTextReconstruction(
                token_ids=ids,
                decoded_text=None,
                token_pieces=None,
                token_char_starts=None,
                token_char_ends=None,
                reason="decoder_prefixes_are_not_monotonic",
            )
        starts.append(len(previous))
        pieces.append(decoded_prefix[len(previous) :])
        ends.append(len(decoded_prefix))
        previous = decoded_prefix

    if previous != full_text or "".join(pieces) != full_text:
        return TokenTextReconstruction(
            token_ids=ids,
            decoded_text=None,
            token_pieces=None,
            token_char_starts=None,
            token_char_ends=None,
            reason="prefix_decoding_does_not_reconstruct_full_text",
        )
    return TokenTextReconstruction(
        token_ids=ids,
        decoded_text=full_text,
        token_pieces=tuple(pieces),
        token_char_starts=tuple(starts),
        token_char_ends=tuple(ends),
        reason=None,
    )


@dataclass(frozen=True)
class CandidateTokenRegion:
    """Token region strictly between outer brace-containing token positions."""

    opening_token_index: int | None
    closing_token_index: int | None
    included_token_indices: tuple[int, ...]
    characters_after_close_in_closing_token: str | None
    reason: str | None

    @property
    def available(self) -> bool:
        return self.opening_token_index is not None and self.closing_token_index is not None


def candidate_token_region(
    reconstruction: TokenTextReconstruction, boundary: ReadoutBoxBoundary
) -> CandidateTokenRegion:
    """Return only whole-token candidate indices; never split a token score."""

    if not reconstruction.available:
        return CandidateTokenRegion(
            opening_token_index=None,
            closing_token_index=None,
            included_token_indices=(),
            characters_after_close_in_closing_token=None,
            reason=reconstruction.reason,
        )
    if not boundary.completed:
        return CandidateTokenRegion(
            opening_token_index=None,
            closing_token_index=None,
            included_token_indices=(),
            characters_after_close_in_closing_token=None,
            reason=boundary.reason or boundary.status.value,
        )
    assert reconstruction.decoded_text is not None
    assert reconstruction.token_char_starts is not None
    assert reconstruction.token_char_ends is not None
    assert boundary.opening_brace_char_index is not None
    assert boundary.closing_brace_char_index is not None
    if reconstruction.decoded_text != boundary.raw_text:
        return CandidateTokenRegion(
            opening_token_index=None,
            closing_token_index=None,
            included_token_indices=(),
            characters_after_close_in_closing_token=None,
            reason="reconstructed_text_does_not_match_box_boundary_text",
        )
    opening = _token_index_for_character(
        reconstruction.token_char_starts,
        reconstruction.token_char_ends,
        boundary.opening_brace_char_index,
    )
    closing = _token_index_for_character(
        reconstruction.token_char_starts,
        reconstruction.token_char_ends,
        boundary.closing_brace_char_index,
    )
    if opening is None or closing is None:
        return CandidateTokenRegion(
            opening_token_index=None,
            closing_token_index=None,
            included_token_indices=(),
            characters_after_close_in_closing_token=None,
            reason="box_brace_has_no_exact_token_character_mapping",
        )
    closing_end = reconstruction.token_char_ends[closing]
    return CandidateTokenRegion(
        opening_token_index=opening,
        closing_token_index=closing,
        included_token_indices=tuple(range(opening + 1, closing)),
        characters_after_close_in_closing_token=reconstruction.decoded_text[
            boundary.closing_brace_char_index + 1 : closing_end
        ],
        reason=None,
    )


def normalise_extracted_answer_for_agreement(text: str | None) -> str | None:
    """Apply the narrow, non-semantic text normalization used for agreement.

    This helper intentionally performs no algebra, evaluation, or access to a
    reference answer.  It only trims and collapses Unicode whitespace so a
    readout's immediately preceding available checkpoint can be compared as an
    observable feature without turning the evaluator into a feature source.
    """

    if text is None:
        return None
    if not isinstance(text, str):
        raise TypeError("extracted answer must be text or None")
    normalized = " ".join(text.split())
    return normalized or None


def _matching_unescaped_brace(text: str, opening_brace_index: int) -> int | None:
    depth = 1
    for index in range(opening_brace_index + 1, len(text)):
        character = text[index]
        if character not in "{}" or _is_escaped(text, index):
            continue
        if character == "{":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return index
    return None


def _is_escaped(text: str, index: int) -> bool:
    slash_count = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        slash_count += 1
        cursor -= 1
    return slash_count % 2 == 1


def _token_index_for_character(
    starts: Sequence[int], ends: Sequence[int], character_index: int
) -> int | None:
    for token_index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        if start <= character_index < end:
            return token_index
    return None
