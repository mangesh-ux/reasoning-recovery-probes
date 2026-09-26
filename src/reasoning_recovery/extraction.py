"""Conservative answer extraction for the reviewed boxed-answer policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ExtractionStatus(str, Enum):
    EXTRACTED = "EXTRACTED"
    MISSING_BOXED_ANSWER = "MISSING_BOXED_ANSWER"
    MALFORMED_BOXED_ANSWER = "MALFORMED_BOXED_ANSWER"
    EMPTY_BOXED_ANSWER = "EMPTY_BOXED_ANSWER"
    ERROR = "ERROR"


@dataclass(frozen=True)
class AnswerExtraction:
    status: ExtractionStatus
    extracted_text: str | None
    start_offset: int | None
    end_offset: int | None
    reason: str | None = None

    @property
    def available(self) -> bool:
        return self.status is ExtractionStatus.EXTRACTED

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "extracted_text": self.extracted_text,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "reason": self.reason,
        }


def extract_last_balanced_boxed(text: str) -> AnswerExtraction:
    """Extract the last balanced boxed expression, retaining nested braces."""

    if not isinstance(text, str):
        raise TypeError("completion text must be a string")

    marker = r"\boxed"
    cursor = 0
    last: AnswerExtraction | None = None
    saw_marker = False
    while True:
        marker_start = text.find(marker, cursor)
        if marker_start < 0:
            break
        after_marker = marker_start + len(marker)
        if after_marker < len(text) and text[after_marker].isalpha():
            cursor = after_marker
            continue
        saw_marker = True
        open_brace = after_marker
        while open_brace < len(text) and text[open_brace].isspace():
            open_brace += 1
        if open_brace >= len(text) or text[open_brace] != "{":
            return AnswerExtraction(
                status=ExtractionStatus.MALFORMED_BOXED_ANSWER,
                extracted_text=None,
                start_offset=marker_start,
                end_offset=None,
                reason="boxed marker is not followed by an opening brace",
            )
        close_brace = _matching_unescaped_brace(text, open_brace)
        if close_brace is None:
            return AnswerExtraction(
                status=ExtractionStatus.MALFORMED_BOXED_ANSWER,
                extracted_text=None,
                start_offset=marker_start,
                end_offset=None,
                reason="boxed expression has no matching closing brace",
            )

        inner = text[open_brace + 1 : close_brace]
        if not inner.strip():
            last = AnswerExtraction(
                status=ExtractionStatus.EMPTY_BOXED_ANSWER,
                extracted_text=None,
                start_offset=marker_start,
                end_offset=close_brace + 1,
                reason="boxed expression is empty",
            )
        else:
            last = AnswerExtraction(
                status=ExtractionStatus.EXTRACTED,
                extracted_text=text[marker_start : close_brace + 1],
                start_offset=marker_start,
                end_offset=close_brace + 1,
            )
        cursor = close_brace + 1

    if last is not None:
        return last
    if saw_marker:
        return AnswerExtraction(
            status=ExtractionStatus.MALFORMED_BOXED_ANSWER,
            extracted_text=None,
            start_offset=None,
            end_offset=None,
            reason="no complete boxed expression was found",
        )
    return AnswerExtraction(
        status=ExtractionStatus.MISSING_BOXED_ANSWER,
        extracted_text=None,
        start_offset=None,
        end_offset=None,
        reason="completion contains no boxed expression",
    )


def _matching_unescaped_brace(text: str, open_brace: int) -> int | None:
    depth = 1
    for index in range(open_brace + 1, len(text)):
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
    count = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        count += 1
        cursor -= 1
    return count % 2 == 1
