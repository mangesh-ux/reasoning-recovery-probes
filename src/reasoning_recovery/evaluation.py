"""Versioned mathematical answer evaluation with explicit non-evaluable states."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Protocol, Sequence

from .extraction import AnswerExtraction


class EvaluationStatus(str, Enum):
    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    NON_EVALUABLE = "NON_EVALUABLE"
    ERROR = "ERROR"


class TransitionLabel(str, Enum):
    W_TO_C = "W_TO_C"
    W_TO_W = "W_TO_W"
    C_TO_C = "C_TO_C"
    C_TO_W = "C_TO_W"


class EvaluationBackend(Protocol):
    name: str
    version: str | None

    def parse(self, text: str) -> Sequence[object]:
        """Parse one math answer into zero or more candidate expressions."""

    def verify(
        self, reference: Sequence[object], candidate: Sequence[object]
    ) -> bool:
        """Return exact Boolean equivalence for parsed expressions."""


@dataclass(frozen=True)
class EvaluationResult:
    status: EvaluationStatus
    correct: bool | None
    backend_name: str | None
    backend_version: str | None
    reason: str | None = None
    error_type: str | None = None
    parsed_reference_count: int | None = None
    parsed_candidate_count: int | None = None

    def __post_init__(self) -> None:
        if self.status in {EvaluationStatus.CORRECT, EvaluationStatus.INCORRECT}:
            if type(self.correct) is not bool:
                raise ValueError("evaluated results require Boolean correctness")
        elif self.correct is not None:
            raise ValueError("non-evaluable/error results cannot carry correctness")

    @property
    def evaluated(self) -> bool:
        return self.status in {EvaluationStatus.CORRECT, EvaluationStatus.INCORRECT}

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "correct": self.correct,
            "backend_name": self.backend_name,
            "backend_version": self.backend_version,
            "reason": self.reason,
            "error_type": self.error_type,
            "parsed_reference_count": self.parsed_reference_count,
            "parsed_candidate_count": self.parsed_candidate_count,
        }


@dataclass(frozen=True)
class MathVerifyBackend:
    """Small adapter that makes the math-verify call surface explicit."""

    parse_function: object
    verify_function: object
    version: str | None
    name: str = "math-verify"

    def parse(self, text: str) -> Sequence[object]:
        return self.parse_function(  # type: ignore[operator]
            text,
            parsing_timeout=None,
            raise_on_error=True,
        )

    def verify(
        self, reference: Sequence[object], candidate: Sequence[object]
    ) -> bool:
        return self.verify_function(  # type: ignore[operator]
            reference,
            candidate,
            timeout_seconds=None,
            raise_on_error=True,
        )


def default_backend() -> EvaluationBackend | None:
    """Load math-verify lazily so offline tests do not need the package."""

    try:
        from math_verify import parse, verify
    except Exception:
        return None
    try:
        backend_version = package_version("math-verify")
    except PackageNotFoundError:
        backend_version = None
    return MathVerifyBackend(
        parse_function=parse,
        verify_function=verify,
        version=backend_version,
    )


def evaluate_extraction(
    reference_answer: str,
    extraction: AnswerExtraction,
    *,
    backend: EvaluationBackend | None = None,
) -> EvaluationResult:
    """Evaluate a structured extraction without guessing a missing answer."""

    if not extraction.available or extraction.extracted_text is None:
        return EvaluationResult(
            status=EvaluationStatus.NON_EVALUABLE,
            correct=None,
            backend_name=_backend_name(backend),
            backend_version=_backend_version(backend),
            reason=f"answer extraction status: {extraction.status.value}",
        )
    return evaluate_answer(reference_answer, extraction.extracted_text, backend=backend)


def evaluate_answer(
    reference_answer: str | None,
    candidate_answer: str | None,
    *,
    backend: EvaluationBackend | None = None,
) -> EvaluationResult:
    """Evaluate exact extracted answers while preserving every failure class."""

    selected = default_backend() if backend is None else backend
    backend_name = _backend_name(selected)
    backend_version = _backend_version(selected)

    if not isinstance(reference_answer, str) or not reference_answer.strip():
        return EvaluationResult(
            status=EvaluationStatus.NON_EVALUABLE,
            correct=None,
            backend_name=backend_name,
            backend_version=backend_version,
            reason="missing or empty reference answer",
        )
    if not isinstance(candidate_answer, str) or not candidate_answer.strip():
        return EvaluationResult(
            status=EvaluationStatus.NON_EVALUABLE,
            correct=None,
            backend_name=backend_name,
            backend_version=backend_version,
            reason="missing or empty candidate answer",
        )
    if selected is None:
        return EvaluationResult(
            status=EvaluationStatus.NON_EVALUABLE,
            correct=None,
            backend_name="math-verify",
            backend_version=None,
            reason="math-verify is unavailable",
        )

    try:
        parsed_reference = selected.parse(reference_answer)
    except Exception as error:
        return EvaluationResult(
            status=EvaluationStatus.ERROR,
            correct=None,
            backend_name=backend_name,
            backend_version=backend_version,
            reason="reference parsing failed",
            error_type=type(error).__name__,
        )
    reference_count = _parsed_count(parsed_reference)
    if reference_count is None:
        return EvaluationResult(
            status=EvaluationStatus.ERROR,
            correct=None,
            backend_name=backend_name,
            backend_version=backend_version,
            reason="reference parser returned an unsupported value",
        )
    if reference_count == 0:
        return EvaluationResult(
            status=EvaluationStatus.NON_EVALUABLE,
            correct=None,
            backend_name=backend_name,
            backend_version=backend_version,
            reason="reference answer did not parse",
            parsed_reference_count=0,
        )

    try:
        parsed_candidate = selected.parse(candidate_answer)
    except Exception as error:
        return EvaluationResult(
            status=EvaluationStatus.ERROR,
            correct=None,
            backend_name=backend_name,
            backend_version=backend_version,
            reason="candidate parsing failed",
            error_type=type(error).__name__,
            parsed_reference_count=reference_count,
        )
    candidate_count = _parsed_count(parsed_candidate)
    if candidate_count is None:
        return EvaluationResult(
            status=EvaluationStatus.ERROR,
            correct=None,
            backend_name=backend_name,
            backend_version=backend_version,
            reason="candidate parser returned an unsupported value",
            parsed_reference_count=reference_count,
        )
    if candidate_count == 0:
        return EvaluationResult(
            status=EvaluationStatus.NON_EVALUABLE,
            correct=None,
            backend_name=backend_name,
            backend_version=backend_version,
            reason="candidate answer did not parse",
            parsed_reference_count=reference_count,
            parsed_candidate_count=0,
        )

    try:
        equivalent = selected.verify(parsed_reference, parsed_candidate)
    except Exception as error:
        return EvaluationResult(
            status=EvaluationStatus.ERROR,
            correct=None,
            backend_name=backend_name,
            backend_version=backend_version,
            reason="symbolic verification failed",
            error_type=type(error).__name__,
            parsed_reference_count=reference_count,
            parsed_candidate_count=candidate_count,
        )
    if type(equivalent) is not bool:
        return EvaluationResult(
            status=EvaluationStatus.ERROR,
            correct=None,
            backend_name=backend_name,
            backend_version=backend_version,
            reason="symbolic verifier returned a non-Boolean value",
            parsed_reference_count=reference_count,
            parsed_candidate_count=candidate_count,
        )
    return EvaluationResult(
        status=EvaluationStatus.CORRECT if equivalent else EvaluationStatus.INCORRECT,
        correct=equivalent,
        backend_name=backend_name,
        backend_version=backend_version,
        parsed_reference_count=reference_count,
        parsed_candidate_count=candidate_count,
    )


def classify_transition(
    checkpoint: EvaluationResult, final: EvaluationResult
) -> TransitionLabel | None:
    """Label an observed checkpoint-to-original-final transition when evaluable."""

    if not checkpoint.evaluated or not final.evaluated:
        return None
    assert checkpoint.correct is not None
    assert final.correct is not None
    mapping = {
        (False, True): TransitionLabel.W_TO_C,
        (False, False): TransitionLabel.W_TO_W,
        (True, True): TransitionLabel.C_TO_C,
        (True, False): TransitionLabel.C_TO_W,
    }
    return mapping[(checkpoint.correct, final.correct)]


def _backend_name(backend: EvaluationBackend | None) -> str | None:
    return getattr(backend, "name", None) if backend is not None else None


def _backend_version(backend: EvaluationBackend | None) -> str | None:
    return getattr(backend, "version", None) if backend is not None else None


def _parsed_count(value: object) -> int | None:
    try:
        count = len(value)  # type: ignore[arg-type]
    except TypeError:
        return None
    return count if isinstance(count, int) and count >= 0 else None
