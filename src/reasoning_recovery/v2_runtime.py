"""Pinned BF16 runtime for Protocol V2.

This is intentionally separate from the P0 runtime.  P0 avoids scores and
activations by design; V2 needs compact deterministic-readout score features
and hook-based residual extraction while preserving an exact provenance
contract.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import gc
from importlib.metadata import PackageNotFoundError, version as package_version
import math
from pathlib import Path
from typing import Any, Iterator, Sequence
import time

from .model_runtime import (
    _cached_asset_manifest,
    _context_limit,
    _generation_defaults,
    _normalise_token_ids,
    _resolved_eos_token_ids,
    _resolve_hub_revision,
)
from .provenance import base_runtime_provenance, sha256_text, sha256_token_ids
from .v2_activations import (
    ActivationExtraction,
    ActivationExtractionError,
    CudaMemorySnapshot,
    V2ActivationInput,
    capture_cuda_memory,
    extract_post_block_last_token_activations,
)
from .v2_boundary import (
    CandidateTokenRegion,
    ReadoutBoxBoundary,
    candidate_token_region,
    find_last_balanced_boxed_boundary,
    reconstruct_token_text,
)
from .v2_config import V2Config


class V2RuntimeQualificationError(RuntimeError):
    """Raised before a V2 request when the pinned runtime cannot be proven."""


class V2GenerationRuntimeError(RuntimeError):
    """A typed terminal outcome for one V2 generation request."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str,
        cuda_oom: bool = False,
        elapsed_seconds: float | None = None,
        memory_before: CudaMemorySnapshot | None = None,
        memory_after: CudaMemorySnapshot | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.cuda_oom = cuda_oom
        self.elapsed_seconds = elapsed_seconds
        self.memory_before = memory_before
        self.memory_after = memory_after


@dataclass(frozen=True)
class V2TokenizedPrompt:
    """Exact template tokenization retained by a V2 raw receipt."""

    user_prompt: str
    chat_prompt_text: str
    token_ids: tuple[int, ...]
    token_ids_sha256: str
    problem_token_count: int

    def safe_summary(self) -> dict[str, object]:
        return {
            "prompt_token_count": len(self.token_ids),
            "prompt_token_ids_sha256": self.token_ids_sha256,
            "problem_token_count": self.problem_token_count,
        }


@dataclass(frozen=True)
class V2DecodingParameters:
    """Exact request-level sampling or deterministic-readout settings."""

    request_kind: str
    max_new_tokens: int
    do_sample: bool
    temperature: float | None
    top_p: float | None
    top_k: int | None
    min_p: float | None
    seed: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "request_kind": self.request_kind,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class V2GeneratedSequence:
    """Generated IDs and aggregate timing/memory, without a logit tensor."""

    input_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    generated_text: str
    elapsed_seconds: float
    termination_status: str
    memory_before: CudaMemorySnapshot
    memory_after: CudaMemorySnapshot
    decoding: V2DecodingParameters

    def to_dict(self, *, include_input_token_ids: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "input_token_count": len(self.input_token_ids),
            "input_token_ids_sha256": sha256_token_ids(self.input_token_ids),
            "generated_token_ids": list(self.generated_token_ids),
            "generated_token_ids_sha256": sha256_token_ids(self.generated_token_ids),
            "generated_token_count": len(self.generated_token_ids),
            "generated_text": self.generated_text,
            "elapsed_seconds": self.elapsed_seconds,
            "termination_status": self.termination_status,
            "memory_before": self.memory_before.to_dict(),
            "memory_after": self.memory_after.to_dict(),
            "decoding": self.decoding.to_dict(),
        }
        if include_input_token_ids:
            payload["input_token_ids"] = list(self.input_token_ids)
        return payload


@dataclass(frozen=True)
class V2ReadoutScores:
    """Compact scalar score evidence from a deterministic V2 readout."""

    selected_token_logprobs: tuple[float, ...]
    first_readout_token_entropy_nats: float | None
    first_readout_token_probability_margin: float | None
    box_boundary: ReadoutBoxBoundary
    candidate_region: CandidateTokenRegion
    mean_answer_token_logprob: float | None
    min_answer_token_logprob: float | None
    answer_token_count: int | None
    reconstruction_reason: str | None

    def safe_summary(self) -> dict[str, object]:
        return {
            "selected_token_logprobs": list(self.selected_token_logprobs),
            "first_readout_token_entropy_nats": self.first_readout_token_entropy_nats,
            "first_readout_token_probability_margin": (
                self.first_readout_token_probability_margin
            ),
            "box_status": self.box_boundary.status.value,
            "box_reason": self.box_boundary.reason,
            "candidate_region": {
                "opening_token_index": self.candidate_region.opening_token_index,
                "closing_token_index": self.candidate_region.closing_token_index,
                "included_token_indices": list(self.candidate_region.included_token_indices),
                "characters_after_close_in_closing_token": (
                    self.candidate_region.characters_after_close_in_closing_token
                ),
                "reason": self.candidate_region.reason,
            },
            "mean_answer_token_logprob": self.mean_answer_token_logprob,
            "min_answer_token_logprob": self.min_answer_token_logprob,
            "answer_token_count": self.answer_token_count,
            "reconstruction_reason": self.reconstruction_reason,
        }


@dataclass(frozen=True)
class V2ReadoutSequence:
    """A deterministic readout plus scalar score summaries."""

    generation: V2GeneratedSequence
    scores: V2ReadoutScores

    def to_dict(self, *, include_input_token_ids: bool = True) -> dict[str, object]:
        payload = self.generation.to_dict(include_input_token_ids=include_input_token_ids)
        payload["observable_scores"] = self.scores.safe_summary()
        return payload


class V2Runtime:
    """One pinned Qwen runtime for V2, never loaded on module import."""

    def __init__(
        self,
        *,
        torch: Any,
        tokenizer: Any,
        model: Any,
        config: V2Config,
        device: Any,
        context_limit_tokens: int | None,
        eos_token_ids: tuple[int, ...],
        provenance: dict[str, object],
    ) -> None:
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._config = config
        self._device = device
        self._context_limit_tokens = context_limit_tokens
        self._eos_token_ids = eos_token_ids
        self._provenance = provenance

    @classmethod
    def load(cls, config: V2Config) -> "V2Runtime":
        """Load only the pinned V2 model/runtime after source gates pass."""

        try:
            import torch
            from huggingface_hub import HfApi
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as error:
            raise V2RuntimeQualificationError(
                "V2 requires torch, transformers, and huggingface_hub"
            ) from error

        _validate_runtime_versions(config)
        if config.runtime.require_cuda and not torch.cuda.is_available():
            raise V2RuntimeQualificationError("V2 requires CUDA; CPU fallback is forbidden")
        device = torch.device(config.model.device)
        if device.type != "cuda":
            raise V2RuntimeQualificationError("V2 config must use a CUDA device")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise V2RuntimeQualificationError(
                f"configured CUDA device is unavailable: {config.model.device}"
            )
        if config.runtime.require_bfloat16 and not torch.cuda.is_bf16_supported():
            raise V2RuntimeQualificationError(
                "V2 requires BF16 support on the configured CUDA device"
            )

        model_revision = _resolve_hub_revision(
            HfApi(), config.model.model_id, config.model.revision, label="V2 model"
        )
        tokenizer_revision = _resolve_hub_revision(
            HfApi(),
            config.model.model_id,
            config.model.tokenizer_revision,
            label="V2 tokenizer",
        )
        if model_revision != config.model.revision:
            raise V2RuntimeQualificationError(
                "resolved model revision differs from frozen V2 model revision"
            )
        if tokenizer_revision != config.model.tokenizer_revision:
            raise V2RuntimeQualificationError(
                "resolved tokenizer revision differs from frozen V2 tokenizer revision"
            )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config.model.model_id,
                revision=tokenizer_revision,
                trust_remote_code=False,
            )
            pad_inherited = False
            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token is None:
                    raise V2RuntimeQualificationError(
                        "V2 tokenizer has neither a pad token nor EOS token"
                    )
                tokenizer.pad_token = tokenizer.eos_token
                pad_inherited = True
            model = AutoModelForCausalLM.from_pretrained(
                config.model.model_id,
                revision=model_revision,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=False,
            )
            model.to(device)
            model.eval()
        except V2RuntimeQualificationError:
            raise
        except Exception as error:
            _safe_empty_cache(torch)
            raise V2RuntimeQualificationError(
                f"V2 model/tokenizer load failed: {type(error).__name__}: {str(error)[:300]}"
            ) from error

        try:
            _validate_model_geometry(model, config)
            properties = torch.cuda.get_device_properties(device)
            context_limit = _context_limit(model, tokenizer)
            eos_ids = _resolved_eos_token_ids(model, tokenizer)
            asset_manifest = _cached_asset_manifest(
                config.model.model_id, (model_revision, tokenizer_revision)
            )
            if asset_manifest.get("status") != "COMPLETE":
                raise V2RuntimeQualificationError(
                    "could not construct a complete cached asset manifest for V2"
                )
            chat_template = getattr(tokenizer, "chat_template", None)
            model_commit = getattr(getattr(model, "config", None), "_commit_hash", None)
            tokenizer_commit = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
            provenance = {
                **base_runtime_provenance(),
                "v2_package_versions": _runtime_package_versions(),
                "model": {
                    "id": config.model.model_id,
                    "requested_revision": config.model.revision,
                    "resolved_revision": model_revision,
                    "loaded_config_commit": model_commit,
                    "loaded_config_commit_matches_resolved": (
                        None if model_commit is None else model_commit == model_revision
                    ),
                    "dtype": config.model.dtype,
                    "device": str(device),
                    "model_type": getattr(getattr(model, "config", None), "model_type", None),
                    "num_hidden_layers": getattr(
                        getattr(model, "config", None), "num_hidden_layers", None
                    ),
                    "hidden_size": getattr(
                        getattr(model, "config", None), "hidden_size", None
                    ),
                    "context_limit_tokens": context_limit,
                },
                "tokenizer": {
                    "requested_revision": config.model.tokenizer_revision,
                    "resolved_revision": tokenizer_revision,
                    "loaded_init_commit": tokenizer_commit,
                    "loaded_init_commit_matches_resolved": (
                        None
                        if tokenizer_commit is None
                        else tokenizer_commit == tokenizer_revision
                    ),
                    "name_or_path": getattr(tokenizer, "name_or_path", None),
                    "tokenizer_eos_token_ids": list(
                        _normalise_token_ids(tokenizer.eos_token_id)
                    ),
                    "pad_token_id": tokenizer.pad_token_id,
                    "pad_inherited_from_eos": pad_inherited,
                    "chat_template_sha256": sha256_text(str(chat_template)),
                },
                "resolved_generation_defaults": _generation_defaults(model),
                "resolved_generation_eos_token_ids": list(eos_ids),
                "model_asset_manifest": asset_manifest,
                "gpu": {
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "capability": [int(properties.major), int(properties.minor)],
                    "torch_cuda_build": getattr(getattr(torch, "version", None), "cuda", None),
                },
                "memory_after_load": capture_cuda_memory(torch, device).to_dict(),
            }
        except Exception:
            del model, tokenizer
            gc.collect()
            _safe_empty_cache(torch)
            raise
        return cls(
            torch=torch,
            tokenizer=tokenizer,
            model=model,
            config=config,
            device=device,
            context_limit_tokens=context_limit,
            eos_token_ids=eos_ids,
            provenance=provenance,
        )

    @property
    def provenance(self) -> dict[str, object]:
        return self._provenance

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        return self._eos_token_ids

    @property
    def device(self) -> Any:
        return self._device

    @property
    def torch(self) -> Any:
        return self._torch

    def tokenize_prompt(self, problem: str) -> V2TokenizedPrompt:
        """Apply the pinned Qwen thinking template to one frozen problem."""

        user_prompt = self._config.prompt.render(problem)
        messages = [{"role": "user", "content": user_prompt}]
        try:
            encoded = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=self._config.model.enable_thinking,
            )
            ids = tuple(int(value) for value in encoded["input_ids"][0].detach().cpu().tolist())
            prompt_text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self._config.model.enable_thinking,
            )
            problem_token_count = len(
                tuple(
                    int(value)
                    for value in self._tokenizer.encode(
                        problem, add_special_tokens=False
                    )
                )
            )
        except Exception as error:
            raise V2RuntimeQualificationError(
                f"V2 thinking chat template failed: {type(error).__name__}: {str(error)[:300]}"
            ) from error
        if not ids:
            raise V2RuntimeQualificationError("V2 chat template produced an empty prompt")
        return V2TokenizedPrompt(
            user_prompt=user_prompt,
            chat_prompt_text=str(prompt_text),
            token_ids=ids,
            token_ids_sha256=sha256_token_ids(ids),
            problem_token_count=problem_token_count,
        )

    def control_token_ids(self, text: str) -> tuple[int, ...]:
        """Tokenize an explicit V2 control string without special tokens."""

        token_ids = tuple(
            int(token_id)
            for token_id in self._tokenizer.encode(text, add_special_tokens=False)
        )
        if not token_ids:
            raise V2RuntimeQualificationError("V2 control text tokenized to no tokens")
        return token_ids

    def validate_readout_contract(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return exact close-marker/cue IDs after proving the V2 cue contract."""

        marker_ids = self.control_token_ids(self._config.generation.close_think_marker_text)
        cue_ids = self.control_token_ids(self._config.readout.cue_text)
        if tuple(cue_ids[: len(marker_ids)]) != marker_ids:
            raise V2RuntimeQualificationError(
                "V2 readout cue must begin with the exact close-thinking marker"
            )
        if _subsequence_start(cue_ids[len(marker_ids) :], marker_ids) is not None:
            raise V2RuntimeQualificationError(
                "V2 readout cue must not include a second close-thinking marker"
            )
        if any(token_id in self._eos_token_ids for token_id in cue_ids):
            raise V2RuntimeQualificationError("V2 readout cue contains an EOS token")
        return marker_ids, cue_ids

    def generate_reasoning(
        self,
        input_token_ids: Sequence[int],
        *,
        seed: int,
        close_think_token_ids: Sequence[int],
    ) -> V2GeneratedSequence:
        """Sample one stochastic trajectory, stopping after the first close marker."""

        decoding = V2DecodingParameters(
            request_kind="V2_BASE_REASONING_STOCHASTIC",
            max_new_tokens=self._config.generation.reasoning_max_new_tokens,
            do_sample=True,
            temperature=self._config.generation.temperature,
            top_p=self._config.generation.top_p,
            top_k=self._config.generation.top_k,
            min_p=self._config.generation.min_p,
            seed=int(seed),
        )
        return self._generate(
            input_token_ids,
            decoding=decoding,
            stop_after_token_sequence=tuple(int(value) for value in close_think_token_ids),
            capture_scores=False,
        )

    def generate_synthetic_reasoning(
        self,
        input_token_ids: Sequence[int],
        *,
        seed: int,
        close_think_token_ids: Sequence[int],
        max_new_tokens: int,
    ) -> V2GeneratedSequence:
        """Run a non-benchmark qualification sample without changing V2 base decoding.

        This public method is intentionally named ``synthetic`` and accepts an
        explicit cap so the hardware preflight cannot accidentally produce a
        scientific DeepMath trajectory.  The sampling distribution is the
        frozen V2 distribution; only the preflight-only length is shorter.
        """

        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise V2RuntimeQualificationError(
                "synthetic V2 max_new_tokens must be a positive integer"
            )
        if max_new_tokens <= 0:
            raise V2RuntimeQualificationError(
                "synthetic V2 max_new_tokens must be positive"
            )
        decoding = V2DecodingParameters(
            request_kind="V2_SYNTHETIC_QUALIFICATION_REASONING",
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=self._config.generation.temperature,
            top_p=self._config.generation.top_p,
            top_k=self._config.generation.top_k,
            min_p=self._config.generation.min_p,
            seed=int(seed),
        )
        return self._generate(
            input_token_ids,
            decoding=decoding,
            stop_after_token_sequence=tuple(
                int(value) for value in close_think_token_ids
            ),
            capture_scores=False,
        )

    def generate_readout(self, input_token_ids: Sequence[int]) -> V2ReadoutSequence:
        """Greedily generate one short V2 readout and reduce scores immediately."""

        decoding = V2DecodingParameters(
            request_kind="V2_SYMMETRIC_GREEDY_READOUT",
            max_new_tokens=self._config.readout.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            min_p=None,
            seed=None,
        )
        sequence, score_tensors = self._generate_with_scores(
            input_token_ids, decoding=decoding
        )
        try:
            scores = self._reduce_readout_scores(
                sequence.generated_token_ids, score_tensors
            )
        finally:
            del score_tensors
            _safe_empty_cache(self._torch)
        return V2ReadoutSequence(generation=sequence, scores=scores)

    def extract_checkpoint_activation(
        self, activation_input: V2ActivationInput
    ) -> ActivationExtraction:
        """Extract the frozen post-block checkpoint representation."""

        self._assert_context(len(activation_input.input_token_ids), max_new_tokens=0)
        return extract_post_block_last_token_activations(
            torch=self._torch,
            model=self._model,
            device=self._device,
            activation_input=activation_input,
            expected_num_hidden_layers=self._config.model.expected_num_hidden_layers,
            expected_hidden_size=self._config.model.expected_hidden_size,
        )

    def close(self) -> None:
        """Release GPU model references only after a bounded V2 command."""

        model = self._model
        self._model = None
        self._tokenizer = None
        del model
        gc.collect()
        _safe_empty_cache(self._torch)

    def _generate(
        self,
        input_token_ids: Sequence[int],
        *,
        decoding: V2DecodingParameters,
        stop_after_token_sequence: tuple[int, ...] | None,
        capture_scores: bool,
    ) -> V2GeneratedSequence:
        if capture_scores:
            raise AssertionError("_generate does not retain score tensors")
        input_ids = tuple(int(value) for value in input_token_ids)
        self._assert_context(len(input_ids), max_new_tokens=decoding.max_new_tokens)
        torch = self._torch
        tensor = None
        attention_mask = None
        sequences = None
        started: float | None = None
        memory_before: CudaMemorySnapshot | None = None
        try:
            torch.cuda.synchronize(self._device)
            torch.cuda.reset_peak_memory_stats(self._device)
            memory_before = capture_cuda_memory(torch, self._device)
            tensor = torch.tensor([input_ids], dtype=torch.long, device=self._device)
            attention_mask = torch.ones_like(tensor, device=self._device)
            kwargs = self._generation_kwargs(decoding)
            if stop_after_token_sequence is not None:
                kwargs["stopping_criteria"] = _stopping_criteria_for_sequence(
                    stop_after_token_sequence
                )
            started = time.perf_counter()
            with _temporary_seed(torch, decoding.seed):
                with torch.inference_mode():
                    sequences = self._model.generate(
                        input_ids=tensor,
                        attention_mask=attention_mask,
                        **kwargs,
                    )
            torch.cuda.synchronize(self._device)
            elapsed = time.perf_counter() - started
            full_ids = tuple(int(value) for value in sequences[0].detach().cpu().tolist())
            if full_ids[: len(input_ids)] != input_ids:
                raise V2GenerationRuntimeError(
                    "generation output did not preserve exact supplied prefix",
                    error_type="PrefixIntegrityError",
                    elapsed_seconds=elapsed,
                    memory_before=memory_before,
                    memory_after=capture_cuda_memory(torch, self._device),
                )
            generated = full_ids[len(input_ids) :]
            return V2GeneratedSequence(
                input_token_ids=input_ids,
                generated_token_ids=generated,
                generated_text=self._decode(generated),
                elapsed_seconds=elapsed,
                termination_status=_termination_status(
                    generated,
                    eos_token_ids=self._eos_token_ids,
                    max_new_tokens=decoding.max_new_tokens,
                    stop_after_token_sequence=stop_after_token_sequence,
                ),
                memory_before=memory_before,
                memory_after=capture_cuda_memory(torch, self._device),
                decoding=decoding,
            )
        except V2GenerationRuntimeError:
            raise
        except Exception as error:
            raise self._generation_error(
                error,
                decoding=decoding,
                started=started,
                memory_before=memory_before,
            ) from error
        finally:
            del sequences, tensor, attention_mask
            _safe_empty_cache(torch)

    def _generate_with_scores(
        self,
        input_token_ids: Sequence[int],
        *,
        decoding: V2DecodingParameters,
    ) -> tuple[V2GeneratedSequence, tuple[Any, ...]]:
        """Run greedy generation while retaining only transient per-step scores."""

        input_ids = tuple(int(value) for value in input_token_ids)
        self._assert_context(len(input_ids), max_new_tokens=decoding.max_new_tokens)
        torch = self._torch
        tensor = None
        attention_mask = None
        output = None
        started: float | None = None
        memory_before: CudaMemorySnapshot | None = None
        try:
            torch.cuda.synchronize(self._device)
            torch.cuda.reset_peak_memory_stats(self._device)
            memory_before = capture_cuda_memory(torch, self._device)
            tensor = torch.tensor([input_ids], dtype=torch.long, device=self._device)
            attention_mask = torch.ones_like(tensor, device=self._device)
            kwargs = self._generation_kwargs(decoding)
            kwargs["return_dict_in_generate"] = True
            kwargs["output_scores"] = True
            started = time.perf_counter()
            with torch.inference_mode():
                output = self._model.generate(
                    input_ids=tensor,
                    attention_mask=attention_mask,
                    **kwargs,
                )
            torch.cuda.synchronize(self._device)
            elapsed = time.perf_counter() - started
            sequences = getattr(output, "sequences", None)
            raw_scores = getattr(output, "scores", None)
            if sequences is None or raw_scores is None:
                raise V2GenerationRuntimeError(
                    "V2 readout generation did not return sequences and scores",
                    error_type="ReadoutScoreContractError",
                    elapsed_seconds=elapsed,
                    memory_before=memory_before,
                    memory_after=capture_cuda_memory(torch, self._device),
                )
            full_ids = tuple(int(value) for value in sequences[0].detach().cpu().tolist())
            if full_ids[: len(input_ids)] != input_ids:
                raise V2GenerationRuntimeError(
                    "readout output did not preserve exact supplied prefix",
                    error_type="PrefixIntegrityError",
                    elapsed_seconds=elapsed,
                    memory_before=memory_before,
                    memory_after=capture_cuda_memory(torch, self._device),
                )
            generated = full_ids[len(input_ids) :]
            scores = tuple(raw_scores)
            if len(scores) != len(generated):
                raise V2GenerationRuntimeError(
                    "readout score count does not equal generated token count",
                    error_type="ReadoutScoreContractError",
                    elapsed_seconds=elapsed,
                    memory_before=memory_before,
                    memory_after=capture_cuda_memory(torch, self._device),
                )
            sequence = V2GeneratedSequence(
                input_token_ids=input_ids,
                generated_token_ids=generated,
                generated_text=self._decode(generated),
                elapsed_seconds=elapsed,
                termination_status=_termination_status(
                    generated,
                    eos_token_ids=self._eos_token_ids,
                    max_new_tokens=decoding.max_new_tokens,
                    stop_after_token_sequence=None,
                ),
                memory_before=memory_before,
                memory_after=capture_cuda_memory(torch, self._device),
                decoding=decoding,
            )
            # Do not delete output until callers reduce raw scores.  The caller
            # owns score_tensors and frees them after scalar extraction.
            return sequence, scores
        except V2GenerationRuntimeError:
            raise
        except Exception as error:
            raise self._generation_error(
                error,
                decoding=decoding,
                started=started,
                memory_before=memory_before,
            ) from error
        finally:
            # output owns scores. Deleting it does not invalidate the tensor
            # references returned in scores, and prevents sequence retention.
            del output, tensor, attention_mask

    def _reduce_readout_scores(
        self,
        generated_token_ids: tuple[int, ...],
        score_tensors: Sequence[Any],
    ) -> V2ReadoutScores:
        """Reduce score tensors to protocol-defined scalars before freeing them."""

        torch = self._torch
        if len(generated_token_ids) != len(score_tensors):
            raise V2GenerationRuntimeError(
                "cannot align V2 readout tokens and score tensors",
                error_type="ReadoutScoreContractError",
            )
        selected_logprobs: list[float] = []
        first_entropy: float | None = None
        first_margin: float | None = None
        for index, (token_id, raw_scores) in enumerate(
            zip(generated_token_ids, score_tensors, strict=True)
        ):
            if tuple(getattr(raw_scores, "shape", ()))[:1] != (1,):
                raise V2GenerationRuntimeError(
                    "V2 readout score tensor must have batch dimension one",
                    error_type="ReadoutScoreContractError",
                )
            logits = raw_scores.float()
            log_probs = torch.log_softmax(logits, dim=-1)
            selected_logprobs.append(float(log_probs[0, int(token_id)].item()))
            if index == 0:
                probabilities = torch.exp(log_probs)
                first_entropy = float(
                    (-(probabilities * log_probs).sum(dim=-1)[0]).item()
                )
                top_two = torch.topk(probabilities[0], k=2).values
                first_margin = float((top_two[0] - top_two[1]).item())
            del logits, log_probs
            if index == 0:
                del probabilities, top_two

        box = find_last_balanced_boxed_boundary(self._decode(generated_token_ids))
        reconstruction = reconstruct_token_text(
            generated_token_ids,
            decode=self._decode,
        )
        region = candidate_token_region(reconstruction, box)
        included = region.included_token_indices
        if included:
            values = tuple(selected_logprobs[index] for index in included)
            mean_logprob = math.fsum(values) / len(values)
            min_logprob = min(values)
            answer_count: int | None = len(values)
        else:
            mean_logprob = None
            min_logprob = None
            answer_count = None
        return V2ReadoutScores(
            selected_token_logprobs=tuple(selected_logprobs),
            first_readout_token_entropy_nats=first_entropy,
            first_readout_token_probability_margin=first_margin,
            box_boundary=box,
            candidate_region=region,
            mean_answer_token_logprob=mean_logprob,
            min_answer_token_logprob=min_logprob,
            answer_token_count=answer_count,
            reconstruction_reason=reconstruction.reason,
        )

    def _generation_kwargs(self, decoding: V2DecodingParameters) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "max_new_tokens": decoding.max_new_tokens,
            "do_sample": decoding.do_sample,
            "num_beams": 1,
            "use_cache": True,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": (
                self._eos_token_ids[0]
                if len(self._eos_token_ids) == 1
                else list(self._eos_token_ids)
            ),
            "return_dict_in_generate": False,
            "output_scores": False,
            "output_hidden_states": False,
            "output_attentions": False,
            # Qwen3's forward otherwise defaults to all sequence logits on a
            # long prefix. Generation needs only the final distribution.
            "logits_to_keep": 1,
        }
        if decoding.do_sample:
            kwargs.update(
                {
                    "temperature": decoding.temperature,
                    "top_p": decoding.top_p,
                    "top_k": decoding.top_k,
                    "min_p": decoding.min_p,
                }
            )
        return kwargs

    def _assert_context(self, input_tokens: int, *, max_new_tokens: int) -> None:
        if input_tokens <= 0:
            raise ValueError("V2 input token sequence cannot be empty")
        required = input_tokens + max_new_tokens
        if (
            self._context_limit_tokens is not None
            and required > self._context_limit_tokens
        ):
            raise V2GenerationRuntimeError(
                f"V2 request needs {required} tokens but context limit is "
                f"{self._context_limit_tokens}",
                error_type="ContextLimitExceeded",
            )

    def _decode(self, token_ids: Sequence[int]) -> str:
        return str(
            self._tokenizer.decode(
                list(int(value) for value in token_ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )

    def _generation_error(
        self,
        error: Exception,
        *,
        decoding: V2DecodingParameters,
        started: float | None,
        memory_before: CudaMemorySnapshot | None,
    ) -> V2GenerationRuntimeError:
        elapsed = time.perf_counter() - started if started is not None else None
        try:
            memory_after = capture_cuda_memory(self._torch, self._device)
        except Exception:
            memory_after = None
        cuda_oom = _is_cuda_oom(self._torch, error)
        if cuda_oom:
            _safe_empty_cache(self._torch)
        return V2GenerationRuntimeError(
            f"{decoding.request_kind} failed: {type(error).__name__}: {str(error)[:300]}",
            error_type=type(error).__name__,
            cuda_oom=cuda_oom,
            elapsed_seconds=elapsed,
            memory_before=memory_before,
            memory_after=memory_after,
        )


def _validate_runtime_versions(config: V2Config) -> None:
    expected = {
        "python": config.runtime.expected_python,
        "torch": config.runtime.expected_torch,
        "transformers": config.runtime.expected_transformers,
        "datasets": config.runtime.expected_datasets,
        "math-verify": config.runtime.expected_math_verify,
        "scikit-learn": config.runtime.expected_sklearn,
        "accelerate": config.runtime.expected_accelerate,
        "huggingface_hub": config.runtime.expected_huggingface_hub,
        "safetensors": config.runtime.expected_safetensors,
    }
    actual = {
        "python": _python_version(),
        **{name: _package_version(name) for name in expected if name != "python"},
    }
    mismatch = {
        name: {"expected": expected[name], "actual": actual[name]}
        for name in expected
        if actual[name] != expected[name]
    }
    if mismatch:
        raise V2RuntimeQualificationError(
            "V2 runtime version mismatch: " + repr(mismatch)
        )


def _validate_model_geometry(model: Any, config: V2Config) -> None:
    model_config = getattr(model, "config", None)
    actual = {
        "model_type": getattr(model_config, "model_type", None),
        "num_hidden_layers": getattr(model_config, "num_hidden_layers", None),
        "hidden_size": getattr(model_config, "hidden_size", None),
    }
    expected = {
        "model_type": config.model.expected_model_type,
        "num_hidden_layers": config.model.expected_num_hidden_layers,
        "hidden_size": config.model.expected_hidden_size,
    }
    if actual != expected:
        raise V2RuntimeQualificationError(
            "loaded V2 model topology differs from frozen contract: "
            + repr({"expected": expected, "actual": actual})
        )


def _runtime_package_versions() -> dict[str, str | None]:
    return {
        name: _package_version(name)
        for name in (
            "torch",
            "transformers",
            "datasets",
            "math-verify",
            "scikit-learn",
            "accelerate",
            "huggingface_hub",
            "safetensors",
        )
    }


def _package_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def _python_version() -> str:
    import platform

    return platform.python_version()


def _stopping_criteria_for_sequence(target: tuple[int, ...]) -> Any:
    if not target:
        raise ValueError("V2 close-thinking target cannot be empty")
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
    except ModuleNotFoundError as error:  # pragma: no cover - runtime import
        raise V2RuntimeQualificationError("Transformers is unavailable") from error

    class _StopAfterSequence(StoppingCriteria):
        def __call__(
            self,
            input_ids: Any,
            scores: Any,
            **kwargs: Any,
        ) -> bool:
            row = tuple(int(value) for value in input_ids[0].detach().cpu().tolist())
            return len(row) >= len(target) and row[-len(target) :] == target

    return StoppingCriteriaList([_StopAfterSequence()])


@contextmanager
def _temporary_seed(torch: Any, seed: int | None) -> Iterator[None]:
    if seed is None:
        yield
        return
    devices: list[int] = []
    index = torch.cuda.current_device() if torch.cuda.is_available() else None
    if index is not None:
        devices = [index]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield


def _termination_status(
    generated_token_ids: Sequence[int],
    *,
    eos_token_ids: Sequence[int],
    max_new_tokens: int,
    stop_after_token_sequence: Sequence[int] | None,
) -> str:
    generated = tuple(int(value) for value in generated_token_ids)
    target = tuple(int(value) for value in stop_after_token_sequence or ())
    if target and len(generated) >= len(target) and generated[-len(target) :] == target:
        return "THINK_CLOSE_MARKER"
    if generated and generated[-1] in {int(value) for value in eos_token_ids}:
        return "EOS"
    if len(generated) >= max_new_tokens:
        return "MAX_NEW_TOKENS"
    return "STOPPED_OTHER"


def _subsequence_start(tokens: Sequence[int], target: Sequence[int]) -> int | None:
    if not target:
        raise ValueError("token target cannot be empty")
    for start in range(len(tokens) - len(target) + 1):
        if tuple(tokens[start : start + len(target)]) == tuple(target):
            return start
    return None


def _is_cuda_oom(torch: Any, error: Exception) -> bool:
    oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
    return (oom_type is not None and isinstance(error, oom_type)) or (
        "out of memory" in str(error).lower()
    )


def _safe_empty_cache(torch: Any) -> None:
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
