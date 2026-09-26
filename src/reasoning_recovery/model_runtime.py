"""Memory-conscious native Transformers runtime for P0.

This module does not load a model on import. The only model-loading path is
TransformersRuntime.load, which is reached by the explicit run command.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import gc
from pathlib import Path
from typing import Any, Iterator, Sequence
import time

from .config import GenerationConfig, ModelConfig
from .provenance import (
    base_runtime_provenance,
    sha256_file,
    sha256_text,
    sha256_token_ids,
)


class RuntimeQualificationError(RuntimeError):
    """Raised before a model request when the declared runtime is unavailable."""


class GenerationRuntimeError(RuntimeError):
    """Raised when a generation request fails after runtime initialization."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str,
        cuda_oom: bool = False,
        elapsed_seconds: float | None = None,
        memory_before: MemorySnapshot | None = None,
        memory_after: MemorySnapshot | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.cuda_oom = cuda_oom
        self.elapsed_seconds = elapsed_seconds
        self.memory_before = memory_before
        self.memory_after = memory_after


@dataclass(frozen=True)
class MemorySnapshot:
    allocated_bytes: int | None
    reserved_bytes: int | None
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None

    def to_dict(self) -> dict[str, int | None]:
        return {
            "allocated_bytes": self.allocated_bytes,
            "reserved_bytes": self.reserved_bytes,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
        }


@dataclass(frozen=True)
class TokenizedPrompt:
    user_prompt: str
    chat_prompt_text: str
    token_ids: tuple[int, ...]
    token_ids_sha256: str


@dataclass(frozen=True)
class DecodingParameters:
    request_kind: str
    max_new_tokens: int
    do_sample: bool
    temperature: float | None
    top_p: float | None
    top_k: int | None
    min_p: float | None
    seed: int | None

    @classmethod
    def base(cls, config: GenerationConfig, seed: int) -> "DecodingParameters":
        return cls(
            request_kind="BASE_STOCHASTIC",
            max_new_tokens=config.max_new_tokens,
            do_sample=True,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            min_p=config.min_p,
            seed=seed,
        )

    @classmethod
    def forced(cls, max_new_tokens: int) -> "DecodingParameters":
        return cls(
            request_kind="FORCED_GREEDY",
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            min_p=None,
            seed=None,
        )

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
class GeneratedSequence:
    input_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    generated_text: str
    elapsed_seconds: float
    termination_status: str
    memory_before: MemorySnapshot
    memory_after: MemorySnapshot
    decoding: DecodingParameters

    def to_dict(self, *, include_input_token_ids: bool = True) -> dict[str, object]:
        """Serialize a completion while avoiding unnecessary child duplication."""

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
            "memory_delta_bytes": {
                "allocated": _difference(
                    self.memory_after.allocated_bytes, self.memory_before.allocated_bytes
                ),
                "reserved": _difference(
                    self.memory_after.reserved_bytes, self.memory_before.reserved_bytes
                ),
            },
            "decoding": self.decoding.to_dict(),
        }
        if include_input_token_ids:
            payload["input_token_ids"] = list(self.input_token_ids)
        return payload


class TransformersRuntime:
    """One-model, batch-one runtime that deliberately avoids hidden states."""

    def __init__(
        self,
        *,
        torch: Any,
        tokenizer: Any,
        model: Any,
        model_config: ModelConfig,
        device: Any,
        context_limit_tokens: int | None,
        eos_token_ids: tuple[int, ...],
        provenance: dict[str, object],
    ) -> None:
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._model_config = model_config
        self._device = device
        self._context_limit_tokens = context_limit_tokens
        self._eos_token_ids = eos_token_ids
        self._provenance = provenance

    @classmethod
    def load(cls, config: ModelConfig) -> "TransformersRuntime":
        """Load the declared model only after the caller has passed all gates."""

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from huggingface_hub import HfApi
        except ModuleNotFoundError as error:
            raise RuntimeQualificationError(
                "torch, transformers, and huggingface_hub are required; install .[runtime] plus a CUDA torch build"
            ) from error

        if not torch.cuda.is_available():
            raise RuntimeQualificationError("CUDA is unavailable; P0 does not fall back to CPU")
        device = torch.device(config.device)
        if device.type != "cuda":
            raise RuntimeQualificationError("P0 requires a CUDA device")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeQualificationError(f"configured CUDA device is unavailable: {config.device}")
        if config.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeQualificationError("configured bfloat16 is unsupported on this CUDA device")

        model_revision = _resolve_hub_revision(
            HfApi(), config.model_id, config.revision, label="model"
        )
        tokenizer_requested_revision = config.tokenizer_revision or config.revision
        tokenizer_revision = (
            model_revision
            if tokenizer_requested_revision == config.revision
            else _resolve_hub_revision(
                HfApi(),
                config.model_id,
                tokenizer_requested_revision,
                label="tokenizer",
            )
        )
        dtype = getattr(torch, config.dtype)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_id,
                revision=tokenizer_revision,
                trust_remote_code=False,
            )
            pad_inherited = False
            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token is None:
                    raise RuntimeQualificationError("tokenizer has neither pad token nor EOS token")
                tokenizer.pad_token = tokenizer.eos_token
                pad_inherited = True
            model = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                revision=model_revision,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=False,
            )
            model.to(device)
            model.eval()
        except RuntimeQualificationError:
            raise
        except Exception as error:
            _safe_empty_cache(torch)
            raise RuntimeQualificationError(
                f"model/tokenizer load failed: {type(error).__name__}: {str(error)[:300]}"
            ) from error

        properties = torch.cuda.get_device_properties(device)
        chat_template = getattr(tokenizer, "chat_template", None)
        model_commit = getattr(getattr(model, "config", None), "_commit_hash", None)
        tokenizer_commit = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        context_limit_tokens = _context_limit(model, tokenizer)
        eos_ids = _resolved_eos_token_ids(model, tokenizer)
        asset_manifest = _cached_asset_manifest(
            config.model_id, (model_revision, tokenizer_revision)
        )
        if asset_manifest["status"] != "COMPLETE":
            del model, tokenizer
            gc.collect()
            _safe_empty_cache(torch)
            raise RuntimeQualificationError(
                "could not produce a complete local asset-hash manifest for the resolved model/tokenizer revisions"
            )
        provenance = {
            **base_runtime_provenance(),
            "model": {
                "id": config.model_id,
                "requested_revision": config.revision,
                "resolved_revision": model_revision,
                "loaded_config_commit": model_commit,
                "loaded_config_commit_matches_resolved": (
                    None if model_commit is None else model_commit == model_revision
                ),
                "dtype": config.dtype,
                "device": str(device),
                "model_type": getattr(getattr(model, "config", None), "model_type", None),
                "context_limit_tokens": context_limit_tokens,
            },
            "tokenizer": {
                "requested_revision": tokenizer_requested_revision,
                "resolved_revision": tokenizer_revision,
                "loaded_init_commit": tokenizer_commit,
                "loaded_init_commit_matches_resolved": (
                    None if tokenizer_commit is None else tokenizer_commit == tokenizer_revision
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
                "total_memory_bytes": properties.total_memory,
                "capability": [properties.major, properties.minor],
                "torch_cuda_build": getattr(getattr(torch, "version", None), "cuda", None),
            },
            "memory_after_load": _memory_snapshot(torch, device).to_dict(),
        }
        return cls(
            torch=torch,
            tokenizer=tokenizer,
            model=model,
            model_config=config,
            device=device,
            context_limit_tokens=context_limit_tokens,
            eos_token_ids=eos_ids,
            provenance=provenance,
        )

    @property
    def provenance(self) -> dict[str, object]:
        return self._provenance

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        return self._eos_token_ids

    def tokenize_prompt(self, user_prompt: str) -> TokenizedPrompt:
        """Tokenize the chat template directly; never decode and re-tokenize it."""

        messages = [{"role": "user", "content": user_prompt}]
        try:
            encoded = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=self._model_config.enable_thinking,
            )
            input_ids = encoded["input_ids"][0].detach().cpu().tolist()
            chat_prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self._model_config.enable_thinking,
            )
        except Exception as error:
            raise RuntimeQualificationError(
                f"thinking chat-template construction failed: {type(error).__name__}"
            ) from error
        token_ids = tuple(int(token_id) for token_id in input_ids)
        if not token_ids:
            raise RuntimeQualificationError("chat template produced an empty prompt")
        return TokenizedPrompt(
            user_prompt=user_prompt,
            chat_prompt_text=str(chat_prompt),
            token_ids=token_ids,
            token_ids_sha256=sha256_token_ids(token_ids),
        )

    def control_token_ids(self, text: str) -> tuple[int, ...]:
        """Tokenize an explicit forced-answer control string without specials."""

        token_ids = tuple(
            int(token_id)
            for token_id in self._tokenizer.encode(text, add_special_tokens=False)
        )
        if not token_ids:
            raise RuntimeQualificationError("control text tokenized to an empty sequence")
        return token_ids

    def validate_forced_cue(
        self, *, close_marker_text: str, cue_text: str
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Validate and return marker and cue IDs before any forced request."""

        marker_ids = self.control_token_ids(close_marker_text)
        cue_ids = self.control_token_ids(cue_text)
        if tuple(cue_ids[: len(marker_ids)]) != marker_ids:
            raise RuntimeQualificationError(
                "forced cue must begin with exactly the close-think marker"
            )
        if _subsequence_start(cue_ids[len(marker_ids) :], marker_ids) is not None:
            raise RuntimeQualificationError(
                "forced cue must not contain a second close-think marker"
            )
        if any(token_id in self.eos_token_ids for token_id in cue_ids):
            raise RuntimeQualificationError("forced cue contains an EOS token")
        return marker_ids, cue_ids

    def generate(
        self, input_token_ids: Sequence[int], decoding: DecodingParameters
    ) -> GeneratedSequence:
        """Generate batch one without scores, attention, or hidden-state retention."""

        if not input_token_ids:
            raise ValueError("input_token_ids cannot be empty")
        input_ids = tuple(int(token_id) for token_id in input_token_ids)
        required_context = len(input_ids) + decoding.max_new_tokens
        if (
            self._context_limit_tokens is not None
            and required_context > self._context_limit_tokens
        ):
            raise GenerationRuntimeError(
                f"request needs {required_context} tokens but runtime limit is "
                f"{self._context_limit_tokens}",
                error_type="ContextLimitExceeded",
            )
        torch = self._torch
        device = self._device
        tensor = None
        attention_mask = None
        sequences = None
        started: float | None = None
        memory_before: MemorySnapshot | None = None
        try:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            memory_before = _memory_snapshot(torch, device)
            tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(tensor, device=device)
            started = time.perf_counter()
            with _temporary_seed(torch, decoding.seed):
                with torch.inference_mode():
                    sequences = self._model.generate(
                        input_ids=tensor,
                        attention_mask=attention_mask,
                        **self._generation_kwargs(decoding),
                    )
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            full_ids = tuple(int(token_id) for token_id in sequences[0].detach().cpu().tolist())
            if full_ids[: len(input_ids)] != input_ids:
                raise GenerationRuntimeError(
                    "generation output did not preserve the supplied saved prefix",
                    error_type="PrefixIntegrityError",
                    elapsed_seconds=elapsed,
                    memory_before=memory_before,
                    memory_after=_memory_snapshot(torch, device),
                )
            generated_ids = full_ids[len(input_ids) :]
            text = self._tokenizer.decode(
                generated_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            return GeneratedSequence(
                input_token_ids=input_ids,
                generated_token_ids=generated_ids,
                generated_text=str(text),
                elapsed_seconds=elapsed,
                termination_status=_termination_status(
                    generated_ids, self.eos_token_ids, decoding.max_new_tokens
                ),
                memory_before=memory_before,
                memory_after=_memory_snapshot(torch, device),
                decoding=decoding,
            )
        except GenerationRuntimeError:
            raise
        except Exception as error:
            is_oom = _is_cuda_oom(torch, error)
            elapsed = time.perf_counter() - started if started is not None else None
            try:
                memory_after = _memory_snapshot(torch, device)
            except Exception:
                memory_after = None
            if is_oom:
                _safe_empty_cache(torch)
            raise GenerationRuntimeError(
                f"{decoding.request_kind} generation failed: {type(error).__name__}: {str(error)[:300]}",
                error_type=type(error).__name__,
                cuda_oom=is_oom,
                elapsed_seconds=elapsed,
                memory_before=memory_before,
                memory_after=memory_after,
            ) from error
        finally:
            del sequences, tensor, attention_mask
            _safe_empty_cache(torch)

    def close(self) -> None:
        """Release model references promptly after a bounded run."""

        model = self._model
        self._model = None
        self._tokenizer = None
        del model
        gc.collect()
        _safe_empty_cache(self._torch)

    def _generation_kwargs(self, decoding: DecodingParameters) -> dict[str, object]:
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


@contextmanager
def _temporary_seed(torch: Any, seed: int | None) -> Iterator[None]:
    """Make one sampled request reproducible without leaking RNG state outward."""

    if seed is None:
        yield
        return
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all()
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        torch.cuda.set_rng_state_all(cuda_states)


def _memory_snapshot(torch: Any, device: Any) -> MemorySnapshot:
    return MemorySnapshot(
        allocated_bytes=int(torch.cuda.memory_allocated(device)),
        reserved_bytes=int(torch.cuda.memory_reserved(device)),
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
    )


def _safe_empty_cache(torch: Any) -> None:
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _difference(after: int | None, before: int | None) -> int | None:
    if after is None or before is None:
        return None
    return after - before


def _context_limit(model: Any, tokenizer: Any) -> int | None:
    candidates = (
        getattr(getattr(model, "config", None), "max_position_embeddings", None),
        getattr(tokenizer, "model_max_length", None),
    )
    for candidate in candidates:
        if isinstance(candidate, int) and 0 < candidate < 10_000_000:
            return candidate
    return None


def _resolve_hub_revision(
    api: Any, model_id: str, requested_revision: str | None, *, label: str
) -> str:
    """Resolve a Hub ref to one immutable commit before downloading assets.

    A floating main branch is useful only for a deliberately reviewed smoke
    run. It is still resolved *before* tokenizer/model loading, so both use
    the same immutable source rather than whatever the branch means later.
    """

    try:
        info = api.model_info(model_id, revision=requested_revision)
    except Exception as error:
        raise RuntimeQualificationError(
            f"could not resolve {label} revision for {model_id!r}: {type(error).__name__}"
        ) from error
    revision = getattr(info, "sha", None)
    if not isinstance(revision, str) or not revision:
        raise RuntimeQualificationError(
            f"Hub did not provide an immutable {label} revision for {model_id!r}"
        )
    return revision


def _cached_asset_manifest(
    model_id: str, revisions: Sequence[str]
) -> dict[str, object]:
    """Hash the local Hub snapshot assets actually available after loading.

    The record deliberately stores repository-relative names, sizes, and
    digests—not cache paths—so it can bind an experiment to exact assets without
    publishing machine-specific locations.  A missing revision fails closed;
    a commit alone is not treated as a substitute for local asset evidence.
    """

    expected = set(revisions)
    files_by_revision: dict[str, list[dict[str, object]]] = {}
    try:
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir()
        repositories = getattr(cache, "repos", ())
        for repository in repositories:
            if getattr(repository, "repo_id", None) != model_id:
                continue
            for revision_info in getattr(repository, "revisions", ()):
                commit_hash = getattr(revision_info, "commit_hash", None)
                if commit_hash not in expected:
                    continue
                files: list[dict[str, object]] = []
                for file_info in getattr(revision_info, "files", ()):
                    file_path = Path(getattr(file_info, "file_path"))
                    if not file_path.is_file():
                        continue
                    file_name = getattr(file_info, "file_name", None)
                    files.append(
                        {
                            "name": str(file_name) if file_name else file_path.name,
                            "size_bytes": file_path.stat().st_size,
                            "sha256": sha256_file(file_path),
                        }
                    )
                if files:
                    files_by_revision[commit_hash] = sorted(
                        files, key=lambda item: str(item["name"])
                    )
    except Exception as error:
        return {
            "status": "UNAVAILABLE",
            "error_type": type(error).__name__,
            "expected_revisions": sorted(expected),
        }

    missing = sorted(revision for revision in expected if revision not in files_by_revision)
    if missing:
        return {
            "status": "INCOMPLETE",
            "expected_revisions": sorted(expected),
            "missing_revisions": missing,
        }
    return {
        "status": "COMPLETE",
        "repository_id": model_id,
        "revisions": [
            {"revision": revision, "files": files_by_revision[revision]}
            for revision in sorted(expected)
        ],
    }


def _generation_defaults(model: Any) -> dict[str, object]:
    config = getattr(model, "generation_config", None)
    names = (
        "max_length",
        "max_new_tokens",
        "do_sample",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
    )
    return {name: getattr(config, name, None) for name in names}


def _normalise_token_ids(value: object) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    raise RuntimeQualificationError("tokenizer EOS token IDs are unavailable")


def _resolved_eos_token_ids(model: Any, tokenizer: Any) -> tuple[int, ...]:
    generation_config = getattr(model, "generation_config", None)
    configured = getattr(generation_config, "eos_token_id", None)
    if configured is None:
        configured = getattr(getattr(model, "config", None), "eos_token_id", None)
    if configured is None:
        configured = tokenizer.eos_token_id
    return _normalise_token_ids(configured)


def _subsequence_start(tokens: Sequence[int], target: Sequence[int]) -> int | None:
    if not target:
        return None
    for start in range(len(tokens) - len(target) + 1):
        if tuple(tokens[start : start + len(target)]) == tuple(target):
            return start
    return None


def _termination_status(
    generated_ids: Sequence[int], eos_token_ids: Sequence[int], max_new_tokens: int
) -> str:
    if generated_ids and generated_ids[-1] in set(eos_token_ids):
        return "EOS"
    if len(generated_ids) >= max_new_tokens:
        return "MAX_NEW_TOKENS"
    return "OTHER_STOP"


def _is_cuda_oom(torch: Any, error: Exception) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    return (oom_type is not None and isinstance(error, oom_type)) or (
        "out of memory" in str(error).lower()
    )
