"""Memory-bounded V2 residual-stream activation extraction.

The V2 representation is exactly the last sequence position from every
post-transformer-block output of the Qwen base transformer.  This module never
asks a CausalLM for logits or output_hidden_states, never includes the readout
cue in its input builder, and transfers one BF16 vector per layer to CPU inside
the hook callback.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import time
from typing import Any, Iterable, Sequence

from .provenance import canonical_json, sha256_token_ids


class ActivationExtractionError(RuntimeError):
    """A typed, non-retryable activation extraction failure for one request."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str,
        cuda_oom: bool = False,
        elapsed_seconds: float | None = None,
        memory_before: "CudaMemorySnapshot | None" = None,
        memory_after: "CudaMemorySnapshot | None" = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.cuda_oom = cuda_oom
        self.elapsed_seconds = elapsed_seconds
        self.memory_before = memory_before
        self.memory_after = memory_after


@dataclass(frozen=True)
class CudaMemorySnapshot:
    """CUDA allocator telemetry; it is not a proof of card-wide free memory."""

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
class V2ActivationInput:
    """Exact prompt-plus-saved-prefix input with no readout cue parameter."""

    prompt_token_ids: tuple[int, ...]
    reasoning_prefix_token_ids: tuple[int, ...]
    input_token_ids: tuple[int, ...]
    prompt_sha256: str
    reasoning_prefix_sha256: str
    input_sha256: str

    def __post_init__(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError("activation input requires a non-empty prompt token sequence")
        if self.input_token_ids != self.prompt_token_ids + self.reasoning_prefix_token_ids:
            raise ValueError("activation input must be exactly prompt plus saved reasoning prefix")
        if self.prompt_sha256 != sha256_token_ids(self.prompt_token_ids):
            raise ValueError("activation prompt hash does not match its exact token IDs")
        if self.reasoning_prefix_sha256 != sha256_token_ids(self.reasoning_prefix_token_ids):
            raise ValueError("activation prefix hash does not match its exact token IDs")
        if self.input_sha256 != sha256_token_ids(self.input_token_ids):
            raise ValueError("activation input hash does not match its exact token IDs")

    def safe_summary(self) -> dict[str, object]:
        """Metadata safe for an aggregate receipt; no raw prompt/prefix IDs."""

        return {
            "input_policy": "prompt-plus-exact-prefix-no-readout-cue-v1",
            "prompt_token_count": len(self.prompt_token_ids),
            "reasoning_prefix_token_count": len(self.reasoning_prefix_token_ids),
            "input_token_count": len(self.input_token_ids),
            "prompt_sha256": self.prompt_sha256,
            "reasoning_prefix_sha256": self.reasoning_prefix_sha256,
            "input_sha256": self.input_sha256,
        }


def build_activation_input(
    prompt_token_ids: Iterable[int], reasoning_prefix_token_ids: Iterable[int]
) -> V2ActivationInput:
    """Build the only supported activation input form.

    There is deliberately no readout-cue argument.  The separate deterministic
    readout builder owns that cue, preventing a measurement-operation state
    from contaminating a reasoning-prefix activation.
    """

    prompt = tuple(int(token_id) for token_id in prompt_token_ids)
    prefix = tuple(int(token_id) for token_id in reasoning_prefix_token_ids)
    if not prompt:
        raise ValueError("prompt_token_ids cannot be empty")
    combined = prompt + prefix
    return V2ActivationInput(
        prompt_token_ids=prompt,
        reasoning_prefix_token_ids=prefix,
        input_token_ids=combined,
        prompt_sha256=sha256_token_ids(prompt),
        reasoning_prefix_sha256=sha256_token_ids(prefix),
        input_sha256=sha256_token_ids(combined),
    )


@dataclass(frozen=True)
class ActivationExtraction:
    """One private BF16 [layer, width] matrix and receipt-safe metadata."""

    matrix: Any
    layer_count: int
    hidden_size: int
    matrix_shape: tuple[int, int]
    matrix_dtype: str
    matrix_sha256: str
    raw_matrix_bytes: int
    input_sha256: str
    elapsed_seconds: float
    memory_before: CudaMemorySnapshot
    memory_after: CudaMemorySnapshot

    def safe_summary(self) -> dict[str, object]:
        return {
            "protocol_id": "post-block-last-reasoning-token-v1",
            "layer_indexing": "zero-based-transformer-block-output-before-final-norm",
            "include_embedding_state": False,
            "include_final_norm_state": False,
            "storage_dtype": "bfloat16",
            "storage_format": "torch-save-bfloat16-matrix-v1",
            "matrix_shape": list(self.matrix_shape),
            "matrix_dtype": self.matrix_dtype,
            "matrix_sha256": self.matrix_sha256,
            "raw_matrix_bytes": self.raw_matrix_bytes,
            "input_sha256": self.input_sha256,
            "elapsed_seconds": self.elapsed_seconds,
            "memory_before": self.memory_before.to_dict(),
            "memory_after": self.memory_after.to_dict(),
        }


def capture_cuda_memory(torch: Any, device: Any) -> CudaMemorySnapshot:
    """Capture allocator telemetry with a single consistent V2 schema."""

    return CudaMemorySnapshot(
        allocated_bytes=int(torch.cuda.memory_allocated(device)),
        reserved_bytes=int(torch.cuda.memory_reserved(device)),
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
    )


def resolve_qwen_transformer_blocks(
    model: Any, *, expected_num_hidden_layers: int
) -> tuple[Any, tuple[Any, ...]]:
    """Resolve only Qwen's base-transformer block stack, never the CausalLM head."""

    if isinstance(expected_num_hidden_layers, bool) or not isinstance(
        expected_num_hidden_layers, int
    ) or expected_num_hidden_layers <= 0:
        raise ActivationExtractionError(
            "expected_num_hidden_layers must be a positive integer",
            error_type="ActivationTopologyError",
        )
    base_transformer = getattr(model, "model", None)
    if base_transformer is None or base_transformer is model:
        raise ActivationExtractionError(
            "V2 requires a distinct CausalLM.model base transformer",
            error_type="ActivationTopologyError",
        )
    layers = getattr(base_transformer, "layers", None)
    if layers is None:
        raise ActivationExtractionError(
            "Qwen base transformer exposes no .layers block stack",
            error_type="ActivationTopologyError",
        )
    try:
        blocks = tuple(layers)
    except TypeError as error:
        raise ActivationExtractionError(
            "Qwen base-transformer layers are not iterable",
            error_type="ActivationTopologyError",
        ) from error
    if len(blocks) != expected_num_hidden_layers:
        raise ActivationExtractionError(
            "loaded transformer layer count does not match the frozen V2 contract",
            error_type="ActivationTopologyError",
        )
    if any(not callable(getattr(block, "register_forward_hook", None)) for block in blocks):
        raise ActivationExtractionError(
            "one or more transformer blocks cannot register a forward hook",
            error_type="ActivationTopologyError",
        )
    return base_transformer, blocks


def extract_post_block_last_token_activations(
    *,
    torch: Any,
    model: Any,
    device: Any,
    activation_input: V2ActivationInput,
    expected_num_hidden_layers: int,
    expected_hidden_size: int,
) -> ActivationExtraction:
    """Run one no-cache base-transformer forward and retain only CPU BF16 vectors.

    Hooks observe each post-block output, copy output[0, -1, :] to CPU BF16,
    and immediately discard their GPU references.  The function does not set
    output_hidden_states and does not invoke the CausalLM wrapper, so it cannot
    retain a layer-by-sequence collection or long-prefix logits.
    """

    if isinstance(expected_hidden_size, bool) or not isinstance(expected_hidden_size, int):
        raise ActivationExtractionError(
            "expected_hidden_size must be a positive integer",
            error_type="ActivationTopologyError",
        )
    if expected_hidden_size <= 0:
        raise ActivationExtractionError(
            "expected_hidden_size must be a positive integer",
            error_type="ActivationTopologyError",
        )
    base_transformer, blocks = resolve_qwen_transformer_blocks(
        model, expected_num_hidden_layers=expected_num_hidden_layers
    )
    if not activation_input.input_token_ids:
        raise ActivationExtractionError(
            "activation input cannot be empty",
            error_type="ActivationInputError",
        )

    handles: list[Any] = []
    captured: dict[int, Any] = {}
    call_counts = [0 for _ in blocks]
    input_ids = None
    attention_mask = None
    forward_output = None
    started: float | None = None
    memory_before: CudaMemorySnapshot | None = None
    try:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        memory_before = capture_cuda_memory(torch, device)
        for layer_index, block in enumerate(blocks):
            handles.append(
                block.register_forward_hook(
                    _last_token_cpu_bf16_hook(
                        layer_index=layer_index,
                        expected_hidden_size=expected_hidden_size,
                        torch=torch,
                        captured=captured,
                        call_counts=call_counts,
                    )
                )
            )
        input_ids = torch.tensor(
            [activation_input.input_token_ids], dtype=torch.long, device=device
        )
        attention_mask = torch.ones_like(input_ids, device=device)
        started = time.perf_counter()
        with torch.inference_mode():
            forward_output = base_transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
            )
        torch.cuda.synchronize(device)
        elapsed_seconds = time.perf_counter() - started
        if tuple(call_counts) != tuple(1 for _ in blocks):
            raise ActivationExtractionError(
                "every expected transformer block must execute exactly once",
                error_type="ActivationHookCoverageError",
                elapsed_seconds=elapsed_seconds,
                memory_before=memory_before,
                memory_after=capture_cuda_memory(torch, device),
            )
        if tuple(sorted(captured)) != tuple(range(len(blocks))):
            raise ActivationExtractionError(
                "activation hook output is missing a transformer layer",
                error_type="ActivationHookCoverageError",
                elapsed_seconds=elapsed_seconds,
                memory_before=memory_before,
                memory_after=capture_cuda_memory(torch, device),
            )
        matrix = torch.stack([captured[index] for index in range(len(blocks))], dim=0)
        matrix = _cpu_bfloat16_contiguous(matrix, torch)
        _validate_activation_matrix(
            matrix,
            torch=torch,
            expected_num_hidden_layers=expected_num_hidden_layers,
            expected_hidden_size=expected_hidden_size,
        )
        memory_after = capture_cuda_memory(torch, device)
        return ActivationExtraction(
            matrix=matrix,
            layer_count=expected_num_hidden_layers,
            hidden_size=expected_hidden_size,
            matrix_shape=(expected_num_hidden_layers, expected_hidden_size),
            matrix_dtype=str(matrix.dtype),
            matrix_sha256=activation_matrix_sha256(matrix, torch=torch),
            raw_matrix_bytes=int(matrix.numel() * matrix.element_size()),
            input_sha256=activation_input.input_sha256,
            elapsed_seconds=elapsed_seconds,
            memory_before=memory_before,
            memory_after=memory_after,
        )
    except ActivationExtractionError:
        raise
    except Exception as error:
        elapsed_seconds = time.perf_counter() - started if started is not None else None
        try:
            memory_after = capture_cuda_memory(torch, device)
        except Exception:
            memory_after = None
        is_oom = _is_cuda_oom(torch, error)
        if is_oom:
            _safe_empty_cache(torch)
        raise ActivationExtractionError(
            f"activation extraction failed: {type(error).__name__}: {str(error)[:300]}",
            error_type=type(error).__name__,
            cuda_oom=is_oom,
            elapsed_seconds=elapsed_seconds,
            memory_before=memory_before,
            memory_after=memory_after,
        ) from error
    finally:
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass
        # The only durable GPU-independent values are the CPU vectors/matrix.
        captured.clear()
        del forward_output, input_ids, attention_mask
        _safe_empty_cache(torch)


def activation_matrix_sha256(matrix: Any, *, torch: Any) -> str:
    """Hash a CPU BF16 matrix together with its unambiguous geometry header."""

    _validate_cpu_bfloat16_matrix(matrix, torch=torch)
    raw = matrix.detach().contiguous().view(torch.uint8).numpy().tobytes()
    header = canonical_json(
        {
            "shape": [int(size) for size in matrix.shape],
            "dtype": "bfloat16",
            "representation": "post-block-last-reasoning-token-v1",
        }
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(raw)
    return digest.hexdigest()


def serialize_activation_matrix(matrix: Any, *, torch: Any) -> bytes:
    """Serialize exactly one CPU BF16 V2 matrix for a private immutable artifact."""

    _validate_cpu_bfloat16_matrix(matrix, torch=torch)
    stream = io.BytesIO()
    torch.save(matrix.detach().contiguous(), stream)
    return stream.getvalue()


def _last_token_cpu_bf16_hook(
    *,
    layer_index: int,
    expected_hidden_size: int,
    torch: Any,
    captured: dict[int, Any],
    call_counts: list[int],
) -> Any:
    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        call_counts[layer_index] += 1
        if call_counts[layer_index] != 1:
            raise ActivationExtractionError(
                "a transformer block executed more than once in one V2 forward",
                error_type="ActivationHookCoverageError",
            )
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if not _looks_like_hidden_state(hidden):
            raise ActivationExtractionError(
                "transformer block hook did not receive a hidden-state tensor",
                error_type="ActivationHookOutputError",
            )
        if int(hidden.shape[0]) != 1 or int(hidden.shape[-1]) != expected_hidden_size:
            raise ActivationExtractionError(
                "post-block hidden-state shape violates the frozen V2 geometry",
                error_type="ActivationHookOutputError",
            )
        # This is the sole retained hook value.  It is copied before returning
        # from the hook, so no GPU output tensor is kept in the capture map.
        vector = hidden[0, -1, :].detach()
        captured[layer_index] = _cpu_bfloat16_contiguous(vector, torch)

    return hook


def _looks_like_hidden_state(value: Any) -> bool:
    shape = getattr(value, "shape", None)
    return shape is not None and len(shape) == 3


def _cpu_bfloat16_contiguous(value: Any, torch: Any) -> Any:
    try:
        converted = value.to(device="cpu", dtype=torch.bfloat16, copy=True)
    except TypeError:
        converted = value.to(device="cpu", dtype=torch.bfloat16).clone()
    return converted.contiguous()


def _validate_activation_matrix(
    matrix: Any,
    *,
    torch: Any,
    expected_num_hidden_layers: int,
    expected_hidden_size: int,
) -> None:
    _validate_cpu_bfloat16_matrix(matrix, torch=torch)
    shape = tuple(int(size) for size in matrix.shape)
    if shape != (expected_num_hidden_layers, expected_hidden_size):
        raise ActivationExtractionError(
            "assembled activation matrix shape violates the frozen V2 geometry",
            error_type="ActivationShapeError",
        )


def _validate_cpu_bfloat16_matrix(matrix: Any, *, torch: Any) -> None:
    if getattr(matrix, "dtype", None) != torch.bfloat16:
        raise ActivationExtractionError(
            "V2 activation matrix must be stored in bfloat16",
            error_type="ActivationDtypeError",
        )
    device = getattr(matrix, "device", None)
    if getattr(device, "type", None) != "cpu":
        raise ActivationExtractionError(
            "V2 activation matrix must be copied to CPU before storage",
            error_type="ActivationDeviceError",
        )
    if len(getattr(matrix, "shape", ())) != 2:
        raise ActivationExtractionError(
            "V2 activation matrix must be two-dimensional",
            error_type="ActivationShapeError",
        )


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
