# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_offload.mediums import BlockIDsLoadStoreSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

logger = init_logger(__name__)


def expand_block_ids(
    block_ids: np.ndarray,
    block_size_factor: int,
    output: np.ndarray,
    skip_count: int = 0,
):
    """
    Convert a list of block IDs to a list of matching block ids,
    assuming each block is composed of actual block_size_factor blocks.
    Outputs to output tensor.
    The first skip_count blocks will be skipped.
    Note that skip_count must be less than block_size_factor.

    For example, if block_ids = [0, 1, 3] and block_size_factor =  4,
    then it yields [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15]
    since 0 maps to [0, 1, 2, 3]
    1 maps to [4, 5, 6, 7]
    and 3 maps to [12, 13, 14, 15]
    """
    assert skip_count < block_size_factor

    first_range = np.arange(skip_count, block_size_factor)
    full_range = np.arange(0, block_size_factor)

    output_idx = 0
    for i, block_id in enumerate(block_ids):
        base_block_id = block_id * block_size_factor
        indices = first_range if i == 0 else full_range
        output_end_idx = output_idx + len(indices)
        output[output_idx:output_end_idx] = base_block_id + indices
        output_idx = output_end_idx


class SingleDirectionOffloadingHandler(OffloadingHandler):
    """
    SingleDirectionOffloadingHandler handles transfers for a single direction,
    either CPU->GPU or GPU->CPU.
    Transfers are guaranteed to be executed in order of their submission.
    Each transfer uses a unique CUDA stream, and its stream will start
    executing only after the streams of previous transfers have finished.
    """

    def __init__(
        self,
        src_tensors: list[torch.Tensor],
        dst_tensors: list[torch.Tensor],
        src_block_size_factor: int,
        dst_block_size_factor: int,
    ):
        """
        Initialize a SingleDirectionOffloadingHandler.

        Args:
            src_tensors: list of KV cache tensors to copy from.
            dst_tensors: list of KV cache tensors to copy to.
                Order should match src_tensors.
            src_block_size_factor: The number of kernel blocks
                per KV block in a source tensor.
            dst_block_size_factor: The number of kernel blocks
                per KV block in a destination tensor.
        """
        assert len(src_tensors) == len(dst_tensors)

        self.src_tensors: list[torch.Tensor] = src_tensors
        self.dst_tensors: list[torch.Tensor] = dst_tensors
        min_block_size_factor = min(src_block_size_factor, dst_block_size_factor)
        self.src_block_size_factor: int = src_block_size_factor // min_block_size_factor
        self.dst_block_size_factor: int = dst_block_size_factor // min_block_size_factor

        self.block_size_in_bytes = [
            tensor.element_size() * tensor.stride(0) * min_block_size_factor
            for tensor in src_tensors
        ]

        assert len(src_tensors) > 0
        self.gpu_to_cpu: bool = self.src_tensors[0].is_cuda

        # job_id -> event
        self._transfer_events: dict[int, torch.Event] = {}
        self._transfer_start_events: dict[int, torch.Event] = {}
        # queue of transfers (job_id, stream, start_event, end_event)
        self._transfers: deque[
            tuple[int, torch.cuda.Stream, torch.Event, torch.Event]
        ] = deque()
        # list of CUDA streams available for re-use
        self._stream_pool: list[torch.cuda.Stream] = []
        # list of CUDA events available for re-use
        self._event_pool: list[torch.Event] = []

    def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
        src_spec, dst_spec = transfer_spec
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        src_sub_block_count = src_blocks.size * self.src_block_size_factor
        dst_sub_block_count = dst_blocks.size * self.dst_block_size_factor
        src_sub_blocks_to_skip = -dst_blocks.size % self.src_block_size_factor

        assert dst_sub_block_count == src_sub_block_count - src_sub_blocks_to_skip

        src_to_dst = np.empty((dst_sub_block_count, 2), dtype=np.int64)
        expand_block_ids(
            src_blocks,
            self.src_block_size_factor,
            src_to_dst[:, 0],
            skip_count=src_sub_blocks_to_skip,
        )
        expand_block_ids(dst_blocks, self.dst_block_size_factor, src_to_dst[:, 1])
        src_to_dst_tensor = torch.from_numpy(src_to_dst)

        stream = self._stream_pool.pop() if self._stream_pool else torch.cuda.Stream()
        start_event = torch.Event(enable_timing=True)
        event = self._event_pool.pop() if self._event_pool else torch.Event(
            enable_timing=True
        )

        if self.gpu_to_cpu:
            # wait for model computation to finish before offloading
            stream.wait_stream(torch.cuda.current_stream())
        if self._transfers:
            _, _, _, last_event = self._transfers[-1]
            # assure job will start only after the previous one completes
            stream.wait_event(last_event)
        with torch.cuda.stream(stream):
            start_event.record(stream)
            for src_tensor, dst_tensor, block_size_in_bytes in zip(
                self.src_tensors,
                self.dst_tensors,
                self.block_size_in_bytes,
            ):
                ops.swap_blocks(
                    src_tensor,
                    dst_tensor,
                    block_size_in_bytes,
                    src_to_dst_tensor,
                )
            event.record(stream)

        self._transfer_start_events[job_id] = start_event
        self._transfer_events[job_id] = event
        self._transfers.append((job_id, stream, start_event, event))

        # success
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        while self._transfers and self._transfers[0][3].query():
            job_id, stream, start_event, event = self._transfers.popleft()
            transfer_ms = start_event.elapsed_time(event)
            results.append((job_id, True, transfer_ms))
            self._stream_pool.append(stream)
            self._event_pool.append(event)
            del self._transfer_start_events[job_id]
            del self._transfer_events[job_id]
        return results

    def wait(self, job_ids: set[int]):
        for job_id in job_ids:
            event = self._transfer_events.get(job_id)
            if event is not None:
                event.synchronize()


class CpuGpuOffloadingHandlers:
    def __init__(
        self,
        gpu_block_size: int,
        cpu_block_size: int,
        num_cpu_blocks: int,
        gpu_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
        persist_path: str | None = None,
        reset_persisted: bool = False,
        persist_config_hash: str = "",
    ):
        assert gpu_caches
        assert cpu_block_size % gpu_block_size == 0

        # find kernel block size and determine layout per each gpu tensor
        kernel_block_size: int | None = None
        # list of (gpu_tensor, split_k_and_v)
        parsed_gpu_tensors: list[tuple[torch.Tensor, bool]] = []
        for layer_name, gpu_tensor in gpu_caches.items():
            gpu_shape = gpu_tensor.shape
            attn_backend = attn_backends[layer_name]
            test_shape = attn_backend.get_kv_cache_shape(
                num_blocks=1234, block_size=16, num_kv_heads=8, head_size=256
            )

            has_layers_dim = False
            split_k_and_v = False
            if len(gpu_shape) != len(test_shape):
                # cross-layers tensor
                # shape is (num_blocks, ...)
                assert len(gpu_shape) == len(test_shape) + 1
                has_layers_dim = True
                # prepend a dummy num_layers=80 to test_shape
                test_shape = (80,) + test_shape
            elif test_shape[0] != 1234:
                # shape should be (2, num_blocks, ...)
                assert test_shape[0] == 2
                assert test_shape[1] == 1234
                assert gpu_shape[0] == 2
                split_k_and_v = True

            try:
                kv_cache_stride_order = attn_backend.get_kv_cache_stride_order(
                    include_num_layers_dimension=has_layers_dim
                )
                assert len(kv_cache_stride_order) == len(gpu_shape)
            except (AttributeError, NotImplementedError):
                kv_cache_stride_order = tuple(range(len(gpu_shape)))

            # permute test_shape according to stride_order
            test_shape = tuple(test_shape[i] for i in kv_cache_stride_order)

            # find block_size (16) dimension index
            block_size_idx = test_shape.index(16)
            if kernel_block_size is not None:
                assert kernel_block_size == gpu_shape[block_size_idx]
            else:
                kernel_block_size = gpu_shape[block_size_idx]
                assert gpu_block_size % kernel_block_size == 0

            parsed_gpu_tensors.append((gpu_tensor, split_k_and_v))

        assert kernel_block_size is not None
        cpu_block_size_factor = cpu_block_size // kernel_block_size
        gpu_block_size_factor = gpu_block_size // kernel_block_size
        num_cpu_kernel_blocks = num_cpu_blocks * cpu_block_size_factor

        # allocate cpu tensors
        pin_memory = is_pin_memory_available()
        persist_root = Path(persist_path) if persist_path else None
        if persist_root:
            persist_root.mkdir(parents=True, exist_ok=True)
            pin_memory = False
        logger.info("Allocating %d CPU tensors...", len(parsed_gpu_tensors))
        gpu_tensors: list[torch.Tensor] = []
        cpu_tensors: list[torch.Tensor] = []
        tensor_manifest: list[dict[str, Any]] = []
        cpu_specs: list[tuple[torch.Tensor, bool, list[int]]] = []
        for tensor_index, (gpu_tensor, split_k_and_v) in enumerate(parsed_gpu_tensors):
            cpu_shape = list(gpu_tensor.shape)
            cpu_shape[1 if split_k_and_v else 0] = num_cpu_kernel_blocks
            cpu_specs.append((gpu_tensor, split_k_and_v, cpu_shape))
            if persist_root:
                tensor_manifest.append(
                    {
                        "index": tensor_index,
                        "file": f"tensor_{tensor_index}.bin",
                        "shape": cpu_shape,
                        "dtype": str(gpu_tensor.dtype),
                    }
                )

        persist_manifest = {
            "version": 1,
            "config_hash": persist_config_hash,
            "num_cpu_blocks": num_cpu_blocks,
            "gpu_block_size": gpu_block_size,
            "cpu_block_size": cpu_block_size,
            "tensors": tensor_manifest,
        }
        if persist_root and not _manifest_matches(
            persist_root / "tensors.json",
            persist_manifest,
        ):
            reset_persisted = True

        for tensor_index, (gpu_tensor, split_k_and_v, cpu_shape) in enumerate(
            cpu_specs
        ):
            logger.debug("Allocating CPU tensor of shape %r", cpu_shape)
            cpu_tensor = _allocate_cpu_tensor(
                shape=cpu_shape,
                dtype=gpu_tensor.dtype,
                pin_memory=pin_memory,
                persist_root=persist_root,
                tensor_index=tensor_index,
                reset_persisted=reset_persisted,
            )

            gpu_tensors.extend(gpu_tensor.unbind(0) if split_k_and_v else [gpu_tensor])
            cpu_tensors.extend(cpu_tensor.unbind(0) if split_k_and_v else [cpu_tensor])
        if persist_root:
            (persist_root / "tensors.json").write_text(
                json.dumps(persist_manifest, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        self.gpu_to_cpu_handler = SingleDirectionOffloadingHandler(
            src_tensors=gpu_tensors,
            dst_tensors=cpu_tensors,
            src_block_size_factor=gpu_block_size_factor,
            dst_block_size_factor=cpu_block_size_factor,
        )

        self.cpu_to_gpu_handler = SingleDirectionOffloadingHandler(
            src_tensors=cpu_tensors,
            dst_tensors=gpu_tensors,
            src_block_size_factor=cpu_block_size_factor,
            dst_block_size_factor=gpu_block_size_factor,
        )


def _allocate_cpu_tensor(
    shape: list[int],
    dtype: torch.dtype,
    pin_memory: bool,
    persist_root: Path | None,
    tensor_index: int,
    reset_persisted: bool,
) -> torch.Tensor:
    if persist_root is None:
        return torch.zeros(
            shape,
            dtype=dtype,
            device="cpu",
            pin_memory=pin_memory,
        )

    tensor_path = persist_root / f"tensor_{tensor_index}.bin"
    numel = int(np.prod(shape))
    element_size = torch.empty((), dtype=dtype).element_size()
    expected_bytes = numel * element_size
    needs_init = (
        reset_persisted
        or not tensor_path.exists()
        or tensor_path.stat().st_size != expected_bytes
    )
    if needs_init:
        with tensor_path.open("wb") as f:
            f.truncate(expected_bytes)

    tensor = torch.from_file(
        str(tensor_path),
        shared=True,
        size=numel,
        dtype=dtype,
    ).reshape(shape)
    return tensor


def _manifest_matches(path: Path, expected: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return actual == expected
