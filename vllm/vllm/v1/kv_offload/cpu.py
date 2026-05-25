# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.arc_manager import ARCOffloadingManager
from vllm.v1.kv_offload.backends.cpu import CPUBackend
from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.cpu_gpu import CpuGpuOffloadingHandlers
from vllm.v1.kv_offload.worker.worker import OffloadingHandler


class CPUOffloadingSpec(OffloadingSpec):
    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise Exception(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )

        # calculate kv_bytes_per_offloaded_block
        assert kv_cache_config is not None
        page_sizes = {
            kv_cache_group.kv_cache_spec.page_size_bytes
            for kv_cache_group in kv_cache_config.kv_cache_groups
        }
        assert len(page_sizes) == 1
        page_size_bytes = page_sizes.pop()
        kv_bytes_per_block = (
            page_size_bytes
            * len(kv_cache_config.kv_cache_tensors)
            * vllm_config.parallel_config.world_size
        )
        kv_bytes_per_offloaded_block = kv_bytes_per_block * (
            self.offloaded_block_size // self.gpu_block_size
        )

        self.num_blocks = (
            int(cpu_bytes_to_use) // kv_bytes_per_offloaded_block
            if kv_bytes_per_offloaded_block > 0
            else 0
        )

        # scheduler-side
        self._manager: OffloadingManager | None = None

        # worker-side
        self._handlers: CpuGpuOffloadingHandlers | None = None

        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")
        self.persist_root = _optional_path(self.extra_config.get("persist_dir"))
        self.persist_reset = _config_bool(self.extra_config.get("persist_reset", False))
        self.persist_config_hash = (
            self._persist_config_hash() if self.persist_root else ""
        )
        self.persist_rank_dir = (
            self.persist_root / self._persist_namespace() / f"rank_{_rank_id()}"
            if self.persist_root
            else None
        )

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None and kv_events_config.enable_kv_cache_events
            )

            backend = CPUBackend(
                block_size=self.offloaded_block_size, num_blocks=self.num_blocks
            )

            if self.eviction_policy == "lru":
                self._manager = LRUOffloadingManager(
                    backend=backend,
                    enable_events=enable_events,
                    persist_path=self._persist_metadata_path(),
                    reset_persisted=self.persist_reset,
                    persist_config_hash=self.persist_config_hash,
                )
            elif self.eviction_policy == "arc":
                self._manager = ARCOffloadingManager(
                    backend=backend,
                    enable_events=enable_events,
                    persist_path=self._persist_metadata_path(),
                    reset_persisted=self.persist_reset,
                    persist_config_hash=self.persist_config_hash,
                )
            else:
                raise ValueError(
                    f"Unknown eviction policy: {self.eviction_policy}. "
                    f"Supported policies: lru, arc"
                )
        return self._manager

    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        if not self._handlers:
            if not current_platform.is_cuda_alike():
                raise Exception(
                    "CPU Offloading is currently only supported on CUDA-alike GPUs"
                )

            self._handlers = CpuGpuOffloadingHandlers(
                attn_backends=attn_backends,
                gpu_block_size=self.gpu_block_size,
                cpu_block_size=self.offloaded_block_size,
                num_cpu_blocks=self.num_blocks,
                gpu_caches=kv_caches,
                persist_path=self._persist_tensor_path(),
                reset_persisted=self.persist_reset,
                persist_config_hash=self.persist_config_hash,
            )

        assert self._handlers is not None
        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handlers.gpu_to_cpu_handler
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handlers.cpu_to_gpu_handler

    def _persist_namespace(self) -> str:
        namespace = self.extra_config.get("persist_namespace")
        if namespace:
            return str(namespace)
        return f"model_{self.persist_config_hash[:16]}"

    def _persist_config_hash(self) -> str:
        model_config = getattr(self.vllm_config, "model_config", None)
        model_name = str(getattr(model_config, "model", "model"))
        payload = {
            "model": model_name,
            "gpu_block_size": self.gpu_block_size,
            "offloaded_block_size": self.offloaded_block_size,
            "num_blocks": self.num_blocks,
            "world_size": self.vllm_config.parallel_config.world_size,
            "kv_cache_tensors": len(self.kv_cache_config.kv_cache_tensors)
            if self.kv_cache_config
            else 0,
        }
        return hashlib.sha1(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _persist_metadata_path(self) -> str | None:
        if self.persist_rank_dir is None:
            return None
        return str(self.persist_rank_dir / "blocks.json")

    def _persist_tensor_path(self) -> str | None:
        if self.persist_rank_dir is None:
            return None
        return str(self.persist_rank_dir / "tensors")


def _optional_path(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    return Path(str(value))


def _rank_id() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return 0


def _config_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return False
