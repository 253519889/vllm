# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Iterable
from pathlib import Path

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    PrepareStoreOutput,
)
from vllm.v1.kv_offload.backend import Backend, BlockStatus


class LRUOffloadingManager(OffloadingManager):
    """
    An OffloadingManager with a pluggable backend, which evicts blocks by LRU.
    """

    def __init__(
        self,
        backend: Backend,
        enable_events: bool = False,
        persist_path: str | None = None,
        reset_persisted: bool = False,
        persist_config_hash: str = "",
    ):
        self.backend: Backend = backend
        # block_hash -> BlockStatus
        self.blocks: OrderedDict[BlockHash, BlockStatus] = OrderedDict()
        self.events: list[OffloadingEvent] | None = [] if enable_events else None
        self.owner_pins: defaultdict[str, set[BlockHash]] = defaultdict(set)
        self.block_pin_count: Counter[BlockHash] = Counter()
        self.persist_path = Path(persist_path) if persist_path else None
        self.persist_config_hash = persist_config_hash
        if reset_persisted and self.persist_path and self.persist_path.exists():
            self.persist_path.unlink()
        self._restore_state()

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:
        hit_count = 0
        for block_hash in block_hashes:
            block = self.blocks.get(block_hash)
            if block is None or not block.is_ready:
                break
            hit_count += 1
        return hit_count

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        blocks = []
        for block_hash in block_hashes:
            block = self.blocks[block_hash]
            assert block.is_ready
            block.ref_cnt += 1
            blocks.append(block)

        return self.backend.get_load_store_spec(block_hashes, blocks)

    def touch(self, block_hashes: Iterable[BlockHash]):
        for block_hash in reversed(list(block_hashes)):
            if self.blocks.get(block_hash):
                self.blocks.move_to_end(block_hash)

    def complete_load(self, block_hashes: Iterable[BlockHash]):
        for block_hash in block_hashes:
            block = self.blocks[block_hash]
            assert block.ref_cnt > 0
            block.ref_cnt -= 1

    def pin(self, owner_id: str, block_hashes: Iterable[BlockHash]):
        if not owner_id:
            return
        owner_blocks = self.owner_pins[owner_id]
        for block_hash in block_hashes:
            if block_hash not in self.blocks or block_hash in owner_blocks:
                continue
            owner_blocks.add(block_hash)
            self.block_pin_count[block_hash] += 1
        self._persist_state()

    def unpin(self, owner_id: str):
        for block_hash in self.owner_pins.pop(owner_id, set()):
            count = self.block_pin_count.get(block_hash, 0)
            if count <= 1:
                self.block_pin_count.pop(block_hash, None)
            else:
                self.block_pin_count[block_hash] = count - 1
        self._persist_state()

    def _is_pinned(self, block_hash: BlockHash) -> bool:
        return self.block_pin_count.get(block_hash, 0) > 0

    def _drop_pin_state(self, block_hash: BlockHash):
        self.block_pin_count.pop(block_hash, None)
        for owner_id, owner_blocks in list(self.owner_pins.items()):
            owner_blocks.discard(block_hash)
            if not owner_blocks:
                self.owner_pins.pop(owner_id, None)

    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        # filter out blocks that are already stored
        block_hashes_to_store = [
            block_hash for block_hash in block_hashes if block_hash not in self.blocks
        ]

        num_blocks_to_evict = (
            len(block_hashes_to_store) - self.backend.get_num_free_blocks()
        )

        # build list of blocks to evict
        to_evict = []
        if num_blocks_to_evict > 0:
            for block_hash, block in self.blocks.items():
                if block.ref_cnt == 0 and not self._is_pinned(block_hash):
                    to_evict.append(block_hash)
                    num_blocks_to_evict -= 1
                    if num_blocks_to_evict == 0:
                        break
            else:
                # we could not evict enough blocks
                return None

        # evict blocks
        for block_hash in to_evict:
            self.backend.free(self.blocks.pop(block_hash))

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=to_evict,
                    block_size=self.backend.block_size,
                    medium=self.backend.medium,
                    removed=True,
                )
            )

        blocks = self.backend.allocate_blocks(block_hashes_to_store)
        assert len(blocks) == len(block_hashes_to_store)

        for block_hash, block in zip(block_hashes_to_store, blocks):
            self.blocks[block_hash] = block

        # build store specs for allocated blocks
        store_spec = self.backend.get_load_store_spec(block_hashes_to_store, blocks)

        return PrepareStoreOutput(
            block_hashes_to_store=block_hashes_to_store,
            store_spec=store_spec,
            block_hashes_evicted=to_evict,
        )

    def complete_store(self, block_hashes: Iterable[BlockHash], success: bool = True):
        stored_block_hashes: list[BlockHash] = []
        if success:
            for block_hash in block_hashes:
                block = self.blocks[block_hash]
                if not block.is_ready:
                    block.ref_cnt = 0
                    stored_block_hashes.append(block_hash)
        else:
            for block_hash in block_hashes:
                block = self.blocks[block_hash]
                if not block.is_ready:
                    self.backend.free(block)
                    del self.blocks[block_hash]
                    self._drop_pin_state(block_hash)

        if stored_block_hashes and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=stored_block_hashes,
                    block_size=self.backend.block_size,
                    medium=self.backend.medium,
                    removed=False,
                )
            )
        self._persist_state()

    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    def _restore_state(self):
        if not self.persist_path or not self.persist_path.exists():
            return
        try:
            payload = json.loads(self.persist_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        records = payload.get("blocks", [])
        if not isinstance(records, list):
            return
        if (
            self.persist_config_hash
            and payload.get("config_hash") != self.persist_config_hash
        ):
            return
        try:
            persisted_block_size = int(payload.get("block_size", -1))
            persisted_num_blocks = int(payload.get("num_blocks", -1))
        except (TypeError, ValueError):
            return
        if persisted_block_size != self.backend.block_size:
            return
        if persisted_num_blocks != getattr(self.backend, "num_blocks", -2):
            return
        block_ids = [
            int(record["block_id"])
            for record in records
            if isinstance(record, dict) and "block_id" in record
        ]
        restore_blocks = getattr(self.backend, "restore_blocks", None)
        if not callable(restore_blocks):
            return
        statuses = restore_blocks(block_ids)
        for record, status in zip(records, statuses):
            block_hash_hex = record.get("block_hash") if isinstance(record, dict) else ""
            if not isinstance(block_hash_hex, str):
                continue
            try:
                block_hash = BlockHash(bytes.fromhex(block_hash_hex))
            except ValueError:
                continue
            self.blocks[block_hash] = status

        owner_pins = payload.get("owner_pins", {})
        if isinstance(owner_pins, dict):
            for owner_id, block_hashes in owner_pins.items():
                if not isinstance(owner_id, str) or not isinstance(block_hashes, list):
                    continue
                for block_hash_hex in block_hashes:
                    if not isinstance(block_hash_hex, str):
                        continue
                    try:
                        block_hash = BlockHash(bytes.fromhex(block_hash_hex))
                    except ValueError:
                        continue
                    if block_hash not in self.blocks:
                        continue
                    self.owner_pins[owner_id].add(block_hash)
                    self.block_pin_count[block_hash] += 1

    def _persist_state(self):
        if not self.persist_path:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        owner_pins = {
            owner_id: sorted(
                block_hash.hex()
                for block_hash in block_hashes
                if block_hash in self.blocks and self.blocks[block_hash].is_ready
            )
            for owner_id, block_hashes in self.owner_pins.items()
        }
        payload = {
            "version": 1,
            "policy": "lru",
            "config_hash": self.persist_config_hash,
            "block_size": self.backend.block_size,
            "num_blocks": getattr(self.backend, "num_blocks", None),
            "blocks": [
                {
                    "block_hash": block_hash.hex(),
                    "block_id": int(getattr(block, "block_id")),
                }
                for block_hash, block in self.blocks.items()
                if block.is_ready and hasattr(block, "block_id")
            ],
            "owner_pins": owner_pins,
        }
        tmp_path = self.persist_path.with_suffix(self.persist_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        tmp_path.replace(self.persist_path)
