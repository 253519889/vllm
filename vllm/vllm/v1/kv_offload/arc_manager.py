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


class ARCOffloadingManager(OffloadingManager):
    """
    An OffloadingManager implementing the ARC (Adaptive Replacement Cache)
    eviction policy with a pluggable backend.

    Data Structures:
        T1: Recent cache containing blocks accessed once.
        T2: Frequent cache containing blocks accessed multiple times.
        B1/B2: Ghost lists tracking recently evicted blocks from T1/T2.
        target_t1_size: Adaptive target size for the T1 partition.

    Algorithm Flow:
        1. Cache lookup (lookup):
           Searches T1 and T2 for block hashes and counts consecutive hits
           until a miss or non-ready block is encountered.

        2. Cache touch (touch) - Adaptive Learning:
           For each block_hash (in reverse order):
           - If in T1: Move to T2 (promotion from recent to frequent).
           - If in T2: Move to MRU position (end of queue).
           - If in B1 ghost list: Increase target_t1_size.
           - If in B2 ghost list: Decrease target_t1_size.

        3. Block eviction (prepare_store) - Adaptive Replacement:
           Determines eviction source based on adaptive target:
           - If T1 size > target_t1_size: Evict from T1, add to B1.
           - Otherwise: Evict from T2, add to B2.
           Finally, bound each ghost list size.

        4. Block insertion (prepare_store):
           New blocks are always inserted into T1 and removed from B1/B2 if
           present. Blocks may later be promoted to T2 during touch operations.

    Adaptive Behavior:
        The algorithm self-tunes the recency vs. frequency trade-off:
        - B1 hit: Recent access patterns matter more → increase T1.
        - B2 hit: Frequent access patterns matter more → decrease T1.
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
        self.target_t1_size: float = 0.0
        self.t1: OrderedDict[BlockHash, BlockStatus] = OrderedDict()
        self.t2: OrderedDict[BlockHash, BlockStatus] = OrderedDict()
        # block_hash -> None (only care about presence)
        self.b1: OrderedDict[BlockHash, None] = OrderedDict()
        self.b2: OrderedDict[BlockHash, None] = OrderedDict()
        self.events: list[OffloadingEvent] | None = [] if enable_events else None
        self.cache_capacity: int = self.backend.get_num_free_blocks()
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
            block = self.t1.get(block_hash) or self.t2.get(block_hash)
            if block is None or not block.is_ready:
                break
            hit_count += 1
        return hit_count

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        blocks = []
        for block_hash in block_hashes:
            block = self.t1.get(block_hash) or self.t2.get(block_hash)
            assert block is not None, f"Block {block_hash!r} not found in cache"
            assert block.is_ready, f"Block {block_hash!r} is not ready for reading"

            block.ref_cnt += 1
            blocks.append(block)

        return self.backend.get_load_store_spec(block_hashes, blocks)

    def touch(self, block_hashes: Iterable[BlockHash]):
        for block_hash in reversed(list(block_hashes)):
            if block_hash in self.t1:
                block = self.t1.pop(block_hash)
                if not block.is_ready:
                    # block was just prepared to be stored, not really touched twice
                    self.t1.move_to_end(block_hash)
                else:
                    self.t2[block_hash] = block

            elif block_hash in self.t2:
                self.t2.move_to_end(block_hash)

            elif block_hash in self.b1:
                delta = max(1, len(self.b2) / len(self.b1))
                self.target_t1_size = min(
                    self.target_t1_size + delta, self.cache_capacity
                )
                # move to MRU position (end) to keep it fresh in the ghost list
                self.b1.move_to_end(block_hash)

            elif block_hash in self.b2:
                delta = max(1, len(self.b1) / len(self.b2))
                self.target_t1_size = max(self.target_t1_size - delta, 0)
                # move to MRU position (end) to keep it fresh in the ghost list
                self.b2.move_to_end(block_hash)

    def complete_load(self, block_hashes: Iterable[BlockHash]):
        for block_hash in block_hashes:
            block = self.t1.get(block_hash) or self.t2.get(block_hash)
            assert block is not None, f"Block {block_hash!r} not found"
            assert block.ref_cnt > 0, f"Block {block_hash!r} ref_cnt is already 0"

            block.ref_cnt -= 1

    def pin(self, owner_id: str, block_hashes: Iterable[BlockHash]):
        if not owner_id:
            return
        owner_blocks = self.owner_pins[owner_id]
        for block_hash in block_hashes:
            block = self.t1.get(block_hash) or self.t2.get(block_hash)
            if block is None or block_hash in owner_blocks:
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
        block_hashes_to_store = []
        for block_hash in block_hashes:
            if block_hash not in self.t1 and block_hash not in self.t2:
                block_hashes_to_store.append(block_hash)

        if not block_hashes_to_store:
            return PrepareStoreOutput(
                block_hashes_to_store=[],
                store_spec=self.backend.get_load_store_spec([], []),
                block_hashes_evicted=[],
            )

        num_blocks_to_evict = (
            len(block_hashes_to_store) - self.backend.get_num_free_blocks()
        )

        to_evict = []
        while num_blocks_to_evict > 0:
            block_to_evict = None
            if len(self.t1) >= int(self.target_t1_size):
                # try to evict the least recently used (oldest) block from T1
                for block_hash, block in self.t1.items():
                    if block.ref_cnt == 0 and not self._is_pinned(block_hash):
                        block_to_evict = (block_hash, block)
                        eviction_t = self.t1
                        eviction_b = self.b1
                        break
            if not block_to_evict:
                # try to evict the least recently used (oldest) block from T2
                for block_hash, block in self.t2.items():
                    if block.ref_cnt == 0 and not self._is_pinned(block_hash):
                        block_to_evict = (block_hash, block)
                        eviction_t = self.t2
                        eviction_b = self.b2
                        break
                else:
                    # cannot evict enough blocks, cache is full of in-use items
                    return None

            block_hash, block = block_to_evict
            del eviction_t[block_hash]
            eviction_b[block_hash] = None
            to_evict.append(block_hash)
            self.backend.free(block)
            num_blocks_to_evict -= 1

        for b in [self.b1, self.b2]:
            for i in range(len(b) - self.cache_capacity):
                b.popitem(last=False)

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
        assert len(blocks) == len(block_hashes_to_store), (
            "Backend did not allocate the expected number of blocks"
        )

        for block_hash, block in zip(block_hashes_to_store, blocks):
            self.t1[block_hash] = block

            self.b1.pop(block_hash, None)
            self.b2.pop(block_hash, None)

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
                block = self.t1.get(block_hash) or self.t2.get(block_hash)

                if block is not None and not block.is_ready:
                    block.ref_cnt = 0
                    stored_block_hashes.append(block_hash)
        else:
            for block_hash in block_hashes:
                block = self.t1.pop(block_hash, None)

                if block is None:
                    block = self.t2.pop(block_hash, None)

                if block is not None and not block.is_ready:
                    self.backend.free(block)
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
            self.t2[block_hash] = status

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
                    if block_hash not in self.t1 and block_hash not in self.t2:
                        continue
                    self.owner_pins[owner_id].add(block_hash)
                    self.block_pin_count[block_hash] += 1

    def _persist_state(self):
        if not self.persist_path:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        ready_blocks = [
            (block_hash, block)
            for block_hash, block in [*self.t1.items(), *self.t2.items()]
            if block.is_ready and hasattr(block, "block_id")
        ]
        ready_hashes = {block_hash for block_hash, _ in ready_blocks}
        owner_pins = {
            owner_id: sorted(
                block_hash.hex()
                for block_hash in block_hashes
                if block_hash in ready_hashes
            )
            for owner_id, block_hashes in self.owner_pins.items()
        }
        payload = {
            "version": 1,
            "policy": "arc",
            "config_hash": self.persist_config_hash,
            "block_size": self.backend.block_size,
            "num_blocks": getattr(self.backend, "num_blocks", None),
            "blocks": [
                {
                    "block_hash": block_hash.hex(),
                    "block_id": int(getattr(block, "block_id")),
                }
                for block_hash, block in ready_blocks
            ],
            "owner_pins": owner_pins,
        }
        tmp_path = self.persist_path.with_suffix(self.persist_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        tmp_path.replace(self.persist_path)
