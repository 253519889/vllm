# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.scorephrase import ConfigCTokenPlanRuntime, FsmSpanDraftState
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import (
    EngineCoreEvent,
    EngineCoreEventType,
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest
    from vllm.v1.core.kv_cache_utils import BlockHash


@dataclass
class StreamingUpdate:
    """Lightweight data for streaming session continuation.

    Contains only the fields needed to update an existing streaming session
    with new input data.
    """

    mm_features: list[MultiModalFeatureSpec] | None
    prompt_token_ids: list[int] | None
    max_tokens: int
    arrival_time: float
    sampling_params: SamplingParams | None

    @classmethod
    def from_request(cls, request: "Request") -> "StreamingUpdate | None":
        if not request.resumable:
            return None
        return cls(
            mm_features=request.mm_features,
            prompt_token_ids=request.prompt_token_ids,
            max_tokens=request.max_tokens,
            arrival_time=request.arrival_time,
            sampling_params=request.sampling_params,
        )


class Request:
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        sampling_params: SamplingParams | None,
        pooling_params: PoolingParams | None,
        eos_token_id: int | None,
        client_index: int = 0,
        arrival_time: float | None = None,
        prompt_embeds: torch.Tensor | None = None,
        mm_features: list[MultiModalFeatureSpec] | None = None,
        lora_request: Optional["LoRARequest"] = None,
        cache_salt: str | None = None,
        priority: int = 0,
        trace_headers: Mapping[str, str] | None = None,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None = None,
        resumable: bool = False,
    ) -> None:
        self.request_id = request_id
        self.client_index = client_index
        self.priority = priority
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
        # Because of LoRA, the eos token id can be different for each request.
        self.eos_token_id = eos_token_id
        self.lora_request = lora_request
        self.structured_output_request = StructuredOutputRequest.from_sampling_params(
            sampling_params
        )
        self.arrival_time = arrival_time if arrival_time is not None else time.time()

        self.status = RequestStatus.WAITING
        self.events: list[EngineCoreEvent] = []
        self.stop_reason: int | str | None = None

        # P/D: Connector-specific KV transfer parameters.
        self.kv_transfer_params: dict[str, Any] | None = None
        self.kv_transfer_metrics: dict[str, Any] = {}

        if pooling_params is not None:
            # Pooling models.
            self.max_tokens = 1
        elif sampling_params is not None:
            # Generative models.
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens
            if self.structured_output_request is not None:
                self.status = RequestStatus.WAITING_FOR_FSM

            if sampling_params.extra_args is not None:
                self.kv_transfer_params = sampling_params.extra_args.get(
                    "kv_transfer_params"
                )
        else:
            raise ValueError("sampling_params and pooling_params can't both be unset")

        extra_args = sampling_params.extra_args if sampling_params is not None else None
        self.profile_cache_id = _get_profile_cache_id(extra_args)
        self.profile_cache_pin = _get_profile_cache_pin(extra_args)
        self.profile_cache_store_l2 = _get_profile_cache_store_l2(extra_args)
        self.profile_cache_reusable_tokens = _get_profile_cache_reusable_tokens(
            extra_args
        )
        self.scorephrase_state = _get_extra_arg(extra_args, "scorephrase_state")
        self.sand_fsm_state = _get_extra_arg(extra_args, "sand_fsm_state")
        self.sand_fsm_runtime = ConfigCTokenPlanRuntime.from_state(
            self.sand_fsm_state
        )
        self.fsm_span_metrics: dict[str, Any] = {}
        self.sand_fsm_force_known_spans = _get_sand_fsm_bool(
            self.sand_fsm_state, "force_known_spans", False
        )
        self.sand_fsm_max_forced_tokens_per_step = _get_sand_fsm_int(
            self.sand_fsm_state,
            "max_forced_tokens_per_step",
            default=64,
            minimum=1,
            maximum=512,
        )
        self.sand_fsm_force_runtime = (
            FsmSpanDraftState.from_state(self.sand_fsm_state)
            if self.sand_fsm_force_known_spans
            else None
        )
        self.sand_fsm_force_disabled_reason = ""
        self.sand_fsm_force_fallback_count = 0
        self.pending_forced_output_token_ids: list[int] = []

        self.prompt_token_ids = prompt_token_ids
        self.prompt_embeds = prompt_embeds
        self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            prompt_token_ids, prompt_embeds
        )
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = (
            self.prompt_token_ids.copy()
            if self.prompt_token_ids is not None
            else [0] * self.num_prompt_tokens
        )

        # Used in async scheduling.
        self.num_output_placeholders = 0
        # Used in forced preemption (reset_prefix_cache) with async scheduling.
        self.discard_latest_async_tokens = False

        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
        self.cache_salt: str | None = cache_salt

        # Multi-modal related
        self.mm_features = mm_features or []

        # Read-only views
        # Prevent directly appending to these lists since
        # they should also be updated simultaneously.
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)
        # trace_headers
        self.trace_headers = trace_headers
        # State
        # The number of tokens with prefix cache hits.
        self.num_cached_tokens = -1

        # The number of NaNs in logits. A value greater than 0
        # indicates that the output is corrupted
        self.num_nans_in_logits = 0

        # The number of times this request has been preempted by the scheduler.
        self.num_preemptions = 0

        # The number of tokens that have been computed remotely.
        self.num_external_computed_tokens = 0

        self.block_hashes: list[BlockHash] = []
        self.get_hash_new_full_blocks: Callable[[], list[BlockHash]] | None = None
        if block_hasher is not None:
            self.get_hash_new_full_blocks = partial(block_hasher, self)
            self.block_hashes = self.get_hash_new_full_blocks()

        self.skip_reading_prefix_cache = self.get_skip_reading_prefix_cache()

        # Used for streaming
        self.resumable = resumable
        # None entry in the queue means finished.
        self.streaming_queue: deque[StreamingUpdate | None] | None = None

    @classmethod
    def from_engine_core_request(
        cls,
        request: EngineCoreRequest,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None,
    ) -> "Request":
        return cls(
            request_id=request.request_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            eos_token_id=request.eos_token_id,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            priority=request.priority,
            trace_headers=request.trace_headers,
            block_hasher=block_hasher,
            resumable=request.resumable,
        )

    def append_output_token_ids(
        self,
        token_ids: int | list[int],
    ) -> None:
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)

        if self.get_hash_new_full_blocks is not None:
            self.block_hashes.extend(self.get_hash_new_full_blocks())

    def peek_sand_fsm_forced_token_ids(self, max_tokens: int) -> list[int]:
        if (
            not self.sand_fsm_force_known_spans
            or self.sand_fsm_force_runtime is None
            or self.sand_fsm_force_disabled_reason
        ):
            return []
        return self.sand_fsm_force_runtime.peek_deterministic(max_tokens)

    def commit_sand_fsm_forced_token_ids(self, token_ids: list[int]) -> list[int]:
        if (
            not token_ids
            or not self.sand_fsm_force_known_spans
            or self.sand_fsm_force_runtime is None
            or self.sand_fsm_force_disabled_reason
        ):
            return []
        committed = self.sand_fsm_force_runtime.commit_forced_token_ids(token_ids)
        if not committed:
            self.disable_sand_fsm_force("token_plan commit failed")
            return []
        self.append_output_token_ids(committed)
        self.pending_forced_output_token_ids.extend(committed)
        return committed

    def take_pending_forced_output_token_ids(self) -> list[int]:
        if not self.pending_forced_output_token_ids:
            return []
        token_ids = self.pending_forced_output_token_ids
        self.pending_forced_output_token_ids = []
        return token_ids

    def disable_sand_fsm_force(self, reason: str) -> None:
        if not self.sand_fsm_force_disabled_reason:
            self.sand_fsm_force_fallback_count += 1
        self.sand_fsm_force_disabled_reason = reason
        self.pending_forced_output_token_ids.clear()

    def observe_sand_fsm_model_token_ids(self, token_ids: list[int]) -> None:
        if (
            not token_ids
            or not self.sand_fsm_force_known_spans
            or self.sand_fsm_force_runtime is None
            or self.sand_fsm_force_disabled_reason
        ):
            return
        self.sand_fsm_force_runtime.observe_token_ids(token_ids)

    def sand_fsm_force_metrics(self) -> dict[str, Any]:
        runtime_metrics = (
            self.sand_fsm_force_runtime.metrics()
            if self.sand_fsm_force_runtime is not None
            else {}
        )
        forced_tokens = _get_int_from_mapping(runtime_metrics, "fsm_forced_tokens", 0)
        kv_advance_tokens = _get_int_from_mapping(
            runtime_metrics, "fsm_kv_advance_tokens", forced_tokens
        )
        force_fallback_count = (
            self.sand_fsm_force_fallback_count
            + _get_int_from_mapping(runtime_metrics, "fsm_force_fallback_count", 0)
            + _get_int_from_mapping(runtime_metrics, "fsm_fallback_count", 0)
        )
        if (
            not self.sand_fsm_force_known_spans
            and not forced_tokens
            and not force_fallback_count
        ):
            return {}
        return {
            "fsm_force_known_spans": self.sand_fsm_force_known_spans,
            "fsm_forced_tokens": forced_tokens,
            "fsm_kv_advance_tokens": kv_advance_tokens,
            "fsm_force_fallback_count": force_fallback_count,
            "fsm_force_disabled_reason": self.sand_fsm_force_disabled_reason
            or str(runtime_metrics.get("fsm_disabled_reason") or ""),
        }

    @property
    def use_structured_output(self) -> bool:
        return self.structured_output_request is not None

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    @property
    def num_encoder_inputs(self) -> int:
        return len(self.mm_features)

    @property
    def has_encoder_inputs(self) -> bool:
        return self.num_encoder_inputs > 0

    def get_skip_reading_prefix_cache(self) -> bool:
        if (
            self.sampling_params is not None
            and self.sampling_params.skip_reading_prefix_cache is not None
        ):
            return self.sampling_params.skip_reading_prefix_cache
        elif (
            self.pooling_params is not None
            and self.pooling_params.skip_reading_prefix_cache is not None
        ):
            return self.pooling_params.skip_reading_prefix_cache
        return False

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> FinishReason | None:
        return RequestStatus.get_finished_reason(self.status)

    def get_num_encoder_embeds(self, input_id: int) -> int:
        assert input_id < len(self.mm_features)
        return self.mm_features[input_id].mm_position.get_num_embeds

    def record_event(
        self,
        event_type: EngineCoreEventType,
        timestamp: float | None = None,
    ) -> None:
        self.events.append(EngineCoreEvent.new_event(event_type, timestamp))

    def take_events(self) -> list[EngineCoreEvent] | None:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events

    def __lt__(self, other: "Request") -> bool:
        """
        Compare two requests based on priority, arrival time, and request ID.
        Used in priority scheduling.
        """
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        if self.request_id != other.request_id:
            return self.request_id < other.request_id
        return id(self) < id(other)


def _get_profile_cache_id(extra_args: dict[str, Any] | None) -> str:
    if not extra_args:
        return ""
    value = extra_args.get("profile_cache_id")
    if value is None:
        value = extra_args.get("profile_id")
    return str(value) if value is not None else ""


def _get_profile_cache_pin(extra_args: dict[str, Any] | None) -> bool:
    if not extra_args:
        return False
    value = extra_args.get("profile_cache_pin", False)
    return _coerce_bool(value)


def _get_profile_cache_store_l2(extra_args: dict[str, Any] | None) -> bool:
    if not extra_args:
        return False
    value = extra_args.get("profile_cache_store_l2", False)
    return _coerce_bool(value)


def _get_profile_cache_reusable_tokens(extra_args: dict[str, Any] | None) -> int:
    if not extra_args:
        return 0
    value = extra_args.get("profile_cache_reusable_tokens", 0)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _get_extra_arg(extra_args: dict[str, Any] | None, key: str) -> Any | None:
    if not extra_args:
        return None
    return extra_args.get(key)


def _get_sand_fsm_bool(
    sand_fsm_state: Any | None,
    key: str,
    default: bool,
) -> bool:
    if not isinstance(sand_fsm_state, Mapping):
        return default
    if key not in sand_fsm_state:
        return default
    return _coerce_bool(sand_fsm_state.get(key))


def _get_sand_fsm_int(
    sand_fsm_state: Any | None,
    key: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    if not isinstance(sand_fsm_state, Mapping):
        return default
    try:
        value = int(sand_fsm_state.get(key, default))
    except (TypeError, ValueError):
        return default
    return min(max(value, minimum), maximum)


def _get_int_from_mapping(
    mapping: Mapping[str, Any],
    key: str,
    default: int,
) -> int:
    try:
        return int(mapping.get(key, default))
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on", "pin", "store"}
    return False


class RequestStatus(enum.IntEnum):
    """Status of a request."""

    WAITING = enum.auto()
    WAITING_FOR_FSM = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    WAITING_FOR_STREAMING_REQ = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_ERROR = enum.auto()

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> FinishReason | None:
        return _FINISHED_REASON_MAP.get(status)


# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ERROR: FinishReason.ERROR,
    RequestStatus.WAITING_FOR_STREAMING_REQ: FinishReason.STOP,
}
