# SPDX-License-Identifier: Apache-2.0
"""Speculative proposer for Config C Sand-FSM deterministic spans."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.scorephrase import FsmSpanDraftState
from vllm.v1.worker.gpu_input_batch import InputBatch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_input_batch import CachedRequestState


class FsmSpanProposer:
    """Drafts request-local deterministic token_plan spans.

    This proposer has no draft model. It only proposes tokens that were encoded
    from deterministic `sand_fsm.token_plan` segments on the engine side.
    """

    def __init__(self, vllm_config: VllmConfig):
        assert vllm_config.speculative_config is not None
        self.k = vllm_config.speculative_config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.states: dict[str, FsmSpanDraftState] = {}

    def propose(
        self,
        input_batch: InputBatch,
        sampled_token_ids: list[list[int]],
        requests: dict[str, "CachedRequestState"],
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,  # unused
    ) -> list[list[int]]:
        del slot_mappings

        draft_token_ids: list[list[int]] = []
        active_req_ids = set(input_batch.req_ids)

        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                draft_token_ids.append([])
                continue

            req_id = input_batch.req_ids[i]
            request = requests.get(req_id)
            if request is None:
                draft_token_ids.append([])
                continue

            num_tokens = input_batch.num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                draft_token_ids.append([])
                continue

            state = self._get_state(req_id, request)
            if state is None:
                draft_token_ids.append([])
                continue

            state.observe_token_ids(sampled_ids)
            max_draft_tokens = min(self.k, self.max_model_len - num_tokens)
            draft_token_ids.append(state.propose(max_draft_tokens))

        for req_id in tuple(self.states):
            if req_id not in active_req_ids and req_id not in requests:
                self.states.pop(req_id, None)

        return draft_token_ids

    def _get_state(
        self, req_id: str, request: "CachedRequestState"
    ) -> FsmSpanDraftState | None:
        state = self.states.get(req_id)
        if state is not None:
            return state

        sampling_params = request.sampling_params
        extra_args = sampling_params.extra_args if sampling_params is not None else None
        sand_fsm_state = (
            extra_args.get("sand_fsm_state") if isinstance(extra_args, dict) else None
        )
        state = FsmSpanDraftState.from_state(sand_fsm_state)
        if state is None:
            return None
        self.states[req_id] = state
        return state

    def get_metrics(self) -> dict[str, dict[str, object]]:
        return {req_id: state.metrics() for req_id, state in self.states.items()}

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass
