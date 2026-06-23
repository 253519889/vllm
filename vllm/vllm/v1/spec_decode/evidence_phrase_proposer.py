# SPDX-License-Identifier: Apache-2.0
"""EvidencePhrase speculative proposer.

This proposer has no draft model. It looks up pre-tokenized phrase candidates
from a server-side phrase index and returns token ids for target verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch

from vllm.config import VllmConfig
from vllm.scorephrase import FsmSpanDraftState, build_evidence_phrase_state
from vllm.scorephrase.fsm_span_drafter import FsmSpanDraftSegment
from vllm.scorephrase.phrase_index import (
    PhraseCandidate,
    PhraseConditions,
    get_phrase_index,
)
from vllm.v1.worker.gpu_input_batch import InputBatch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_input_batch import CachedRequestState


@dataclass
class PendingDraft:
    phrase_id: str
    stage: str
    token_ids: list[int]
    proposed_tokens: int


@dataclass
class EvidencePhraseRuntime:
    state: Mapping[str, Any]
    tracker: FsmSpanDraftState
    max_spec_tokens: int
    used_phrase_ids: set[str] = field(default_factory=set)
    pending: PendingDraft | None = None
    stage_index: int = 0
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    zero_accept_count: int = 0
    fallback_count: int = 0
    disabled_reason: str = ""
    proposed_by_stage: dict[str, int] = field(default_factory=dict)
    accepted_by_stage: dict[str, int] = field(default_factory=dict)
    proposed_by_phrase_id: dict[str, int] = field(default_factory=dict)
    accepted_by_phrase_id: dict[str, int] = field(default_factory=dict)

    @property
    def enabled_paths(self) -> tuple[str, ...]:
        return tuple(str(path) for path in self.state.get("enabled_paths", ()))

    @property
    def enabled_stages(self) -> tuple[str, ...]:
        return tuple(str(stage) for stage in self.state.get("enabled_stages", ()))

    def observe(self, token_ids: Sequence[int]) -> None:
        if self.disabled_reason:
            return
        self.tracker.observe_token_ids(token_ids)
        if self.pending is None:
            return

        accepted = 0
        remaining = list(self.pending.token_ids)
        for raw_token_id in token_ids:
            if not remaining:
                break
            token_id = int(raw_token_id)
            if token_id != remaining[0]:
                break
            accepted += 1
            del remaining[0]

        self.accepted_tokens += accepted
        self.accepted_by_stage[self.pending.stage] = (
            self.accepted_by_stage.get(self.pending.stage, 0) + accepted
        )
        self.accepted_by_phrase_id[self.pending.phrase_id] = (
            self.accepted_by_phrase_id.get(self.pending.phrase_id, 0) + accepted
        )
        if accepted == 0:
            self.zero_accept_count += 1
        else:
            self._advance_stage(self.pending.stage)
        self.pending = None

    def current_json_path(self) -> str:
        if self.disabled_reason or self.tracker.completed:
            return ""
        if not 0 <= self.tracker.segment_index < len(self.tracker.segments):
            return ""
        segment = self.tracker.segments[self.tracker.segment_index]
        if not _is_model_span(segment):
            return ""
        return segment.json_path

    def current_stage(self) -> str:
        stages = self.enabled_stages
        if not stages:
            return ""
        index = min(self.stage_index, len(stages) - 1)
        return stages[index]

    def record_proposal(self, candidate: PhraseCandidate, draft: list[int]) -> None:
        self.used_phrase_ids.add(candidate.phrase_id)
        self.pending = PendingDraft(
            phrase_id=candidate.phrase_id,
            stage=candidate.stage,
            token_ids=list(draft),
            proposed_tokens=len(draft),
        )
        self.proposed_tokens += len(draft)
        self.proposed_by_stage[candidate.stage] = (
            self.proposed_by_stage.get(candidate.stage, 0) + len(draft)
        )
        self.proposed_by_phrase_id[candidate.phrase_id] = (
            self.proposed_by_phrase_id.get(candidate.phrase_id, 0) + len(draft)
        )

    def metrics(self) -> dict[str, Any]:
        acceptance_rate = (
            self.accepted_tokens / self.proposed_tokens
            if self.proposed_tokens
            else 0.0
        )
        acceptance_by_stage: dict[str, float] = {}
        for stage, proposed in self.proposed_by_stage.items():
            acceptance_by_stage[stage] = (
                self.accepted_by_stage.get(stage, 0) / proposed if proposed else 0.0
            )
        acceptance_by_phrase_id: dict[str, dict[str, int]] = {}
        for phrase_id, proposed in self.proposed_by_phrase_id.items():
            acceptance_by_phrase_id[phrase_id] = {
                "proposed_tokens": proposed,
                "accepted_tokens": self.accepted_by_phrase_id.get(phrase_id, 0),
            }
        return {
            "evidence_phrase_enabled": True,
            "evidence_phrase_proposed_tokens": self.proposed_tokens,
            "evidence_phrase_accepted_tokens": self.accepted_tokens,
            "evidence_phrase_acceptance_rate": acceptance_rate,
            "evidence_phrase_zero_accept_count": self.zero_accept_count,
            "evidence_phrase_fallback_count": self.fallback_count,
            "evidence_phrase_disabled": bool(self.disabled_reason),
            "evidence_phrase_disabled_reason": self.disabled_reason,
            "summary_proposed_tokens": self.proposed_tokens,
            "summary_accepted_tokens": self.accepted_tokens,
            "summary_token_acceptance_rate": acceptance_rate,
            "summary_acceptance_by_stage": acceptance_by_stage,
            "summary_acceptance_by_phrase_id": acceptance_by_phrase_id,
        }

    def disable(self, reason: str) -> None:
        if not self.disabled_reason:
            self.fallback_count += 1
            self.disabled_reason = reason

    def _advance_stage(self, accepted_stage: str) -> None:
        stages = self.enabled_stages
        if not stages:
            return
        try:
            accepted_index = stages.index(accepted_stage)
        except ValueError:
            accepted_index = self.stage_index
        self.stage_index = min(max(self.stage_index, accepted_index + 1), len(stages) - 1)


class EvidencePhraseProposer:
    """Drafts target-mined phrase token ids for open Config C text fields."""

    def __init__(self, vllm_config: VllmConfig):
        assert vllm_config.speculative_config is not None
        self.k = vllm_config.speculative_config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.default_config = (
            vllm_config.speculative_config.evidence_phrase_config or {}
        )
        self.runtimes: dict[str, EvidencePhraseRuntime] = {}

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
            req_id = input_batch.req_ids[i]
            request = requests.get(req_id)
            if request is None or not sampled_ids:
                draft_token_ids.append([])
                continue

            runtime = self._get_runtime(req_id, request)
            if runtime is None:
                draft_token_ids.append([])
                continue

            runtime.observe(sampled_ids)
            num_tokens = input_batch.num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                draft_token_ids.append([])
                continue

            draft = self._propose_one(runtime, self.max_model_len - num_tokens)
            draft_token_ids.append(draft)

        for req_id in tuple(self.runtimes):
            if req_id not in active_req_ids and req_id not in requests:
                self.runtimes.pop(req_id, None)

        return draft_token_ids

    def _get_runtime(
        self, req_id: str, request: "CachedRequestState"
    ) -> EvidencePhraseRuntime | None:
        runtime = self.runtimes.get(req_id)
        if runtime is not None:
            return runtime

        sampling_params = request.sampling_params
        extra_args = sampling_params.extra_args if sampling_params is not None else None
        state_payload = (
            extra_args.get("evidence_phrase_state")
            if isinstance(extra_args, Mapping)
            else None
        )
        try:
            state = build_evidence_phrase_state(state_payload, self.default_config)
        except ValueError:
            return None
        if state is None:
            return None

        sand_fsm_state = (
            extra_args.get("sand_fsm_state") if isinstance(extra_args, Mapping) else None
        )
        tracker = FsmSpanDraftState.from_state(sand_fsm_state)
        if tracker is None:
            return None
        runtime = EvidencePhraseRuntime(
            state=state,
            tracker=tracker,
            max_spec_tokens=self.k,
        )
        self.runtimes[req_id] = runtime
        return runtime

    def _propose_one(
        self, runtime: EvidencePhraseRuntime, remaining_model_tokens: int
    ) -> list[int]:
        if runtime.disabled_reason or runtime.pending is not None:
            return []
        json_path = runtime.current_json_path()
        if not json_path or json_path not in runtime.enabled_paths:
            return []

        try:
            index = get_phrase_index(str(runtime.state["phrase_db"]))
        except OSError as exc:
            runtime.disable(f"phrase_db load failed: {exc}")
            return []
        except ValueError as exc:
            runtime.disable(str(exc))
            return []

        conditions = PhraseConditions(
            json_path=json_path,
            enabled_stages=runtime.enabled_stages,
            stage=runtime.current_stage(),
            risk_pattern=str(runtime.state.get("risk_pattern") or ""),
            dominant_dimensions=tuple(runtime.state.get("dominant_dimensions") or ()),
            levels=runtime.state.get("levels") or {},
            evidence_balance=str(runtime.state.get("evidence_balance") or ""),
            clinical_interpretation=str(
                runtime.state.get("clinical_interpretation") or ""
            ),
            evidence_groups=frozenset(runtime.state.get("evidence_groups") or ()),
            profile_ids=frozenset(runtime.state.get("profile_ids") or ()),
            min_acceptance_ema=float(runtime.state.get("min_acceptance_ema") or 0.0),
            used_phrase_ids=frozenset(runtime.used_phrase_ids),
        )
        candidates = index.query(conditions)
        if not candidates:
            return []

        candidate = candidates[0]
        gamma = self._gamma(runtime, candidate, remaining_model_tokens)
        if gamma <= 0:
            return []
        draft = list(candidate.token_ids[:gamma])
        runtime.record_proposal(candidate, draft)
        return draft

    def _gamma(
        self,
        runtime: EvidencePhraseRuntime,
        candidate: PhraseCandidate,
        remaining_model_tokens: int,
    ) -> int:
        request_limit = int(runtime.state.get("max_draft_tokens") or self.k)
        gamma = min(self.k, request_limit, candidate.token_count, remaining_model_tokens)
        if not runtime.state.get("dynamic_gamma"):
            return gamma

        estimate = max(candidate.acceptance_rate_ema, candidate.offline_acceptance_estimate)
        if estimate >= 0.80:
            return min(gamma, 16)
        if estimate >= 0.65:
            return min(gamma, 12)
        if estimate >= 0.50:
            return min(gamma, 8)
        return 0

    def get_metrics(self) -> dict[str, dict[str, Any]]:
        return {req_id: runtime.metrics() for req_id, runtime in self.runtimes.items()}

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass


def _is_model_span(segment: FsmSpanDraftSegment) -> bool:
    return segment.kind == "model_span"

