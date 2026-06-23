# SPDX-License-Identifier: Apache-2.0
"""Server-side phrase index for EvidencePhrase speculative drafting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class PhraseCandidate:
    phrase_id: str
    json_path: str
    stage: str
    token_ids: tuple[int, ...]
    token_count: int
    text: str
    risk_pattern: str
    dominant_dimensions: tuple[str, ...]
    levels: Mapping[str, int]
    evidence_balance: str
    clinical_interpretation: str
    evidence_groups: tuple[str, ...]
    polarity: str
    anchor_type: str
    match_scope: str
    required_profile_ids: tuple[str, ...]
    priority: float
    source: str
    offline_acceptance_estimate: float
    acceptance_rate_ema: float
    proposed_count: int
    disabled: bool
    safe: bool


@dataclass(frozen=True)
class PhraseConditions:
    json_path: str
    enabled_stages: tuple[str, ...]
    stage: str
    risk_pattern: str = ""
    dominant_dimensions: tuple[str, ...] = ()
    levels: Mapping[str, int] | None = None
    evidence_balance: str = ""
    clinical_interpretation: str = ""
    evidence_groups: frozenset[str] = frozenset()
    profile_ids: frozenset[str] = frozenset()
    min_acceptance_ema: float = 0.0
    used_phrase_ids: frozenset[str] = frozenset()


class PhraseIndex:
    """Load and query an EvidencePhrase JSONL phrase database."""

    def __init__(self, phrase_db: str):
        self.phrase_db = phrase_db
        self.candidates = tuple(self._load(Path(phrase_db).expanduser()))
        self.by_path_stage: dict[tuple[str, str], list[PhraseCandidate]] = {}
        for candidate in self.candidates:
            self.by_path_stage.setdefault(
                (candidate.json_path, candidate.stage), []
            ).append(candidate)
        for bucket in self.by_path_stage.values():
            bucket.sort(key=self._sort_key)

    def query(self, conditions: PhraseConditions) -> list[PhraseCandidate]:
        stages = (conditions.stage,) if conditions.stage else ()
        stages = stages + tuple(
            stage
            for stage in conditions.enabled_stages
            if stage not in stages and stage != "generic"
        )
        if "generic" in conditions.enabled_stages and "generic" not in stages:
            stages = stages + ("generic",)

        candidates: list[PhraseCandidate] = []
        for stage in stages:
            for candidate in self.by_path_stage.get((conditions.json_path, stage), []):
                if self._matches(candidate, conditions):
                    candidates.append(candidate)
        candidates.sort(
            key=lambda candidate: self._match_sort_key(candidate, conditions)
        )
        return candidates

    def _matches(
        self, candidate: PhraseCandidate, conditions: PhraseConditions
    ) -> bool:
        if not candidate.safe or candidate.disabled or not candidate.token_ids:
            return False
        if candidate.phrase_id in conditions.used_phrase_ids:
            return False
        if candidate.stage not in conditions.enabled_stages:
            return False
        if (
            candidate.proposed_count >= 30
            and candidate.acceptance_rate_ema
            and candidate.acceptance_rate_ema < conditions.min_acceptance_ema
        ):
            return False
        if candidate.match_scope == "toy_exact" and not set(
            candidate.required_profile_ids
        ).issubset(conditions.profile_ids):
            return False
        if candidate.evidence_groups and not (
            set(candidate.evidence_groups) & conditions.evidence_groups
        ):
            return False
        return True

    def _match_sort_key(
        self, candidate: PhraseCandidate, conditions: PhraseConditions
    ) -> tuple[float, float, float, float, str]:
        dominant = set(conditions.dominant_dimensions)
        cand_dims = set(candidate.dominant_dimensions)
        evidence_groups = conditions.evidence_groups
        levels = conditions.levels or {}
        level_match = sum(
            1
            for dimension, level in candidate.levels.items()
            if levels.get(dimension) == level
        )
        score = 0.0
        if candidate.stage == conditions.stage:
            score += 25
        if candidate.risk_pattern and candidate.risk_pattern == conditions.risk_pattern:
            score += 12
        if cand_dims and cand_dims <= dominant:
            score += 10
        elif cand_dims & dominant:
            score += 5
        if level_match:
            score += 3 * level_match
        if candidate.evidence_balance == conditions.evidence_balance:
            score += 6
        if candidate.clinical_interpretation == conditions.clinical_interpretation:
            score += 4
        if candidate.evidence_groups:
            score += 4 * len(set(candidate.evidence_groups) & evidence_groups)
        return (
            -score,
            -candidate.acceptance_rate_ema,
            -candidate.offline_acceptance_estimate,
            -candidate.priority,
            candidate.phrase_id,
        )

    @staticmethod
    def _sort_key(candidate: PhraseCandidate) -> tuple[float, float, float, str]:
        return (
            -candidate.acceptance_rate_ema,
            -candidate.offline_acceptance_estimate,
            -candidate.priority,
            candidate.phrase_id,
        )

    @staticmethod
    def _load(path: Path) -> Iterable[PhraseCandidate]:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid phrase JSON at {path}:{line_no}: {exc}"
                    ) from exc
                candidate = _candidate_from_record(raw, path, line_no)
                if candidate is not None:
                    yield candidate


_INDEX_CACHE: dict[str, PhraseIndex] = {}


def get_phrase_index(phrase_db: str) -> PhraseIndex:
    index = _INDEX_CACHE.get(phrase_db)
    if index is None:
        index = PhraseIndex(phrase_db)
        _INDEX_CACHE[phrase_db] = index
    return index


def _candidate_from_record(
    raw: Mapping[str, Any], path: Path, line_no: int
) -> PhraseCandidate | None:
    token_ids = _normalize_token_ids(raw.get("token_ids"))
    if not token_ids:
        return None
    phrase_id = str(raw.get("phrase_id") or f"{path}:{line_no}")
    json_path = str(raw.get("json_path") or "").strip()
    stage = str(raw.get("stage") or "generic").strip()
    if not json_path or not stage:
        return None
    safe = bool(raw.get("safe", True))
    disabled = bool(raw.get("disabled", False))
    if not safe or disabled:
        return None
    return PhraseCandidate(
        phrase_id=phrase_id,
        json_path=json_path,
        stage=stage,
        token_ids=tuple(token_ids),
        token_count=int(raw.get("token_count") or len(token_ids)),
        text=str(raw.get("text") or ""),
        risk_pattern=str(raw.get("risk_pattern") or ""),
        dominant_dimensions=tuple(_normalize_str_list(raw.get("dominant_dimensions"))),
        levels=_normalize_levels(raw.get("levels")),
        evidence_balance=str(raw.get("evidence_balance") or ""),
        clinical_interpretation=str(raw.get("clinical_interpretation") or ""),
        evidence_groups=tuple(_normalize_str_list(raw.get("evidence_groups"))),
        polarity=str(raw.get("polarity") or ""),
        anchor_type=str(raw.get("anchor_type") or ""),
        match_scope=str(raw.get("match_scope") or "state_generic"),
        required_profile_ids=tuple(_normalize_str_list(raw.get("required_profile_ids"))),
        priority=float(raw.get("priority") or 0.0),
        source=str(raw.get("source") or ""),
        offline_acceptance_estimate=float(
            raw.get("offline_acceptance_estimate") or 0.0
        ),
        acceptance_rate_ema=float(raw.get("acceptance_rate_ema") or 0.0),
        proposed_count=int(raw.get("proposed_count") or 0),
        disabled=disabled,
        safe=safe,
    )


def _normalize_token_ids(value: Any) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    token_ids: list[int] = []
    for item in value:
        try:
            token_ids.append(int(item))
        except (TypeError, ValueError):
            return []
    return token_ids


def _normalize_str_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_levels(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, int] = {}
    for dimension, level in value.items():
        try:
            normalized[str(dimension)] = int(level)
        except (TypeError, ValueError):
            continue
    return normalized

