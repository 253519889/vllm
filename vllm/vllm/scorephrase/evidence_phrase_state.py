# SPDX-License-Identifier: Apache-2.0
"""EvidencePhrase request-local state helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence


DEFAULT_ENABLED_PATHS = ("综合评估.心理状态概述",)
DEFAULT_ENABLED_STAGES = (
    "opening",
    "transition",
    "core_issue",
    "evidence",
    "resource",
    "closing",
)
VALID_STAGES = frozenset((*DEFAULT_ENABLED_STAGES, "generic"))


def build_evidence_phrase_state(
    payload: Mapping[str, Any] | None,
    default_config: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Validate and normalize an EvidencePhrase request payload."""

    merged: dict[str, Any] = {}
    if default_config:
        if not isinstance(default_config, Mapping):
            raise ValueError("evidence_phrase_config must be an object")
        merged.update(default_config)
    if payload is not None:
        if not isinstance(payload, Mapping):
            raise ValueError("evidence_phrase must be an object")
        merged.update(payload)

    if not merged:
        return None
    if merged.get("enabled", True) is False:
        return None

    phrase_db = str(merged.get("phrase_db") or "").strip()
    if not phrase_db:
        raise ValueError("evidence_phrase.phrase_db is required")

    enabled_paths = _normalize_str_list(
        merged.get("enabled_paths", merged.get("paths")),
        default=DEFAULT_ENABLED_PATHS,
        label="evidence_phrase.enabled_paths",
    )
    enabled_stages = _normalize_str_list(
        merged.get("enabled_stages", merged.get("stages")),
        default=DEFAULT_ENABLED_STAGES,
        label="evidence_phrase.enabled_stages",
    )
    unknown_stages = sorted(set(enabled_stages) - VALID_STAGES)
    if unknown_stages:
        raise ValueError(
            "evidence_phrase.enabled_stages contains unknown stages: "
            + ", ".join(unknown_stages)
        )

    return {
        "enabled": True,
        "phrase_db": phrase_db,
        "enabled_paths": list(enabled_paths),
        "enabled_stages": list(enabled_stages),
        "risk_pattern": _optional_str(merged.get("risk_pattern")),
        "dominant_dimensions": list(
            _normalize_str_list(
                merged.get("dominant_dimensions"),
                default=(),
                label="evidence_phrase.dominant_dimensions",
            )
        ),
        "levels": _normalize_levels(merged.get("levels", {})),
        "evidence_balance": _optional_str(merged.get("evidence_balance")),
        "clinical_interpretation": _optional_str(
            merged.get("clinical_interpretation")
        ),
        "evidence_groups": list(
            _normalize_str_list(
                merged.get("evidence_groups"),
                default=(),
                label="evidence_phrase.evidence_groups",
            )
        ),
        "profile_ids": list(
            _normalize_str_list(
                merged.get("profile_ids"),
                default=(),
                label="evidence_phrase.profile_ids",
            )
        ),
        "max_draft_tokens": _normalize_int(
            merged.get("max_draft_tokens", 16),
            default=16,
            minimum=1,
            maximum=64,
            label="evidence_phrase.max_draft_tokens",
        ),
        "dynamic_gamma": _coerce_bool(merged.get("dynamic_gamma", False)),
        "min_acceptance_ema": _normalize_float(
            merged.get("min_acceptance_ema", 0.0),
            default=0.0,
            minimum=0.0,
            maximum=1.0,
            label="evidence_phrase.min_acceptance_ema",
        ),
        "metadata": deepcopy(merged.get("metadata", {}))
        if isinstance(merged.get("metadata", {}), Mapping)
        else {},
    }


def _normalize_str_list(
    value: Any,
    *,
    default: Sequence[str],
    label: str,
) -> tuple[str, ...]:
    if value is None:
        return tuple(str(item) for item in default)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a list")
    normalized = tuple(str(item).strip() for item in value if str(item).strip())
    if not normalized and default:
        raise ValueError(f"{label} must not be empty")
    return normalized


def _normalize_levels(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("evidence_phrase.levels must be an object")
    normalized: dict[str, int] = {}
    for raw_dimension, raw_level in value.items():
        dimension = str(raw_dimension).strip()
        if not dimension:
            continue
        try:
            normalized[dimension] = int(raw_level)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"evidence_phrase.levels[{dimension}] must be an integer"
            ) from exc
    return normalized


def _normalize_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int,
    label: str,
) -> int:
    try:
        normalized = int(value if value is not None else default)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if not minimum <= normalized <= maximum:
        raise ValueError(f"{label} must be in [{minimum}, {maximum}]")
    return normalized


def _normalize_float(
    value: Any,
    *,
    default: float,
    minimum: float,
    maximum: float,
    label: str,
) -> float:
    try:
        normalized = float(value if value is not None else default)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc
    if not minimum <= normalized <= maximum:
        raise ValueError(f"{label} must be in [{minimum}, {maximum}]")
    return normalized


def _optional_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)

