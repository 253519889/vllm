# SPDX-License-Identifier: Apache-2.0
"""ScorePhrase request-local Score-729 state helpers.

This module intentionally lives under vLLM because it is part of the online
serving path. The sandplay package may provide offline/dry-run helpers, but the
server must be able to construct and validate the score regex itself.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


DEFAULT_DIMENSIONS = (
    "\u6291\u90c1",
    "\u5f3a\u8feb",
    "\u7126\u8651",
    "\u4eba\u9645\u654f\u611f",
    "\u654c\u5bf9",
    "\u504f\u6267",
)
DEFAULT_LEVELS = (
    {"code": "0", "label": "\u6b63\u5e38", "value": 0},
    {"code": "1", "label": "\u8f7b\u5ea6", "value": 1},
    {"code": "2", "label": "\u91cd\u5ea6", "value": 2},
)


def build_scorephrase_score_space(
    payload: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Build request-local ScorePhrase state from OpenAI extra_body.

    Expected minimal payload:

    {
      "enabled": true,
      "allowed_levels": {"抑郁": [0], "敌对": [0, 1, 2]}
    }

    Missing dimensions default to level 0, so callers can send only the
    dimensions opened by Step2 pruning.
    """

    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError("scorephrase must be an object")
    if payload.get("enabled", True) is False:
        return None

    dimensions = _normalize_dimensions(payload.get("dimensions"))
    levels = _normalize_levels(payload.get("levels"))
    allowed_levels = _normalize_allowed_levels(
        payload.get("allowed_levels"),
        dimensions=dimensions,
        levels=levels,
    )
    regex = score_regex_from_allowed_levels(
        allowed_levels,
        dimensions=dimensions,
        levels=levels,
    )
    state = {
        "mode": str(payload.get("mode") or "score_code"),
        "dimensions": list(dimensions),
        "levels": list(levels),
        "allowed_levels": {
            dimension: list(allowed_levels[dimension])
            for dimension in dimensions
        },
        "score_regex": regex,
        "score_code_count": score_code_count(allowed_levels, dimensions),
    }
    for key in (
        "evidence_ids_by_dimension",
        "dimension_risk_scores",
        "dimension_protective_scores",
        "matched_profile_ids",
        "missing_toys",
    ):
        if key in payload:
            state[key] = payload[key]
    return state


def score_regex_from_allowed_levels(
    allowed_levels: Mapping[str, Sequence[int]],
    dimensions: Sequence[str] = DEFAULT_DIMENSIONS,
    levels: Sequence[dict[str, Any]] = DEFAULT_LEVELS,
) -> str:
    code_by_value = _code_by_value(levels)
    return "^" + "".join(
        _levels_to_regex(allowed_levels.get(dimension, (0,)), code_by_value)
        for dimension in dimensions
    ) + "$"


def score_code_count(
    allowed_levels: Mapping[str, Sequence[int]],
    dimensions: Sequence[str] = DEFAULT_DIMENSIONS,
) -> int:
    count = 1
    for dimension in dimensions:
        count *= max(1, len(tuple(allowed_levels.get(dimension, (0,)))))
    return count


def _normalize_dimensions(value: Any) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_DIMENSIONS
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError("scorephrase.dimensions must be a list")
    dimensions = tuple(str(item).strip() for item in value if str(item).strip())
    if not dimensions:
        raise ValueError("scorephrase.dimensions must not be empty")
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("scorephrase.dimensions must be unique")
    return dimensions


def _normalize_levels(value: Any) -> tuple[dict[str, Any], ...]:
    raw_levels = DEFAULT_LEVELS if value is None else value
    if not isinstance(raw_levels, Sequence) or isinstance(raw_levels, str):
        raise ValueError("scorephrase.levels must be a list")
    levels: list[dict[str, Any]] = []
    for item in raw_levels:
        if not isinstance(item, Mapping):
            raise ValueError("scorephrase.levels items must be objects")
        code = str(item.get("code", "")).strip()
        label = str(item.get("label", "")).strip()
        try:
            level_value = int(item.get("value"))
        except (TypeError, ValueError) as exc:
            raise ValueError("scorephrase.levels value must be an integer") from exc
        if not code:
            raise ValueError("scorephrase.levels code must not be empty")
        levels.append({"code": code, "label": label, "value": level_value})
    if not levels:
        raise ValueError("scorephrase.levels must not be empty")
    if len({level["code"] for level in levels}) != len(levels):
        raise ValueError("scorephrase.levels codes must be unique")
    if len({level["value"] for level in levels}) != len(levels):
        raise ValueError("scorephrase.levels values must be unique")
    return tuple(levels)


def _normalize_allowed_levels(
    value: Any,
    dimensions: Sequence[str],
    levels: Sequence[dict[str, Any]],
) -> dict[str, tuple[int, ...]]:
    if value is None:
        raise ValueError("scorephrase.allowed_levels is required")
    if not isinstance(value, Mapping):
        raise ValueError("scorephrase.allowed_levels must be an object")

    value_by_code = _value_by_code(levels)
    valid_values = set(value_by_code.values())
    allowed: dict[str, tuple[int, ...]] = {}
    unknown_dimensions = set(str(key) for key in value) - set(dimensions)
    if unknown_dimensions:
        raise ValueError(
            "scorephrase.allowed_levels contains unknown dimensions: "
            f"{sorted(unknown_dimensions)}"
        )

    for dimension in dimensions:
        raw_levels = value.get(dimension, (0,))
        if isinstance(raw_levels, (str, int)):
            raw_levels = (raw_levels,)
        if not isinstance(raw_levels, Sequence):
            raise ValueError(
                f"scorephrase.allowed_levels[{dimension}] must be a list"
            )
        normalized: list[int] = []
        for raw_level in raw_levels:
            level_value = _coerce_level(raw_level, value_by_code)
            if level_value not in valid_values:
                raise ValueError(
                    f"scorephrase.allowed_levels[{dimension}] has invalid "
                    f"level {raw_level!r}"
                )
            if level_value not in normalized:
                normalized.append(level_value)
        allowed[dimension] = tuple(sorted(normalized)) or (0,)
    return allowed


def _levels_to_regex(
    values: Sequence[int],
    code_by_value: Mapping[int, str],
) -> str:
    codes = "".join(code_by_value[int(value)] for value in sorted(set(values)))
    if len(codes) == 1:
        return codes
    return f"[{codes}]"


def _coerce_level(value: Any, value_by_code: Mapping[str, int]) -> int:
    if isinstance(value, str) and not value.isdigit():
        if value not in value_by_code:
            raise ValueError(f"unknown score level code: {value}")
        return value_by_code[value]
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid score level: {value!r}") from exc


def _value_by_code(levels: Sequence[dict[str, Any]]) -> dict[str, int]:
    return {str(level["code"]): int(level["value"]) for level in levels}


def _code_by_value(levels: Sequence[dict[str, Any]]) -> dict[int, str]:
    return {int(level["value"]): str(level["code"]) for level in levels}
