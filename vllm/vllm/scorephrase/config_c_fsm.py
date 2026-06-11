# SPDX-License-Identifier: Apache-2.0
"""Config C Sand-FSM request-local state helpers.

This module lives under vLLM because Config C field jump state is part of the
online serving path. Client-side helpers can prepare the payload, but the server
must validate and keep the request-local FSM state itself.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence


DEFAULT_VERSION = "config_c_fsm_v1"
GENERATION_MODES = ("PROGRAM", "CHOICE", "SCORE", "PHRASE", "REVISION", "SPAN")
TOKEN_PLAN_KINDS = ("fixed", "program", "choice", "score", "model_span")
TOP_LEVEL_ORDER = (
    "视觉校验",
    "视觉特征",
    "符号解析",
    "动力学评估",
    "综合评估",
    "mental_dims",
    "reasoning",
)
EVIDENCE_BALANCE_CHOICES = ("受损主导", "愈合有效对抗", "证据不足")
CLINICAL_INTERPRETATION_CHOICES = (
    "按临床问题处理",
    "处于应激下的自我调节",
    "未见明显临床特征",
)

DEFAULT_FIELD_NODES = (
    {
        "json_path": "视觉校验.已确认沙具",
        "mode": "PROGRAM",
        "value_type": "list[str]",
        "locked": True,
        "source": "toy_list",
    },
    {
        "json_path": "视觉校验.未确认沙具",
        "mode": "PROGRAM",
        "value_type": "list[str]",
        "locked": True,
        "fallback_modes": ["REVISION"],
        "source": "visual_check",
    },
    {
        "json_path": "视觉校验.图中额外沙具",
        "mode": "PROGRAM",
        "value_type": "list[str]",
        "locked": True,
        "fallback_modes": ["REVISION"],
        "source": "visual_check",
    },
    {
        "json_path": "视觉特征.关键沙具",
        "mode": "PHRASE",
        "value_type": "str",
        "fallback_modes": ["SPAN", "REVISION"],
        "source": "phrase_db",
    },
    {
        "json_path": "视觉特征.构图特征",
        "mode": "PHRASE",
        "value_type": "str",
        "fallback_modes": ["SPAN", "REVISION"],
        "source": "phrase_db",
    },
    {
        "json_path": "视觉特征.动态时序",
        "mode": "PROGRAM",
        "value_type": "str",
        "locked": True,
        "fallback_modes": ["PHRASE"],
        "source": "operations",
    },
    {
        "json_path": "符号解析.核心沙具原型",
        "mode": "PHRASE",
        "value_type": "str",
        "fallback_modes": ["REVISION"],
        "source": "toy_evidence_profile",
    },
    {
        "json_path": "符号解析.自述与构图关联",
        "mode": "SPAN",
        "value_type": "str",
        "fallback_modes": ["REVISION"],
        "source": "own_story",
    },
    {
        "json_path": "动力学评估.信息对撞",
        "mode": "SPAN",
        "value_type": "str",
        "fallback_modes": ["REVISION"],
        "source": "visual_story_contrast",
    },
    {
        "json_path": "动力学评估.整体动力",
        "mode": "PHRASE",
        "value_type": "str",
        "fallback_modes": ["REVISION"],
        "source": "phrase_db",
    },
    {
        "json_path": "综合评估.受损证据",
        "mode": "PROGRAM",
        "value_type": "str",
        "fallback_modes": ["PHRASE", "REVISION"],
        "source": "risk_evidence_cards",
    },
    {
        "json_path": "综合评估.愈合资源",
        "mode": "PROGRAM",
        "value_type": "str",
        "fallback_modes": ["PHRASE", "REVISION"],
        "source": "protective_evidence_cards",
    },
    {
        "json_path": "综合评估.证据权衡",
        "mode": "CHOICE",
        "value_type": "str",
        "locked": True,
        "choices": list(EVIDENCE_BALANCE_CHOICES),
        "source": "score_cards",
    },
    {
        "json_path": "综合评估.临床解读",
        "mode": "CHOICE",
        "value_type": "str",
        "locked": True,
        "choices": list(CLINICAL_INTERPRETATION_CHOICES),
        "source": "score_cards",
    },
    {
        "json_path": "mental_dims",
        "mode": "SCORE",
        "value_type": "object[int]",
        "locked": True,
        "source": "score_code",
    },
    {
        "json_path": "reasoning",
        "mode": "PHRASE",
        "value_type": "str",
        "fallback_modes": ["REVISION"],
        "source": "score_cards",
    },
)


def build_config_c_fsm_state(
    payload: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate and normalize a Config C Sand-FSM payload."""

    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError("sand_fsm must be an object")
    if payload.get("enabled", True) is False:
        return None

    top_level_order = _normalize_top_level_order(payload.get("top_level_order"))
    field_nodes = _normalize_field_nodes(payload.get("field_nodes"))
    field_map = {node["json_path"]: node for node in field_nodes}
    program_fields = _normalize_program_fields(
        payload.get("program_fields", {}),
        field_map,
    )
    score_fields = _normalize_score_fields(
        payload.get("score_fields", payload.get("mental_dims", {}))
    )
    locked_paths = _normalize_paths(
        payload.get("locked_paths"),
        field_map,
        default=_default_locked_paths(field_nodes, program_fields, bool(score_fields)),
        label="sand_fsm.locked_paths",
    )
    jump_paths = _normalize_paths(
        payload.get("jump_paths"),
        field_map,
        default=_default_jump_paths(field_nodes, program_fields, bool(score_fields)),
        label="sand_fsm.jump_paths",
    )
    model_paths = _normalize_paths(
        payload.get("model_paths"),
        field_map,
        default=_default_model_paths(field_nodes),
        label="sand_fsm.model_paths",
    )
    pending_paths = _normalize_paths(
        payload.get("pending_paths"),
        field_map,
        default=tuple(path for path in model_paths if path not in locked_paths),
        label="sand_fsm.pending_paths",
    )
    token_plan = _normalize_token_plan(payload.get("token_plan"), field_map)
    force_known_spans = _coerce_bool(payload.get("force_known_spans", False))
    max_forced_tokens_per_step = _normalize_positive_int(
        payload.get("max_forced_tokens_per_step", 64),
        default=64,
        minimum=1,
        maximum=512,
        label="sand_fsm.max_forced_tokens_per_step",
    )

    return {
        "version": str(payload.get("version") or DEFAULT_VERSION),
        "mode": str(payload.get("mode") or "config_c"),
        "top_level_order": list(top_level_order),
        "field_nodes": deepcopy(field_nodes),
        "program_fields": program_fields,
        "score_fields": score_fields,
        "locked_paths": list(locked_paths),
        "jump_paths": list(jump_paths),
        "model_paths": list(model_paths),
        "pending_paths": list(pending_paths),
        "token_plan": token_plan,
        "force_known_spans": force_known_spans,
        "max_forced_tokens_per_step": max_forced_tokens_per_step,
    }


def _normalize_top_level_order(value: Any) -> tuple[str, ...]:
    if value is None:
        return TOP_LEVEL_ORDER
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError("sand_fsm.top_level_order must be a list")
    normalized = tuple(str(item) for item in value)
    if normalized != TOP_LEVEL_ORDER:
        raise ValueError(
            "sand_fsm.top_level_order must match PROMPT_CONFIG_C top-level order"
        )
    return normalized


def _normalize_field_nodes(value: Any) -> tuple[dict[str, Any], ...]:
    raw_nodes = DEFAULT_FIELD_NODES if value is None else value
    if not isinstance(raw_nodes, Sequence) or isinstance(raw_nodes, str):
        raise ValueError("sand_fsm.field_nodes must be a list")
    nodes = []
    seen = set()
    for item in raw_nodes:
        if not isinstance(item, Mapping):
            raise ValueError("sand_fsm.field_nodes items must be objects")
        json_path = str(item.get("json_path") or "").strip()
        mode = str(item.get("mode") or "").strip()
        if not json_path:
            raise ValueError("sand_fsm.field_nodes json_path must not be empty")
        if json_path in seen:
            raise ValueError(f"duplicate sand_fsm json_path: {json_path}")
        if mode not in GENERATION_MODES:
            raise ValueError(f"invalid sand_fsm mode for {json_path}: {mode}")
        fallback_modes = item.get("fallback_modes", [])
        if not isinstance(fallback_modes, Sequence) or isinstance(fallback_modes, str):
            raise ValueError(f"sand_fsm fallback_modes for {json_path} must be a list")
        normalized_fallbacks = [str(mode) for mode in fallback_modes]
        unknown_fallbacks = set(normalized_fallbacks) - set(GENERATION_MODES)
        if unknown_fallbacks:
            raise ValueError(
                f"invalid sand_fsm fallback_modes for {json_path}: "
                f"{sorted(unknown_fallbacks)}"
            )
        choices = item.get("choices", [])
        if not isinstance(choices, Sequence) or isinstance(choices, str):
            raise ValueError(f"sand_fsm choices for {json_path} must be a list")
        nodes.append(
            {
                "json_path": json_path,
                "mode": mode,
                "value_type": str(item.get("value_type") or "str"),
                "locked": bool(item.get("locked", False)),
                "fallback_modes": normalized_fallbacks,
                "choices": [str(choice) for choice in choices],
                "source": str(item.get("source") or ""),
            }
        )
        seen.add(json_path)
    if tuple(node["json_path"] for node in nodes) != tuple(
        node["json_path"] for node in DEFAULT_FIELD_NODES
    ):
        raise ValueError("sand_fsm.field_nodes must match Config C field order")
    return tuple(nodes)


def _normalize_program_fields(
    value: Any,
    field_map: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("sand_fsm.program_fields must be an object")
    result: dict[str, Any] = {}
    for raw_path, raw_value in value.items():
        json_path = str(raw_path)
        node = _node_for_path(json_path, field_map)
        if node["mode"] not in {"PROGRAM", "CHOICE"}:
            raise ValueError(f"{json_path} is not a program/choice field")
        if node["mode"] == "CHOICE":
            choices = node.get("choices", [])
            if str(raw_value) not in choices:
                raise ValueError(
                    f"sand_fsm.program_fields[{json_path}] must be one of {choices}"
                )
            result[json_path] = str(raw_value)
        else:
            result[json_path] = deepcopy(raw_value)
    return result


def _normalize_score_fields(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("sand_fsm.score_fields must be an object")
    result: dict[str, int] = {}
    for dimension, score in value.items():
        try:
            result[str(dimension)] = int(score)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"sand_fsm.score_fields[{dimension}] must be an integer"
            ) from exc
    return result


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return False


def _normalize_positive_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int,
    label: str,
) -> int:
    if value is None:
        return default
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if normalized < minimum:
        return minimum
    if normalized > maximum:
        return maximum
    return normalized


def _normalize_paths(
    value: Any,
    field_map: Mapping[str, Mapping[str, Any]],
    default: Sequence[str],
    label: str,
) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"{label} must be a list")
    paths = []
    for raw_path in value:
        json_path = str(raw_path)
        _node_for_path(json_path, field_map)
        if json_path not in paths:
            paths.append(json_path)
    return tuple(paths)


def _normalize_token_plan(
    value: Any,
    field_map: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError("sand_fsm.token_plan must be a list")
    result: list[dict[str, Any]] = []
    for index, raw_segment in enumerate(value):
        if not isinstance(raw_segment, Mapping):
            raise ValueError("sand_fsm.token_plan items must be objects")
        kind = str(raw_segment.get("kind") or "").strip()
        if kind not in TOKEN_PLAN_KINDS:
            raise ValueError(
                f"sand_fsm.token_plan[{index}].kind must be one of "
                f"{TOKEN_PLAN_KINDS}"
            )

        text = raw_segment.get("text", "")
        if kind != "model_span" and (not isinstance(text, str) or text == ""):
            raise ValueError(
                f"sand_fsm.token_plan[{index}].text must be a non-empty string"
            )
        if kind == "model_span" and text and not isinstance(text, str):
            raise ValueError(f"sand_fsm.token_plan[{index}].text must be a string")

        json_path = str(raw_segment.get("json_path") or "").strip()
        node: Mapping[str, Any] | None = None
        if kind != "fixed" and not json_path:
            raise ValueError(
                f"sand_fsm.token_plan[{index}].json_path must not be empty"
            )
        if json_path:
            node = _node_for_path(json_path, field_map)

        mode = str(raw_segment.get("mode") or "").strip()
        if not mode and node is not None:
            mode = str(node.get("mode") or "")
        if mode and mode not in GENERATION_MODES:
            raise ValueError(
                f"invalid sand_fsm token_plan mode for {json_path}: {mode}"
            )

        value_type = str(raw_segment.get("value_type") or "").strip()
        if not value_type and node is not None:
            value_type = str(node.get("value_type") or "")

        segment: dict[str, Any] = {"kind": kind}
        if text:
            segment["text"] = text
        if json_path:
            segment["json_path"] = json_path
        if mode:
            segment["mode"] = mode
        if value_type:
            segment["value_type"] = value_type
        result.append(segment)
    return result


def _node_for_path(
    json_path: str,
    field_map: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    node = field_map.get(json_path)
    if node is None:
        raise ValueError(f"unknown Config C FSM path: {json_path}")
    return node


def _default_locked_paths(
    field_nodes: Sequence[Mapping[str, Any]],
    program_fields: Mapping[str, Any],
    has_scores: bool,
) -> tuple[str, ...]:
    locked = set(program_fields)
    if has_scores:
        locked.add("mental_dims")
    return tuple(node["json_path"] for node in field_nodes if node["json_path"] in locked)


def _default_jump_paths(
    field_nodes: Sequence[Mapping[str, Any]],
    program_fields: Mapping[str, Any],
    has_scores: bool,
) -> tuple[str, ...]:
    jump = set(program_fields)
    if has_scores:
        jump.add("mental_dims")
    return tuple(node["json_path"] for node in field_nodes if node["json_path"] in jump)


def _default_model_paths(
    field_nodes: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    return tuple(
        node["json_path"]
        for node in field_nodes
        if node.get("mode") in {"PHRASE", "REVISION", "SPAN"}
    )
