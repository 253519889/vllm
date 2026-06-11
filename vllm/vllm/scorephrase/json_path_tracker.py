# SPDX-License-Identifier: Apache-2.0
"""Config C token-plan runtime and JSON path tracker.

The runtime is deliberately observational: it never edits model output. It
tracks accepted text against the request-local Config C token plan so later
draft/jump code can know which JSON path is currently active.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence


DETERMINISTIC_KINDS = {"fixed", "program", "choice", "score"}
MODEL_SPAN_KIND = "model_span"


class ConfigCTokenPlanRuntime:
    """Track progress through a Config C token plan using output text deltas."""

    def __init__(self, token_plan: Sequence[Mapping[str, Any]]) -> None:
        self.token_plan = [_normalize_segment(segment) for segment in token_plan]
        self.segment_index = 0
        self.segment_offset = 0
        self.observed_chars = 0
        self.ignored_whitespace_chars = 0
        self.visited_paths: list[str] = []
        self.disabled_reason = ""

        self._deterministic_in_string = False
        self._deterministic_escape = False
        self._model_string_escape = False
        self._model_value_started = False
        self._model_value_complete = False
        self._model_depth = 0
        self._model_in_string = False
        self._model_escape = False
        self._reset_segment_state()

    @classmethod
    def from_state(
        cls, sand_fsm_state: Mapping[str, Any] | None
    ) -> "ConfigCTokenPlanRuntime | None":
        if not sand_fsm_state:
            return None
        token_plan = sand_fsm_state.get("token_plan")
        if not token_plan:
            return None
        if not isinstance(token_plan, Sequence) or isinstance(token_plan, str):
            return None
        return cls(token_plan)

    @property
    def enabled(self) -> bool:
        return not self.disabled_reason

    @property
    def completed(self) -> bool:
        return self.enabled and self.segment_index >= len(self.token_plan)

    @property
    def current_json_path(self) -> str:
        if not self.enabled:
            return ""
        return self._path_for_segment_index(self.segment_index)

    def observe_text(self, text_delta: str) -> None:
        if not text_delta or not self.enabled:
            return
        for char in text_delta:
            if not self.enabled:
                return
            self.observed_chars += 1
            self._observe_char(char)

    def next_deterministic_span(self) -> str:
        if not self.enabled or self.segment_index >= len(self.token_plan):
            return ""
        segment = self.token_plan[self.segment_index]
        if segment["kind"] in DETERMINISTIC_KINDS:
            return segment.get("text", "")[self.segment_offset :]
        if segment["kind"] == MODEL_SPAN_KIND and self._model_value_complete:
            next_segment = self._next_deterministic_segment()
            return next_segment.get("text", "") if next_segment else ""
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "completed": self.completed,
            "disabled_reason": self.disabled_reason,
            "segment_index": self.segment_index,
            "segment_offset": self.segment_offset,
            "current_json_path": self.current_json_path,
            "visited_paths": list(self.visited_paths),
            "observed_chars": self.observed_chars,
            "ignored_whitespace_chars": self.ignored_whitespace_chars,
        }

    def disable(self, reason: str) -> None:
        self._disable(reason)

    def _observe_char(self, char: str) -> None:
        if self.segment_index >= len(self.token_plan):
            if char.isspace():
                self.ignored_whitespace_chars += 1
                return
            self._disable(f"unexpected trailing text: {char!r}")
            return

        segment = self.token_plan[self.segment_index]
        if segment["kind"] == MODEL_SPAN_KIND:
            self._observe_model_span_char(char, segment)
            return
        self._observe_deterministic_char(char, segment)

    def _observe_deterministic_char(
        self, char: str, segment: Mapping[str, Any]
    ) -> None:
        expected_text = segment.get("text", "")
        if self.segment_offset >= len(expected_text):
            self._advance_segment()
            self._observe_char(char)
            return

        expected = expected_text[self.segment_offset]
        if (
            char.isspace()
            and not expected.isspace()
            and not self._deterministic_in_string
        ):
            self.ignored_whitespace_chars += 1
            return

        if char != expected:
            self._disable(
                "token_plan mismatch at segment "
                f"{self.segment_index}, offset {self.segment_offset}: "
                f"expected {expected!r}, got {char!r}"
            )
            return

        self.segment_offset += 1
        self._update_deterministic_string_state(char)
        if self.segment_offset == len(expected_text):
            if segment.get("json_path") and segment["kind"] != "fixed":
                self._visit_path(str(segment["json_path"]))
            self._advance_segment()

    def _observe_model_span_char(
        self, char: str, segment: Mapping[str, Any]
    ) -> None:
        value_type = str(segment.get("value_type") or "")
        if value_type.startswith("str"):
            self._observe_string_model_span_char(char, segment)
            return
        if value_type.startswith("object") or value_type.startswith("list"):
            self._observe_json_value_model_span_char(char, segment)
            return
        self._observe_generic_model_span_char(char, segment)

    def _observe_string_model_span_char(
        self, char: str, segment: Mapping[str, Any]
    ) -> None:
        if char == '"' and not self._model_string_escape:
            self._complete_model_span(segment)
            self._observe_char(char)
            return

        if self._model_string_escape:
            self._model_string_escape = False
        elif char == "\\":
            self._model_string_escape = True

    def _observe_json_value_model_span_char(
        self, char: str, segment: Mapping[str, Any]
    ) -> None:
        if self._model_value_complete:
            self._complete_model_span(segment)
            self._observe_char(char)
            return

        if not self._model_value_started:
            if char.isspace():
                self.ignored_whitespace_chars += 1
                return
            expected_openers = "{[" if str(segment.get("value_type", "")).startswith(
                ("object", "list")
            ) else ""
            if char not in expected_openers:
                self._disable(
                    f"model span {segment.get('json_path', '')} expected JSON "
                    f"value, got {char!r}"
                )
                return
            self._model_value_started = True
            self._model_depth = 1
            return

        if self._model_in_string:
            if self._model_escape:
                self._model_escape = False
            elif char == "\\":
                self._model_escape = True
            elif char == '"':
                self._model_in_string = False
            return

        if char == '"':
            self._model_in_string = True
        elif char in "{[":
            self._model_depth += 1
        elif char in "}]":
            self._model_depth -= 1
            if self._model_depth == 0:
                self._model_value_complete = True
            elif self._model_depth < 0:
                self._disable(
                    f"model span {segment.get('json_path', '')} depth underflow"
                )

    def _observe_generic_model_span_char(
        self, char: str, segment: Mapping[str, Any]
    ) -> None:
        next_segment = self._next_deterministic_segment()
        if not next_segment:
            return
        expected_text = next_segment.get("text", "")
        if not expected_text:
            return
        if char == expected_text[0]:
            self._complete_model_span(segment)
            self._observe_char(char)

    def _advance_segment(self) -> None:
        self.segment_index += 1
        self.segment_offset = 0
        self._reset_segment_state()

    def _reset_segment_state(self) -> None:
        self._deterministic_in_string = False
        self._deterministic_escape = False
        self._model_string_escape = False
        self._model_value_started = False
        self._model_value_complete = False
        self._model_depth = 0
        self._model_in_string = False
        self._model_escape = False

    def _update_deterministic_string_state(self, char: str) -> None:
        if self._deterministic_escape:
            self._deterministic_escape = False
            return
        if self._deterministic_in_string and char == "\\":
            self._deterministic_escape = True
            return
        if char == '"':
            self._deterministic_in_string = not self._deterministic_in_string

    def _complete_model_span(self, segment: Mapping[str, Any]) -> None:
        if segment.get("json_path"):
            self._visit_path(str(segment["json_path"]))
        self._advance_segment()

    def _next_deterministic_segment(self) -> Mapping[str, Any] | None:
        for segment in self.token_plan[self.segment_index + 1 :]:
            if segment["kind"] in DETERMINISTIC_KINDS:
                return segment
        return None

    def _path_for_segment_index(self, index: int) -> str:
        if index >= len(self.token_plan):
            return ""
        segment = self.token_plan[index]
        if segment.get("json_path"):
            return str(segment["json_path"])
        for next_segment in self.token_plan[index + 1 :]:
            if next_segment.get("json_path"):
                return str(next_segment["json_path"])
        return ""

    def _visit_path(self, json_path: str) -> None:
        if json_path and json_path not in self.visited_paths:
            self.visited_paths.append(json_path)

    def _disable(self, reason: str) -> None:
        self.disabled_reason = reason


def _normalize_segment(segment: Mapping[str, Any]) -> dict[str, Any]:
    kind = str(segment.get("kind") or "")
    data: dict[str, Any] = {"kind": kind}
    if "text" in segment:
        data["text"] = str(segment.get("text") or "")
    if "json_path" in segment:
        data["json_path"] = str(segment.get("json_path") or "")
    if "mode" in segment:
        data["mode"] = str(segment.get("mode") or "")
    if "value_type" in segment:
        data["value_type"] = str(segment.get("value_type") or "")
    return deepcopy(data)
