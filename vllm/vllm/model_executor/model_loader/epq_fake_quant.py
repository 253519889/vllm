# SPDX-License-Identifier: Apache-2.0
"""Startup-only fake quantization for evidence-preserving sensitivity tests.

This module intentionally does not install real quantized kernels. It mutates
selected floating-point weight tensors after checkpoint loading by quantizing
and dequantizing them back to their original dtype. The goal is layer/module
sensitivity profiling while keeping the normal vLLM execution path intact.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class EPQFakeQuantRule:
    patterns: tuple[str, ...]
    precision: str = "int4"
    group_size: int = 128
    regex: bool = False
    exclude_patterns: tuple[str, ...] = ()
    name: str = ""


@dataclass(frozen=True)
class EPQFakeQuantStats:
    matched_params: int
    quantized_params: int
    skipped_params: int
    quantized_elements: int
    kept_params: int = 0


def apply_epq_fake_quant_from_config(
    model: nn.Module,
    config: Any,
) -> EPQFakeQuantStats:
    rules = parse_epq_fake_quant_config(config)
    if not rules:
        return EPQFakeQuantStats(
            matched_params=0,
            quantized_params=0,
            skipped_params=0,
            quantized_elements=0,
            kept_params=0,
        )

    matched = 0
    quantized = 0
    skipped = 0
    kept = 0
    elements = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            rule = _matching_rule(name, rules)
            if rule is None:
                continue
            matched += 1
            if rule.precision == "fp16":
                kept += 1
                logger.info("EPQ fake quant keep fp16: %s", name)
                continue
            if not _can_fake_quantize(param):
                skipped += 1
                logger.info(
                    "EPQ fake quant skip %s shape=%s dtype=%s",
                    name,
                    tuple(param.shape),
                    param.dtype,
                )
                continue
            param.data.copy_(
                fake_quantize_weight(
                    param.data,
                    precision=rule.precision,
                    group_size=rule.group_size,
                )
            )
            quantized += 1
            elements += param.numel()
            logger.info(
                "EPQ fake quant applied: %s precision=%s group_size=%d shape=%s",
                name,
                rule.precision,
                rule.group_size,
                tuple(param.shape),
            )

    stats = EPQFakeQuantStats(
        matched_params=matched,
        quantized_params=quantized,
        skipped_params=skipped,
        quantized_elements=elements,
        kept_params=kept,
    )
    logger.info(
        "EPQ fake quant summary: matched=%d quantized=%d skipped=%d kept=%d elements=%d",
        stats.matched_params,
        stats.quantized_params,
        stats.skipped_params,
        stats.kept_params,
        stats.quantized_elements,
    )
    return stats


def parse_epq_fake_quant_config(config: Any) -> tuple[EPQFakeQuantRule, ...]:
    """Parse config from --epq-fake-quant-config or model_loader_extra_config.

    Supported examples:
      {"patterns":["layers.24."],"bits":4,"group_size":128}
      {"patterns":["layers.24."],"precision":"fp8","group_size":128}
      {"rules":[{"patterns":["layers.0."],"precision":"int4"},{"patterns":["lm_head"],"precision":"fp16"}]}
      "layers.24.:int4,layers.25.:fp8,lm_head:fp16"
    """

    if not config:
        return ()
    if isinstance(config, str):
        return _parse_string_config(config)
    if isinstance(config, Mapping):
        if config.get("enabled") is False:
            return ()
        raw_rules = config.get("rules")
        if raw_rules is not None:
            return tuple(_rule_from_mapping(rule) for rule in raw_rules)
        return (_rule_from_mapping(config),)
    if isinstance(config, Sequence):
        return tuple(_rule_from_mapping(rule) for rule in config)
    raise TypeError(f"Unsupported EPQ fake quant config type: {type(config)!r}")


def fake_quantize_weight(
    weight: torch.Tensor,
    *,
    bits: int | None = None,
    precision: str | None = None,
    group_size: int = 128,
) -> torch.Tensor:
    precision = _parse_precision(precision if precision is not None else bits or 4)
    if precision == "fp16":
        return weight.detach().clone()
    if weight.ndim < 2:
        raise ValueError("EPQ fake quant expects a matrix-like weight tensor")
    if precision == "fp8":
        return fake_quantize_fp8_e4m3_weight(weight, group_size=group_size)

    bits = _int_precision_bits(precision)

    original_dtype = weight.dtype
    working = weight.detach().to(torch.float32)
    flat = working.reshape(-1, working.shape[-1])
    qmax = (2 ** (bits - 1)) - 1
    chunk_size = flat.shape[-1] if group_size <= 0 else int(group_size)

    chunks: list[torch.Tensor] = []
    for start in range(0, flat.shape[-1], chunk_size):
        chunk = flat[:, start : start + chunk_size]
        scale = chunk.abs().amax(dim=1, keepdim=True).div(float(qmax))
        scale = scale.clamp(min=1e-8)
        quantized = torch.round(chunk / scale).clamp(-qmax, qmax)
        chunks.append(quantized * scale)
    return torch.cat(chunks, dim=1).reshape_as(weight).to(original_dtype)


def fake_quantize_fp8_e4m3_weight(
    weight: torch.Tensor,
    *,
    group_size: int = 128,
) -> torch.Tensor:
    """Approximate block-scaled FP8 E4M3 quantize-dequantize for weights.

    This is intentionally a sensitivity approximation, not a deployment kernel.
    It scales each row/group to the E4M3 finite max range, maps values onto an
    E4M3-like grid, then dequantizes back to the original dtype.
    """

    if weight.ndim < 2:
        raise ValueError("EPQ fake FP8 expects a matrix-like weight tensor")

    original_dtype = weight.dtype
    working = weight.detach().to(torch.float32)
    flat = working.reshape(-1, working.shape[-1])
    chunk_size = flat.shape[-1] if group_size <= 0 else int(group_size)
    fp8_max = 448.0

    chunks: list[torch.Tensor] = []
    for start in range(0, flat.shape[-1], chunk_size):
        chunk = flat[:, start : start + chunk_size]
        scale = chunk.abs().amax(dim=1, keepdim=True).div(fp8_max)
        scale = scale.clamp(min=1e-8)
        normalized = chunk / scale
        chunks.append(_quantize_e4m3fn_grid(normalized) * scale)
    return torch.cat(chunks, dim=1).reshape_as(weight).to(original_dtype)


def _quantize_e4m3fn_grid(value: torch.Tensor) -> torch.Tensor:
    sign = torch.sign(value)
    abs_value = value.abs().clamp(max=448.0)
    nonzero = abs_value > 0

    # E4M3FN is approximated as exponent range [-6, 8], 3 mantissa bits,
    # finite max 448, and subnormal step 2^-9.
    min_normal = 2.0 ** -6
    subnormal_step = 2.0 ** -9
    exponent = torch.floor(torch.log2(abs_value.clamp(min=min_normal)))
    exponent = exponent.clamp(min=-6, max=8)
    step = torch.pow(
        torch.full_like(abs_value, 2.0),
        exponent - 3.0,
    )
    normal = torch.round(abs_value / step) * step
    subnormal = torch.round(abs_value / subnormal_step) * subnormal_step
    quantized_abs = torch.where(abs_value < min_normal, subnormal, normal)
    quantized_abs = quantized_abs.clamp(max=448.0)
    return torch.where(nonzero, sign * quantized_abs, torch.zeros_like(value))


def _rule_from_mapping(value: Any) -> EPQFakeQuantRule:
    if not isinstance(value, Mapping):
        raise TypeError("Each EPQ fake quant rule must be a mapping")
    patterns = _string_tuple(value.get("patterns") or value.get("pattern"))
    if not patterns:
        raise ValueError("EPQ fake quant rule requires non-empty patterns")
    precision = _parse_precision(
        value.get("precision", value.get("dtype", value.get("bits", 4)))
    )
    group_size = int(value.get("group_size", 128))
    regex = bool(value.get("regex", False))
    exclude_patterns = _string_tuple(value.get("exclude_patterns") or ())
    return EPQFakeQuantRule(
        patterns=patterns,
        precision=precision,
        group_size=group_size,
        regex=regex,
        exclude_patterns=exclude_patterns,
        name=str(value.get("name") or ""),
    )


def _parse_string_config(config: str) -> tuple[EPQFakeQuantRule, ...]:
    rules: list[EPQFakeQuantRule] = []
    for item in config.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            pattern, raw_bits = item.rsplit(":", 1)
        else:
            pattern, raw_bits = item, "int4"
        rules.append(
            EPQFakeQuantRule(
                patterns=(pattern.strip(),),
                precision=_parse_precision(raw_bits),
                group_size=128,
            )
        )
    return tuple(rules)


def _parse_precision(value: Any) -> str:
    if value is None:
        return "int4"
    if isinstance(value, int) and not isinstance(value, bool):
        value = f"int{value}"
    text = str(value).lower().strip()
    aliases = {
        "4": "int4",
        "8": "int8",
        "w4": "int4",
        "w8": "int8",
        "int4": "int4",
        "int8": "int8",
        "fp8": "fp8",
        "float8": "fp8",
        "e4m3": "fp8",
        "e4m3fn": "fp8",
        "fp16": "fp16",
        "float16": "fp16",
        "half": "fp16",
        "none": "fp16",
        "keep": "fp16",
    }
    if text not in aliases:
        raise ValueError(
            "EPQ fake quant precision must be one of int4, int8, fp8, fp16; "
            f"got {value}"
        )
    return aliases[text]


def _int_precision_bits(precision: str) -> int:
    if precision == "int4":
        return 4
    if precision == "int8":
        return 8
    raise ValueError(f"EPQ integer fake quant requires int4 or int8, got {precision}")


def _matching_rule(
    name: str,
    rules: Sequence[EPQFakeQuantRule],
) -> EPQFakeQuantRule | None:
    if not name.endswith(".weight"):
        return None
    for rule in rules:
        if _matches_any(name, rule.exclude_patterns, rule.regex):
            continue
        if _matches_any(name, rule.patterns, rule.regex):
            return rule
    return None


def _matches_any(name: str, patterns: Sequence[str], regex: bool) -> bool:
    if regex:
        return any(re.search(pattern, name) for pattern in patterns)
    return any(pattern in name for pattern in patterns)


def _can_fake_quantize(param: torch.Tensor) -> bool:
    return param.is_floating_point() and param.ndim >= 2


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, Sequence):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()
