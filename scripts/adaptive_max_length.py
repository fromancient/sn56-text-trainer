"""
Dataset-aware max sequence length with batch-size compensation.

Uses baseline stats when available, otherwise inspects local JSON length
distribution. Buckets: short (1024-1536), normal (2048), long (3072-4096).
"""

from __future__ import annotations

import json
import os
from typing import Optional

_MIN_MAX_LENGTH = 128
_ALIGNMENT = 64

SHORT_BUCKET = (1024, 1536)
NORMAL_BUCKET = 2048
LONG_BUCKET = (3072, 4096)


def _align_up(value: int, alignment: int = _ALIGNMENT) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _length_stats_from_file(dataset_path: str) -> Optional[dict]:
    if not dataset_path or not os.path.isfile(dataset_path):
        return None
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or not data:
            return None
        lengths = []
        sample = data[: min(500, len(data))]
        for row in sample:
            if isinstance(row, dict):
                lengths.append(sum(len(str(v)) for v in row.values()) // 4)
        if not lengths:
            return None
        lengths.sort()
        n = len(lengths)
        return {
            "p50": lengths[n // 2],
            "p95": lengths[int(n * 0.95)],
            "p99": lengths[int(n * 0.99)],
            "max": lengths[-1],
            "mean": sum(lengths) // n,
        }
    except Exception:
        return None


def classify_length_bucket(p99: int) -> str:
    if p99 <= 900:
        return "short"
    if p99 <= 2200:
        return "normal"
    return "long"


def bucket_max_length(
    bucket: str,
    model_max_length: Optional[int] = None,
) -> int:
    if bucket == "short":
        target = SHORT_BUCKET[1] if model_max_length and model_max_length >= SHORT_BUCKET[1] else SHORT_BUCKET[0]
    elif bucket == "long":
        ceiling = LONG_BUCKET[1]
        if model_max_length:
            ceiling = min(ceiling, model_max_length)
        target = ceiling
    else:
        target = NORMAL_BUCKET
        if model_max_length:
            target = min(target, model_max_length)
    return _align_up(max(_MIN_MAX_LENGTH, target))


def scale_batch_for_max_length(batch_size: int, max_length: int, reference: int = 2048) -> int:
    if max_length <= reference:
        return batch_size
    if max_length <= 4096:
        return max(1, batch_size // 2)
    return max(1, batch_size // 4)


def compute_max_length(
    seq_length_distribution: Optional[dict],
    default: int = 2048,
    packing: bool = True,
    model_max_length: Optional[int] = None,
    dataset_path: Optional[str] = None,
) -> int:
    dist = seq_length_distribution
    if dist is None and dataset_path:
        dist = _length_stats_from_file(dataset_path)

    if dist is None:
        return default

    p99 = dist.get("p99")
    p50 = dist.get("p50")
    if p99 is None or p99 <= 0:
        return default

    bucket = classify_length_bucket(int(p99))
    target = bucket_max_length(bucket, model_max_length)

    if packing and p50:
        target = max(target, _align_up(int(p99 * 1.05)))
        if bucket == "short":
            target = min(target, SHORT_BUCKET[1])
        elif bucket == "normal":
            target = min(target, NORMAL_BUCKET)
        else:
            ceiling = LONG_BUCKET[1]
            if model_max_length:
                ceiling = min(ceiling, model_max_length)
            target = min(target, ceiling)

    target = max(_MIN_MAX_LENGTH, target)
    print(
        f"[adaptive_max_length] bucket={bucket} max_length={target} "
        f"(p99={p99}, p50={p50}, default={default})",
        flush=True,
    )
    return target


def compute_prompt_length(max_length: int, ratio: float = 0.6) -> int:
    return _align_up(max(128, min(int(max_length * ratio), max_length - 128)))
