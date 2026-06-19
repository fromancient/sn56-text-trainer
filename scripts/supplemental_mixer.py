"""
Blend whitelisted supplemental SFT data into tournament datasets when appropriate.

Requested datasets are mounted read-only under MINER_DATASETS_DIR. This module
only mixes when the tournament dataset profile suggests code/tool/reasoning or
when the primary dataset is very small.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

CODE_TOOL_KEYWORDS = (
    "code",
    "python",
    "tool",
    "function",
    "api",
    "sql",
    "bash",
    "json",
    "reasoning",
    "math",
    "intercode",
    "pvp",
)

GENERIC_INSTRUCTION_KEYWORDS = (
    "instruct",
    "instruction",
    "general",
    "chat",
    "assistant",
    "helpful",
)


def _load_json_records(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return []


def _dataset_profile(records: list[dict[str, Any]], sample_n: int = 40) -> dict[str, Any]:
    if not records:
        return {"size": 0, "avg_chars": 0, "code_like_ratio": 0.0, "generic_ratio": 0.0}

    sample = records[:sample_n]
    total_chars = 0
    code_hits = 0
    generic_hits = 0
    for row in sample:
        blob = " ".join(str(v) for v in row.values()).lower()
        total_chars += len(blob)
        if any(k in blob for k in CODE_TOOL_KEYWORDS):
            code_hits += 1
        if any(k in blob for k in GENERIC_INSTRUCTION_KEYWORDS):
            generic_hits += 1

    n = len(sample)
    return {
        "size": len(records),
        "avg_chars": total_chars // max(n, 1),
        "code_like_ratio": code_hits / n,
        "generic_ratio": generic_hits / n,
    }


def _supplemental_paths() -> list[str]:
    root = os.environ.get("MINER_DATASETS_DIR")
    names = [n.strip() for n in os.environ.get("MINER_DATASETS", "").split(",") if n.strip()]
    if not root:
        return []
    paths = []
    for name in names:
        candidate = Path(root) / name
        if candidate.is_dir():
            for fname in ("train.json", "data.json", "dataset.json"):
                p = candidate / fname
                if p.exists():
                    paths.append(str(p))
                    break
            else:
                json_files = list(candidate.glob("*.json"))
                if json_files:
                    paths.append(str(json_files[0]))
        elif candidate.is_file():
            paths.append(str(candidate))
    return paths


def _mix_ratio(profile: dict[str, Any]) -> float:
    size = profile["size"]
    if profile["generic_ratio"] > 0.6 and profile["code_like_ratio"] < 0.15:
        return 0.0
    if size >= 8000:
        return 0.0
    if profile["code_like_ratio"] >= 0.25:
        return 0.12
    if size < 500:
        return 0.08
    if profile["code_like_ratio"] >= 0.10:
        return 0.06
    return 0.0


def maybe_blend_supplemental(primary_path: str, task_id: str) -> str:
    """Return path to dataset (possibly blended). Writes a new file beside primary."""
    supplemental = _supplemental_paths()
    if not supplemental:
        return primary_path

    primary_records = _load_json_records(primary_path)
    profile = _dataset_profile(primary_records)
    ratio = _mix_ratio(profile)
    if ratio <= 0:
        print(
            f"[supplemental] skip mix size={profile['size']} "
            f"code_like={profile['code_like_ratio']:.2f}",
            flush=True,
        )
        return primary_path

    extra: list[dict[str, Any]] = []
    for sp in supplemental:
        try:
            extra.extend(_load_json_records(sp))
        except Exception as e:
            print(f"[supplemental] failed to load {sp}: {e}", flush=True)

    if not extra:
        return primary_path

    take = max(1, int(len(primary_records) * ratio))
    take = min(take, len(extra))
    random.seed(hash(task_id) % (2**32))
    picked = random.sample(extra, take)
    blended = primary_records + picked
    random.shuffle(blended)

    out_path = str(Path(primary_path).with_name(f"{task_id}_blended_train_data.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(blended, f, ensure_ascii=False)

    print(
        f"[supplemental] blended {take} examples ({ratio:.0%}) -> {out_path}",
        flush=True,
    )
    return out_path
