"""
Filter raw DPO JSON before tokenization to avoid random-preference training.

Removes empty prompt/chosen/rejected rows, identical chosen==rejected pairs,
and weak preference pairs that teach the wrong length or near-random signal.
"""

from __future__ import annotations

import json
import os
from typing import Any


def _word_set_similarity(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def filter_dpo_dataset(dataset_path: str, dataset_type: dict) -> str:
    if not dataset_path or not os.path.isfile(dataset_path):
        return dataset_path

    with open(dataset_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        return dataset_path

    prompt_f = dataset_type.get("field_prompt", "prompt")
    chosen_f = dataset_type.get("field_chosen", "chosen")
    rejected_f = dataset_type.get("field_rejected", "rejected")

    kept = []
    dropped_empty = dropped_identical = dropped_weak = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        prompt = str(row.get(prompt_f, row.get("prompt", ""))).strip()
        chosen = str(row.get(chosen_f, row.get("chosen", ""))).strip()
        rejected = str(row.get(rejected_f, row.get("rejected", ""))).strip()
        if not prompt or not chosen or not rejected:
            dropped_empty += 1
            continue
        if chosen == rejected:
            dropped_identical += 1
            continue
        if len(chosen) < 8 or len(rejected) < 8:
            dropped_weak += 1
            continue
        if _word_set_similarity(chosen, rejected) > 0.92:
            dropped_weak += 1
            continue
        if len(rejected) > 3 * len(chosen) and len(chosen) < 80:
            dropped_weak += 1
            continue
        kept.append(row)

    if dropped_empty == 0 and dropped_identical == 0 and dropped_weak == 0:
        return dataset_path

    out_name = os.path.basename(dataset_path).replace(
        "_train_data.json", "_filtered_train_data.json"
    )
    out_path = os.path.join("/tmp", out_name)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False)

    print(
        f"[dpo_filter] {len(rows)} -> {len(kept)} "
        f"(empty={dropped_empty}, identical={dropped_identical}, weak={dropped_weak})",
        flush=True,
    )
    return out_path
