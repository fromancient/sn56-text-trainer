"""
Pre-training task diagnosis — routes config by runtime signals validators expose.

Inspects task type, model size, baseline stats, dataset file, KL env vars,
DPO pair quality, and GRPO reward function shape before config selection.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from model_utility import get_model_architecture, get_model_num_params
from strategy_router import resolve_size_bucket


def _load_dataset_rows(path: str) -> list[dict]:
    if not path or not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _seq_stats(baseline_stats: Optional[dict]) -> dict[str, int]:
    if not baseline_stats:
        return {"p50": 0, "p95": 0, "p99": 0, "size": 0}
    ds = baseline_stats.get("dataset", {})
    dist = ds.get("seq_length_distribution") or {}
    return {
        "p50": int(dist.get("p50") or 0),
        "p95": int(dist.get("p95") or 0),
        "p99": int(dist.get("p99") or 0),
        "size": int(ds.get("num_rows") or ds.get("size") or 0),
    }


def _kl_env() -> tuple[bool, float]:
    use_kl = os.environ.get("USE_KL") == "1"
    coef = 0.0
    raw = os.environ.get("KL_COEF")
    if use_kl and raw:
        try:
            coef = float(raw)
        except (ValueError, TypeError):
            coef = 0.0
    return use_kl and coef > 0, coef


def _dpo_quality(rows: list[dict], dataset_type: dict) -> dict[str, Any]:
    if not rows:
        return {"identical_rate": 0.0, "empty_rate": 0.0, "is_clean": True}

    prompt_f = dataset_type.get("field_prompt", "prompt")
    chosen_f = dataset_type.get("field_chosen", "chosen")
    rejected_f = dataset_type.get("field_rejected", "rejected")

    identical = empty = 0
    for row in rows:
        prompt = str(row.get(prompt_f, row.get("prompt", ""))).strip()
        chosen = str(row.get(chosen_f, row.get("chosen", ""))).strip()
        rejected = str(row.get(rejected_f, row.get("rejected", ""))).strip()
        if not prompt or not chosen or not rejected:
            empty += 1
        elif chosen == rejected:
            identical += 1

    n = len(rows)
    identical_rate = identical / n
    empty_rate = empty / n
    return {
        "identical_rate": identical_rate,
        "empty_rate": empty_rate,
        "is_clean": identical_rate < 0.05 and empty_rate < 0.02,
    }


def _grpo_reward_profile(dataset_type: dict) -> str:
    rewards = dataset_type.get("reward_functions") or []
    if not rewards:
        return "generic"

    blob = " ".join(
        rf.get("reward_func", "") for rf in rewards if isinstance(rf, dict)
    ).lower()

    if any(k in blob for k in ("langcheck", "detoxify", "textstat")):
        return "slow_external"
    if any(k in blob for k in ("sat_reward", "ded_reward", "abd_reward")):
        return "code_execution"
    if any(k in blob for k in ("regex", "exact", "match", "startswith", "endswith")):
        return "exact_match"
    if any(k in blob for k in ("math", "sympy", "latex", "calculate")):
        return "math_reasoning"
    if len(rewards) > 1:
        return "multi_reward"
    return "generic"


def diagnose_task(train_info: dict, task_type: str) -> dict[str, Any]:
    model_name = train_info["model_name"]
    model_path = train_info["model_path"]
    param_count = get_model_num_params(model_name, model_path) or 4_000_000_000
    baseline_stats = train_info.get("baseline_stats")
    rows = _load_dataset_rows(train_info.get("dataset", ""))
    seq = _seq_stats(baseline_stats)
    dataset_size = seq["size"] or len(rows)

    use_kl, kl_coef = _kl_env()
    diagnosis: dict[str, Any] = {
        "task_type": task_type,
        "param_count": param_count,
        "size_bucket": resolve_size_bucket(param_count),
        "model_family": get_model_architecture(model_path),
        "dataset_size": dataset_size,
        "p95_length": seq["p95"],
        "p99_length": seq["p99"],
        "long_sequences": seq["p95"] > 2200 or seq["p99"] > 2800,
        "small_dataset": dataset_size < 500,
        "use_kl": use_kl,
        "kl_coef": kl_coef,
        "instruct_mode": "standard",
        "dpo_beta": 0.08,
        "dpo_epochs": 2,
        "grpo_profile": "generic",
    }

    if task_type in ("InstructTextTask", "ChatTask"):
        if use_kl:
            diagnosis["instruct_mode"] = "kl_conservative"
        elif diagnosis["small_dataset"]:
            diagnosis["instruct_mode"] = "small_data"
        elif diagnosis["long_sequences"]:
            diagnosis["instruct_mode"] = "long_context"

    elif task_type == "DpoTask":
        quality = _dpo_quality(rows, train_info.get("dataset_type", {}))
        diagnosis["dpo_quality"] = quality
        diagnosis["dpo_beta"] = 0.10 if quality["is_clean"] else 0.05
        diagnosis["dpo_epochs"] = 2 if quality["is_clean"] and dataset_size >= 300 else 1

    elif task_type == "GrpoTask":
        diagnosis["grpo_profile"] = _grpo_reward_profile(train_info.get("dataset_type", {}))

    print(
        f"[task_diagnosis] type={task_type} mode={diagnosis.get('instruct_mode')} "
        f"size={dataset_size} p95={seq['p95']} kl={use_kl} "
        f"grpo={diagnosis.get('grpo_profile')}",
        flush=True,
    )
    return diagnosis
