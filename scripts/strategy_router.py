"""
Task-type and model-family routing for SN56 text training.

Centralizes config bucket selection, LoRA/DeepSpeed decisions, and
family-specific batch/LR adjustments so each task strategy stays consistent.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from model_utility import get_model_architecture, get_model_num_params

# Size buckets (parameter count) — different boundaries than position-1 winner.
SIZE_BUCKETS = (
    (1_000_000_000, "sub_1b"),
    (2_000_000_000, "1_2b"),
    (4_000_000_000, "2_4b"),
    (9_000_000_000, "4_9b"),
    (12_000_000_000, "9_12b"),
    (40_000_000_000, "12_40b"),
    (80_000_000_000, "40_80b"),
)

FAMILY_BATCH_SCALE: dict[str, float] = {
    "gptneoxforcausallm": 0.45,
    "gptjforcausallm": 0.45,
    "phiforcausallm": 0.30,
    "falconforcausallm": 0.50,
    "bloomforcausallm": 0.35,
    "optforcausallm": 0.85,
    "llamaforcausallm": 1.0,
    "mistralforcausallm": 0.90,
    "qwen2forcausallm": 1.0,
    "qwen3forcausallm": 1.0,
    "gemma2forcausallm": 0.88,
}

FAMILY_LR_SCALE: dict[str, float] = {
    "gptneoxforcausallm": 0.85,
    "bloomforcausallm": 0.80,
    "phiforcausallm": 0.75,
    "falconforcausallm": 0.90,
    "gptossforcausallm": 0.70,
}

KNOWN_SMALL_BS = {
    "EleutherAI/gpt-neo-125m": 44,
    "EleutherAI/gpt-neo-1.3B": 32,
    "bigscience/bloom-560m": 8,
    "facebook/opt-125m": 44,
    "facebook/opt-350m": 32,
    "facebook/opt-1.3b": 34,
    "microsoft/phi-2": None,  # handled via family scale
    "microsoft/phi-1_5": None,
}


def resolve_size_bucket(param_count: int) -> str:
    if param_count is None or param_count <= 0:
        return "4_9b"
    for limit, label in SIZE_BUCKETS:
        if param_count < limit:
            return label
    return "40_80b"


def should_use_lora(param_count: int, task_type: str) -> bool:
    if task_type == "DpoTask":
        return param_count >= 2_000_000_000
    if task_type == "GrpoTask":
        return param_count >= 1_000_000_000
    # Instruct / Chat
    return param_count >= 9_000_000_000


def should_use_deepspeed(param_count: int, task_type: str) -> bool:
    if task_type == "GrpoTask":
        return param_count >= 20_000_000_000
    return param_count >= 12_000_000_000


def apply_family_rules(
    model_name: str,
    model_path: str,
    batch_size: int,
    learning_rate: float,
) -> tuple[int, float]:
    arch = get_model_architecture(model_path).strip().lower()
    bs_scale = FAMILY_BATCH_SCALE.get(arch, 1.0)
    lr_scale = FAMILY_LR_SCALE.get(arch, 1.0)

    if model_name in KNOWN_SMALL_BS and KNOWN_SMALL_BS[model_name] is not None:
        return KNOWN_SMALL_BS[model_name], learning_rate * lr_scale

    if "pythia" in model_name.lower():
        bs_scale *= 0.55
    if "mistral-7b" in model_name.lower():
        bs_scale *= 0.75
    if "bloom-560m" in model_name or "bloomz-560m" in model_name:
        return 8, learning_rate * lr_scale

    new_bs = max(1, int(batch_size * bs_scale))
    return new_bs, learning_rate * lr_scale


def route_task_strategy(task_type: str) -> str:
    if task_type in ("InstructTextTask", "ChatTask"):
        return "instruct"
    if task_type == "DpoTask":
        return "dpo"
    if task_type == "GrpoTask":
        return "grpo"
    raise ValueError(f"Unsupported task type: {task_type}")


def build_runtime_profile(
    model_name: str,
    model_path: str,
    task_type: str,
    hours_to_complete: float,
) -> dict[str, Any]:
    param_count = get_model_num_params(model_name, model_path) or 4_000_000_000
    bucket = resolve_size_bucket(param_count)
    strategy = route_task_strategy(task_type)
    use_lora = should_use_lora(param_count, task_type)
    use_ds = should_use_deepspeed(param_count, task_type)

    warmup_steps = max(10, min(180, int(hours_to_complete * 55)))

    return {
        "param_count": param_count,
        "size_bucket": bucket,
        "strategy": strategy,
        "use_lora": use_lora,
        "distributed": "ds" if use_ds else "ddp",
        "warmup_steps": warmup_steps,
        "min_lr_rate": 0.22 if strategy == "instruct" else 0.25,
    }
