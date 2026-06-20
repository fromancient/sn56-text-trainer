from model_utility import get_model_architecture, get_model_num_params, get_use_liger, disable_flash_attention, get_gradient_checkpointing, get_gpu_count
from copy import deepcopy
from lr_finder import estimate_starting_lr
from adaptive_max_length import compute_max_length, compute_prompt_length, scale_batch_for_max_length
from strategy_router import build_runtime_profile, apply_family_rules, resolve_size_bucket
from task_diagnosis import diagnose_task

DPO_CONFIG = {
    "sub_1b": {"lr": 3e-5, "distributed": "ddp", "gpu_count": 1, "batch_size": 10, "use_lora": False},
    "1_2b": {"lr": 2e-5, "distributed": "ddp", "gpu_count": 1, "batch_size": 8},
    "2_4b": {"lr": 1.2e-5, "distributed": "ddp", "gpu_count": 2, "batch_size": 6, "use_lora": True},
    "4_9b": {"lr": 8e-6, "distributed": "ddp", "gpu_count": 2, "batch_size": 5, "use_lora": True},
    "9_12b": {"lr": 6e-6, "distributed": "ds", "gpu_count": 4, "use_lora": True, "batch_size": 8},
    "12_40b": {"lr": 5e-6, "distributed": "ds", "gpu_count": 8, "use_lora": True, "batch_size": 4},
    "40_80b": {"lr": 5e-6, "distributed": "ds", "gpu_count": 8, "use_lora": True, "batch_size": 2},
}

for key in DPO_CONFIG:
    DPO_CONFIG[key]["label"] = key
    

def get_config(param_nums: int, profile: dict | None = None) -> dict:
    bucket = (profile or {}).get("size_bucket") or resolve_size_bucket(param_nums or 4_000_000_000)
    result = deepcopy(
        DPO_CONFIG.get(
            bucket,
            {"lr": 4e-6, "distributed": "ds", "gpu_count": 8, "batch_size": 2, "use_lora": True},
        )
    )
    if profile:
        if profile.get("use_lora"):
            result["use_lora"] = True
        result["distributed"] = profile.get("distributed", result.get("distributed", "ddp"))
    return result


def get_run_cmd(config: dict, gpu_nums: int):
    required_keys = [
        "epoch_num",
        "batch_size",
        "learning_rate",
        "min_lr_rate",
        "use_liger",
        "optimizer",
        "disable_fa",
        "beta",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Required key {key} not found in config")
    gpu_nums = get_gpu_count()
    start_cmd = "python"
    run_type = config.get("distributed", "ddp")
    if gpu_nums > 1 and run_type == "ddp":
        start_cmd = f"torchrun --nproc_per_node={gpu_nums}"
    elif run_type == "ds":
        start_cmd = f"deepspeed"

    template = (
        start_cmd
        + """ train_dpo.py \
    --request_path {request_path} \
    --bf16 True \
    --report_to wandb \
    --output_dir {output_dir} \
    --num_train_epochs {epoch_num} \
    --per_device_train_batch_size {batch_size} \
    --per_device_eval_batch_size {batch_size} \
    --gradient_accumulation_steps {gradient_accumulation_steps} \
    --eval_accumulation_steps 1 \
    --eval_strategy no \
    --save_strategy no \
    --logging_steps 5 \
    --learning_rate {learning_rate} \
    --beta {beta} \
    --weight_decay 0. \
    --warmup_steps {warmup_steps} \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs "{\\"min_lr_rate\\": {min_lr_rate}}" \
    --tf32 True \
    --gradient_checkpointing {gradient_checkpointing} \
    --optim {optimizer} \
    --use_liger {use_liger} --disable_fa {disable_fa} \
    --max_length {max_length} --max_prompt_length {max_prompt_length} \
    --dataloader_pin_memory True"""
    )

    if config.get("use_lora", False):
        template += (
            " --use_peft --lora_r 128 --lora_alpha 256 --lora_target_modules all-linear"
        )

    if run_type == "ds":
        template = template + """ --deepspeed ds_config/zero3.json"""

    for key, value in config.items():
        template = template.replace("{" + key + "}", str(value))
    
    if config.get("use_attn_implementation", ""):
        use_attn_implementation = config["use_attn_implementation"]
        template = template + f""" --use_attn_implementation {use_attn_implementation}"""
        
    return template


def get_training_json(train_info: dict) -> dict:
    model_name = train_info["model_name"]
    model_path = train_info["model_path"]
    model_architecture = get_model_architecture(model_path)
    param_nums = get_model_num_params(model_name, model_path)
    profile = build_runtime_profile(
        model_name, model_path, "DpoTask", train_info.get("hours_to_complete", 2)
    )
    diagnosis = diagnose_task(train_info, "DpoTask")
    config = get_config(param_nums, profile)

    model_max_pos = None
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_path)
        model_max_pos = getattr(cfg, "max_position_embeddings", None)
    except Exception:
        pass

    baseline_stats = train_info.get("baseline_stats")
    dataset_stats = (baseline_stats or {}).get("dataset", {})
    seq_dist = dataset_stats.get("seq_length_distribution")
    max_length = compute_max_length(
        seq_dist,
        default=1024,
        packing=False,
        model_max_length=model_max_pos,
        dataset_path=train_info.get("dataset"),
    )
    max_prompt_length = compute_prompt_length(max_length, ratio=0.65)

    warmup_steps = profile["warmup_steps"]
    _bs = max(1, config["batch_size"] // 2)
    _bs = scale_batch_for_max_length(_bs, max_length, reference=1024)
    _bs, lr = apply_family_rules(model_name, model_path, _bs, config["lr"])

    run_config = {
        "epoch_num": diagnosis["dpo_epochs"],
        "batch_size": _bs,
        "learning_rate": lr,
        "min_lr_rate": profile["min_lr_rate"],
        "warmup_steps": warmup_steps,
        "use_liger": get_use_liger(model_architecture),
        "optimizer": "paged_adamw_8bit",
        "use_lora": config.get("use_lora", profile.get("use_lora", False)),
        "disable_fa": disable_flash_attention(model_architecture, model_name),
        "gpu_nums": config["gpu_count"],
        "output_dir": train_info["output_dir"],
        "request_path": train_info["request_path"],
        "distributed": config.get("distributed", "ddp"),
        "gradient_checkpointing": get_gradient_checkpointing(model_name),
        "gradient_accumulation_steps": 2,
        "max_length": max_length,
        "max_prompt_length": max_prompt_length,
        "beta": diagnosis["dpo_beta"],
        "use_attn_implementation": "kernels-community/vllm-flash-attn3" if train_info.get("is_openai", False) else ""
    }
    
    if not config.get("gradient_checkpointing", True):
        run_config["gradient_checkpointing"] = False
    
    total_batch_size = run_config["batch_size"] * run_config["gpu_nums"]
    if total_batch_size < 64:
        run_config["gradient_accumulation_steps"] = min(4, int(64 / total_batch_size))
    
    if train_info["find_lk_lr"]:
        effective_bs = run_config["batch_size"] * run_config["gradient_accumulation_steps"] * run_config["gpu_nums"]
        lr = estimate_starting_lr(
            train_info.get("baseline_stats"),
            "DpoTask",
            param_nums,
            effective_bs,
            fallback_lr=run_config["learning_rate"],
        )
        if lr is not None:
            # DPO's risk is asymmetric: too-low LR is recoverable (the in-train
            # held-out LR search nudges it back up), too-high LR collapses the
            # policy irrecoverably. So the finder may go BELOW the hand-tuned
            # bucket LR (useful per-model adaptivity) but never ABOVE it — the
            # bucket value is the hottest LR we trust for this size.
            run_config["learning_rate"] = min(lr, config["lr"])

    run_config["learning_rate"] *= train_info["reg_ratio"]
    run_cmd = get_run_cmd(run_config, run_config["gpu_nums"])
    if run_config["disable_fa"] == "False":
        run_cmd = run_cmd + " --padding_free True"
    train_request = deepcopy(train_info)
    train_request["save_before_remaining_time"] = 3
    train_request["min_steps"] = 100
    train_request["adjust_batch_size"] = False
    train_request["periodic_save_steps"] = 500
    train_request["checking_step"] = 80
    train_request["task_diagnosis"] = diagnosis

    return {
        "train_request": train_request,
        "run_cmd": run_cmd
    }
