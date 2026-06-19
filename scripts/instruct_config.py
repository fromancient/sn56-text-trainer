from model_utility import (
    get_model_architecture,
    get_model_num_params,
    get_use_liger,
    disable_flash_attention,
    get_gpu_count,
)
from copy import deepcopy
from lr_finder import estimate_starting_lr
from adaptive_max_length import compute_max_length, scale_batch_for_max_length
from strategy_router import build_runtime_profile, apply_family_rules


FIXED_BS_CONFIG = {
    "EleutherAI/gpt-neo-1.3B": {"batch_size": 36},
    "EleutherAI/gpt-neo-125m": {"batch_size": 48},
    "bigscience/bloom-560m": {"batch_size": 10},
    "facebook/opt-1.3b": {"batch_size": 38},
    "facebook/opt-350m": {"batch_size": 36},
    "facebook/opt-125m": {"batch_size": 48},
}

INSTRUCT_CONFIG = {
    "sub_1b": {
        "lr": 1e-4,
        "distributed": "ddp",
        "gpu_count": 1,
        "batch_size": 120,
        "use_lora": False,
    },
    "1_2b": {
        "lr": 9e-5,
        "distributed": "ddp",
        "gpu_count": 1,
        "use_lora": False,
        "batch_size": 88,
    },
    "2_4b": {
        "lr": 8e-5,
        "distributed": "ddp",
        "gpu_count": 1,
        "batch_size": 44,
    },
    "4_9b": {
        "lr": 5e-5,
        "distributed": "ddp",
        "gpu_count": 2,
        "batch_size": 26,
    },
    "9_12b": {
        "lr": 7e-5,
        "distributed": "ddp",
        "gpu_count": 2,
        "use_lora": True,
        "batch_size": 20,
    },
    "12_40b": {
        "lr": 6e-5,
        "distributed": "ds",
        "gpu_count": 4,
        "use_lora": True,
        "batch_size": 14,
    },
    "40_80b": {
        "lr": 5e-5,
        "distributed": "ds",
        "gpu_count": 8,
        "use_lora": True,
        "batch_size": 6,
    },
}

for key in INSTRUCT_CONFIG:
    INSTRUCT_CONFIG[key]["label"] = key


def get_instruct_config(param_nums: int, profile: dict | None = None) -> dict:
    from strategy_router import resolve_size_bucket

    bucket = (profile or {}).get("size_bucket") or resolve_size_bucket(param_nums or 4_000_000_000)
    result = deepcopy(
        INSTRUCT_CONFIG.get(
            bucket,
            {
                "lr": 4e-5,
                "distributed": "ds",
                "gpu_count": 8,
                "batch_size": 4,
                "use_lora": True,
            },
        )
    )
    if profile:
        result["use_lora"] = profile.get("use_lora", result.get("use_lora", False))
        result["distributed"] = profile.get("distributed", result.get("distributed", "ddp"))
    if 4_000_000_000 <= param_nums < 5_000_000_000:
        result["batch_size"] = int(result["batch_size"] * 1.15)
    return result


def get_run_cmd(config: dict, gpu_nums: int):
    required_keys = [
        "epoch_num",
        "batch_size",
        "learning_rate",
        "min_lr_rate",
        "use_liger",
        "optimizer",
        "use_lora",
        "packing",
        "disable_fa",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Required key {key} not found in config")

    gpu_nums = get_gpu_count()
    start_cmd = "python"
    run_type = config["distributed"]
    if gpu_nums > 1 and run_type == "ddp":
        start_cmd = f"torchrun --nproc_per_node={gpu_nums}"
    elif run_type == "ds":
        start_cmd = f"deepspeed"

    template = (
        start_cmd
        + """ train_instruct.py \
    --request_path {request_path} \
    --bf16 True \
    --report_to wandb \
    --output_dir {output_dir} \
    --num_train_epochs {epoch_num} \
    --per_device_train_batch_size {batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps {gradient_accumulation_steps} \
    --eval_accumulation_steps 1 \
    --eval_strategy no \
    --save_strategy epoch \
    --logging_steps 5 \
    --learning_rate {learning_rate} \
    --weight_decay 0. \
    --warmup_steps {warmup_steps} \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs "{\\"min_lr_rate\\": {min_lr_rate}}" \
    --tf32 True \
    --gradient_checkpointing {gradient_checkpointing} \
    --optim {optimizer} \
    --use_liger {use_liger} \
    --packing {packing} --disable_fa {disable_fa} \
    --dataloader_pin_memory True"""
    )
    if run_type == "ds":
        template = template + """ --deepspeed ds_config/zero3.json"""

    if config["use_lora"]:
        template = template + """ --use_lora True"""

    for key, value in config.items():
        template = template.replace("{" + key + "}", str(value))

    if config.get("use_attn_implementation", ""):
        use_attn_implementation = config["use_attn_implementation"]
        template = (
            template + f""" --use_attn_implementation {use_attn_implementation}"""
        )

    return template


def get_training_json(train_info: dict) -> dict:
    model_name = train_info["model_name"]
    model_path = train_info["model_path"]
    model_architecture = get_model_architecture(model_path)
    param_nums = get_model_num_params(model_name, model_path)
    profile = build_runtime_profile(
        model_name, model_path, "InstructTextTask", train_info.get("hours_to_complete", 2)
    )
    config = get_instruct_config(param_nums, profile)
    warmup_steps = profile["warmup_steps"]

    run_config = {
        "epoch_num": 3,
        "batch_size": config["batch_size"],
        "learning_rate": config["lr"],
        "min_lr_rate": profile["min_lr_rate"],
        "warmup_steps": warmup_steps,
        "use_liger": get_use_liger(model_architecture),
        "optimizer": "paged_adamw_8bit",
        "use_lora": config.get("use_lora", False),
        "disable_fa": disable_flash_attention(model_architecture, model_name),
        "packing": "True",
        "gpu_nums": config["gpu_count"],
        "output_dir": train_info["output_dir"],
        "request_path": train_info["request_path"],
        "distributed": config.get("distributed", "ddp"),
        "gradient_checkpointing": "True",
        "gradient_accumulation_steps": 4,
        "use_attn_implementation": (
            "kernels-community/vllm-flash-attn3"
            if train_info.get("is_openai", False)
            else ""
        ),
    }

    run_config["batch_size"], run_config["learning_rate"] = apply_family_rules(
        model_name, model_path, run_config["batch_size"], run_config["learning_rate"]
    )

    if model_name in FIXED_BS_CONFIG:
        run_config["batch_size"] = FIXED_BS_CONFIG[model_name]["batch_size"]

    if run_config["disable_fa"] == "True" or model_architecture.strip().lower() in [
        "optforcausallm"
    ]:
        run_config["packing_mode"] = "naive"
    else:
        run_config["packing_mode"] = "fa"

    data_per_step = run_config["batch_size"] * run_config["gpu_nums"]
    if data_per_step >= 64:
        run_config["gradient_accumulation_steps"] = 1
    else:
        run_config["gradient_accumulation_steps"] = int(64 / data_per_step)

    if model_architecture.strip().lower() in ["gptossforcausallm"]:
        run_config["use_lora"] = False  # currently, gptoss does not support lora

    if train_info["find_lk_lr"]:
        effective_bs = run_config["batch_size"] * run_config["gradient_accumulation_steps"] * run_config["gpu_nums"]
        lr = estimate_starting_lr(
            train_info.get("baseline_stats"),
            "InstructTextTask",
            param_nums,
            effective_bs,
            fallback_lr=run_config["learning_rate"],
        )
        if lr is not None:
            run_config["learning_rate"] = lr

    baseline_stats = train_info.get("baseline_stats")
    model_max_pos = None
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(train_info["model_path"])
        model_max_pos = getattr(cfg, "max_position_embeddings", None)
    except Exception:
        pass

    dataset_stats = (baseline_stats or {}).get("dataset", {})
    seq_dist = dataset_stats.get("seq_length_distribution")
    packing_enabled = run_config["packing"] == "True"
    default_max = 2048
    max_length = compute_max_length(
        seq_dist,
        default=default_max,
        packing=packing_enabled,
        model_max_length=model_max_pos,
        dataset_path=train_info.get("dataset"),
    )
    old_bs = run_config["batch_size"]
    run_config["batch_size"] = scale_batch_for_max_length(old_bs, max_length, default_max)
    if run_config["batch_size"] != old_bs:
        print(
            f"[instruct_config] max_length={max_length}, batch_size {old_bs} -> {run_config['batch_size']}",
            flush=True,
        )

    run_config["learning_rate"] *= train_info["reg_ratio"]
    run_cmd = get_run_cmd(run_config, run_config["gpu_nums"])
    train_request = deepcopy(train_info)
    train_request["save_before_remaining_time"] = 3
    train_request["adjust_batch_size"] = False
    train_request["periodic_save_steps"] = 500
    train_request["checking_step"] = 70
    if max_length is not None:
        train_request["max_length"] = max_length
    train_request["packing_mode"] = run_config.get("packing_mode", "fa")
        train_request["min_steps"] = max(
            int(train_info["hours_to_complete"] * 100), train_request["min_steps"]
        )

    elif param_nums < 9_000_000_000:
        train_request["min_steps"] = max(
            int(train_info["hours_to_complete"] * 70), train_request["min_steps"]
        )

    return {"train_request": train_request, "run_cmd": run_cmd}
