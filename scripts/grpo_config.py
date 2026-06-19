from model_utility import (
    get_model_architecture,
    get_model_num_params,
    get_use_liger,
    disable_flash_attention,
    get_use_vllm,
    get_gradient_checkpointing,
    get_gpu_count,
)
from copy import deepcopy
from adaptive_max_length import compute_max_length, scale_batch_for_max_length
from strategy_router import build_runtime_profile, resolve_size_bucket

GRPO_CONFIG = {
    "0_1_b": {
        "lr": 8e-6,
        "distributed": "ddp",
        "gpu_count": 1,
        "batch_size": 40,
        "vllm_gpu_memory_utilization": 0.4,
    },
    "1_2_b": {
        "lr": 8e-6,
        "distributed": "ddp",
        "gpu_count": 1,
        "batch_size": 40,
        "vllm_gpu_memory_utilization": 0.4,
    },
    "2_4_b": {
        "lr": 8e-6,
        "distributed": "ddp",
        "gpu_count": 2,
        "batch_size": 42,
        "vllm_gpu_memory_utilization": 0.35,
        "use_lora": True,
    },
    "4_5_b": {
        "lr": 6e-6,
        "distributed": "ddp",
        "gpu_count": 2,
        "batch_size": 42,
        "use_lora": True,
        "vllm_gpu_memory_utilization": 0.4,
    },
    "5_6_b": {
        "lr": 6e-6,
        "distributed": "ddp",
        "gpu_count": 2,
        "batch_size": 42,
        "use_lora": True,
        "vllm_gpu_memory_utilization": 0.4,
    },
    "6_9_b": {
        "lr": 6e-6,
        "distributed": "ddp",
        "gpu_count": 4,
        "batch_size": 24,
        "use_lora": True,
        "vllm_gpu_memory_utilization": 0.5,
    },
    "9_12_b": {
        "lr": 6e-6,
        "distributed": "ddp",
        "gpu_count": 4,
        "use_lora": True,
        "batch_size": 16,
        "vllm_gpu_memory_utilization": 0.6,
    },
    "12_15_b": {
        "lr": 5e-6,
        "distributed": "ddp",
        "gpu_count": 4,
        "use_lora": True,
        "batch_size": 2,
        "vllm_gpu_memory_utilization": 0.8,
    },
    "15_20_b": {
        "lr": 5e-6,
        "distributed": "ddp",
        "gpu_count": 4,
        "use_lora": True,
        "batch_size": 16,
        "vllm_gpu_memory_utilization": 0.6,
        "use_vllm": False,
    },
    "20_40_b": {
        "lr": 4e-6,
        "distributed": "ddp",
        "gpu_count": 8,
        "use_lora": True,
        "batch_size": 16,
        "vllm_gpu_memory_utilization": 0.6,
        "use_vllm": False,
        "use_4bit": True,
    },
    "40_80_b": {
        "lr": 3e-6,
        "distributed": "ddp",
        "gpu_count": 8,
        "use_lora": True,
        "batch_size": 2,
        "vllm_gpu_memory_utilization": 0.7,
        "use_vllm": False,
        "use_4bit": True,
    },
}

for key in GRPO_CONFIG:
    GRPO_CONFIG[key]["label"] = key


def if_contain_slow_reward_function(dataset_type: dict) -> bool:
    reward_functions = dataset_type["reward_functions"]
    for reward_func in reward_functions:
        func_def = reward_func["reward_func"]
        keywords = [
            "import langcheck",
            "from langcheck",
            "import detoxify",
            "from detoxify",
            "import textstat",
            "from textstat",
        ]
        if any(keyword in func_def for keyword in keywords):
            return True
    return False


def get_grpo_config(param_nums: int, profile: dict | None = None) -> dict:
    if param_nums < 1_000_000_000:
        key = "0_1_b"
    elif param_nums < 2_000_000_000:
        key = "1_2_b"
    elif param_nums < 4_000_000_000:
        key = "2_4_b"
    elif param_nums < 5_000_000_000:
        key = "4_5_b"
    elif param_nums < 6_000_000_000:
        key = "5_6_b"
    elif param_nums < 9_000_000_000:
        key = "6_9_b"
    elif param_nums < 12_000_000_000:
        key = "9_12_b"
    elif param_nums < 15_000_000_000:
        key = "12_15_b"
    elif param_nums < 20_000_000_000:
        key = "15_20_b"
    elif param_nums < 40_000_000_000:
        key = "20_40_b"
    else:
        key = "40_80_b"

    base = deepcopy(GRPO_CONFIG.get(key, GRPO_CONFIG["40_80_b"]))
    if profile and profile.get("use_lora"):
        base["use_lora"] = True
    if param_nums >= 15_000_000_000:
        base["use_vllm"] = False
        base["use_4bit"] = True
    return base


def contain_python_execution(dataset_type: dict) -> bool:
    reward_functions = dataset_type["reward_functions"]
    for reward_func in reward_functions:
        func_def = reward_func["reward_func"]
        keywords = ["sat_reward_function", "ded_reward_function", "abd_reward_function"]
        if any(keyword in func_def for keyword in keywords):
            return True
    return False


def get_run_cmd(config: dict, gpu_nums: int):
    required_keys = [
        "epoch_num",
        "batch_size",
        "learning_rate",
        "min_lr_rate",
        "use_liger",
        "optimizer",
        "vllm_gpu_memory_utilization",
        "num_generations",
        "disable_fa",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Required key {key} not found in config")

    start_cmd = "python"
    run_type = config["distributed"]
    # if gpu_nums > 1 and run_type == "ddp":
    gpu_nums = get_gpu_count()
    start_cmd = f"torchrun --nproc_per_node={gpu_nums}"
    if run_type == "ds":
        start_cmd = f"deepspeed"

    template = (
        start_cmd
        + """ train_grpo.py \
    --request_path {request_path} \
    --bf16 True \
    --report_to wandb \
    --output_dir /workspace/data/trained_model \
    --num_train_epochs {epoch_num} \
    --per_device_train_batch_size {batch_size} \
    --per_device_eval_batch_size {eval_batch_size} \
    --gradient_accumulation_steps {gradient_accumulation_steps} \
    --eval_accumulation_steps 1 \
    --eval_strategy no \
    --save_strategy no \
    --logging_steps 5 \
    --learning_rate {learning_rate} \
    --weight_decay 0. \
    --warmup_steps {warmup_steps} \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs "{\\"min_lr_rate\\": {min_lr_rate}}" \
    --tf32 True \
    --gradient_checkpointing {gradient_checkpointing} \
    --optim {optimizer} \
    --use_liger {use_liger} --num_generations {num_generations} --vllm_mode colocate --vllm_gpu_memory_utilization {vllm_gpu_memory_utilization} \
    --disable_fa {disable_fa} \
    --beta {beta} \
    --dataloader_pin_memory True"""
    )

    if config.get("use_lora", False):
        template += (
            " --use_peft --lora_r 128 --lora_alpha 256 --lora_target_modules all-linear"
        )

    if config.get("use_vllm", True):
        template += " --use_vllm True"
    else:
        template += " --use_vllm False"

    if run_type == "ds":
        template = template + """ --deepspeed ds_config/zero3.json"""

    for key, value in config.items():
        template = template.replace("{" + key + "}", str(value))

    if config.get("tensor_parallel", False):
        template = template + f" --vllm_tensor_parallel_size {gpu_nums}"

    if config.get("use_4bit", False):
        template = (
            template
            + " --load_in_4bit True --use_bnb_nested_quant True --bnb_4bit_quant_type nf4"
        )
    return template


def get_training_json(train_info: dict) -> dict:
    model_name = train_info["model_name"]
    model_path = train_info["model_path"]
    model_architecture = get_model_architecture(model_path)
    param_nums = get_model_num_params(model_name, model_path)
    profile = build_runtime_profile(
        model_name, model_path, "GrpoTask", train_info.get("hours_to_complete", 2)
    )
    config = get_grpo_config(param_nums, profile)
    warmup_steps = profile["warmup_steps"]

    run_config = {
        "epoch_num": 1,
        "batch_size": 1,
        "learning_rate": min(config["lr"], 1e-5),
        "min_lr_rate": profile["min_lr_rate"],
        "warmup_steps": warmup_steps,
        "use_liger": get_use_liger(model_architecture),
        "optimizer": "paged_adamw_8bit",
        "use_lora": True,
        "disable_fa": disable_flash_attention(model_architecture, model_name),
        "gpu_nums": config["gpu_count"],
        "output_dir": train_info["output_dir"],
        "request_path": train_info["request_path"],
        "distributed": config.get("distributed", "ddp"),
        "gradient_checkpointing": get_gradient_checkpointing(model_name),
        "gradient_accumulation_steps": 12,
        "vllm_gpu_memory_utilization": min(config.get("vllm_gpu_memory_utilization", 0.35), 0.45),
        "num_generations": 2,
        "beta": 0.08,
        "use_vllm": get_use_vllm(model_architecture, model_name, model_path),
        "tensor_parallel": config.get("tensor_parallel", False),
        "use_4bit": config.get("use_4bit", param_nums >= 12_000_000_000),
    }

    if model_name == "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5":
        run_config["use_lora"] = True

    if "starcoder" in model_name.lower():
        run_config["batch_size"] = int(run_config["batch_size"] / 1.5)

    baseline_stats = train_info.get("baseline_stats")
    dataset_stats = (baseline_stats or {}).get("dataset", {})
    prompt_dist = dataset_stats.get("prompt_length_distribution")
    max_prompt_length = compute_max_length(
        prompt_dist,
        default=1024,
        packing=False,
        dataset_path=train_info.get("dataset"),
    )
    max_completion_length = min(512, max(128, max_prompt_length // 2))
    run_config["batch_size"] = scale_batch_for_max_length(
        run_config["batch_size"], max_prompt_length + max_completion_length, 1024
    )

    train_request = deepcopy(train_info)
    train_request["save_before_remaining_time"] = 3
    train_request["min_steps"] = 80
    train_request["adjust_batch_size"] = False
    train_request["periodic_save_steps"] = 500
    train_request["max_prompt_length"] = max_prompt_length
    train_request["max_completion_length"] = max_completion_length

    if if_contain_slow_reward_function(train_info["dataset_type"]):
        train_request["save_before_remaining_time"] = 12
        if config["label"] == "0_1_b":
            run_config["batch_size"] = 8
        elif config["label"] == "1_2_b":
            run_config["batch_size"] = 10
        elif config["label"] == "2_4_b":
            run_config["batch_size"] = 16
        elif config["label"] == "4_5_b":
            run_config["batch_size"] = 16
        elif config["label"] == "5_6_b":
            run_config["batch_size"] = 16
        elif config["label"] == "6_9_b":
            run_config["batch_size"] = 16
            if (
                model_name == "unsloth/gemma-2-9b-it"
            ):  # encounter OOM error with batch_size 12
                run_config["batch_size"] = 8
        elif config["label"] == "9_12_b":
            run_config["batch_size"] = 16
        elif config["label"] == "12_15_b":
            run_config["batch_size"] = 2
        elif config["label"] == "15_20_b":
            run_config["batch_size"] = 2
        elif config["label"] == "20_40_b":
            run_config["batch_size"] = 16  # this is high because we use 4bit
        elif config["label"] == "40_80_b":
            run_config["batch_size"] = 2

        elif config["label"] == "13_15_b":
            run_config["batch_size"] = 12

    # Scale batch size with prompt length — memory scales linearly with total
    # sequence length (prompt + completion). Reference: default TRL GRPOConfig
    # uses max_prompt_length=512 and max_completion_length=512 → total=1024.
    # Same 1/S linear scaling used by DPO and instruct.
    _ref_total = 1024   # 512 prompt + 512 completion (TRL defaults)
    _default_completion = 512
    if max_prompt_length is not None:
        _actual_total = max_prompt_length + _default_completion
        if _actual_total > _ref_total:
            _scale = _ref_total / _actual_total
            _old_bs = run_config["batch_size"]
            run_config["batch_size"] = max(1, int(_old_bs * _scale))
            print(
                f"[sn56][grpo-bs] max_prompt_length={max_prompt_length} > 512, "
                f"batch {_old_bs} -> {run_config['batch_size']}",
                flush=True,
            )

    total_batch_size = run_config["batch_size"] * run_config["gpu_nums"]
    if total_batch_size < 32:
        run_config["gradient_accumulation_steps"] = max(8, int(32 / max(total_batch_size, 1)))

    run_config["eval_batch_size"] = 4
    if run_config["batch_size"] <= 4:
        run_config["eval_batch_size"] = 2

    if not config.get("use_vllm", True):
        run_config["use_vllm"] = False

    run_config["learning_rate"] *= train_info["reg_ratio"]

    run_cmd = get_run_cmd(run_config, run_config["gpu_nums"])

    return {"train_request": train_request, "run_cmd": run_cmd}
