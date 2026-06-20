"""Pre-download langcheck / detoxify assets for offline GRPO reward validation."""

from __future__ import annotations


def warm_slow_external_reward_deps(dataset_type: dict) -> None:
    blob = " ".join(
        rf.get("reward_func", "") for rf in dataset_type.get("reward_functions", [])
    ).lower()
    if "langcheck" not in blob:
        return
    try:
        import langcheck

        langcheck.metrics.fluency(["warmup sentence for cache."])
        print("[grpo_warmup] langcheck fluency model cached", flush=True)
    except Exception as e:
        print(f"[grpo_warmup] langcheck warmup failed: {e}", flush=True)
