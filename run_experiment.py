from __future__ import annotations

import argparse
from pathlib import Path

from inference.common.config import DATA_DIR, RESULTS_DIR, ExperimentConfig, ModelRuntimeConfig
from inference.experiments import run_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local LLM inference-time scaling experiments.")
    parser.add_argument("--experiment-name", default="v2_benchmark")
    parser.add_argument("--dataset-path", type=Path, default=DATA_DIR / "tasks.json")
    parser.add_argument("--dataset-name", default="prototype_v1")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["baseline", "chain_of_thought", "adaptive", "self_consistency"],
        choices=["baseline", "chain_of_thought", "adaptive", "self_consistency"],
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "v2")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--backend",
        default=None,
        choices=["local_transformers", "huggingface_api"],
        help="Inference backend to use for this run.",
    )
    parser.add_argument(
        "--api-model-name",
        default=None,
        help="Hosted Hugging Face model id to call when using the huggingface_api backend.",
    )
    parser.add_argument(
        "--api-provider",
        default=None,
        help="Hosted Hugging Face provider to use for API inference, for example 'auto' or 'hf-inference'.",
    )
    parser.add_argument(
        "--api-task-mode",
        default=None,
        choices=["auto", "text_generation", "chat_completion"],
        help="Hosted Hugging Face task mode to prefer. 'auto' will fall back between text generation and chat completion.",
    )
    parser.add_argument(
        "--endpoint-url",
        default=None,
        help="Dedicated Hugging Face Inference Endpoint URL. When set, provider routing is bypassed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_config = ModelRuntimeConfig()
    if args.model_name:
        runtime_config.model_name = args.model_name
    if args.backend:
        runtime_config.backend = args.backend
    if args.api_model_name:
        runtime_config.api_model_name = args.api_model_name
    if args.api_provider:
        runtime_config.api_provider = args.api_provider
    if args.api_task_mode:
        runtime_config.api_task_mode = args.api_task_mode
    if args.endpoint_url:
        runtime_config.endpoint_url = args.endpoint_url

    experiment_config = ExperimentConfig(
        experiment_name=args.experiment_name,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        max_tasks=args.max_tasks,
    )
    run_experiments(runtime_config, experiment_config, args.strategies)


if __name__ == "__main__":
    main()
