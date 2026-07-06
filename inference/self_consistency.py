from inference.common.config import DATA_DIR, RESULTS_DIR, ExperimentConfig, ModelRuntimeConfig
from inference.experiments import run_experiments


def main() -> None:
    run_experiments(
        ModelRuntimeConfig(),
        ExperimentConfig(
            experiment_name="prototype_v1",
            dataset_path=DATA_DIR / "tasks.json",
            dataset_name="prototype_v1",
            output_dir=RESULTS_DIR,
        ),
        ["self_consistency"],
    )


if __name__ == "__main__":
    main()
