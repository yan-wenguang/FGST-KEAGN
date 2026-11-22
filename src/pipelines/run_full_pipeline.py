from __future__ import annotations

import argparse
from pathlib import Path

from src.trainers.stage1_trainer import train_stage1
from src.trainers.stage2_trainer import train_stage2
from src.utils.io import ensure_dir, save_json, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full forest fire pipeline.")
    parser.add_argument("--config", required=True, help="Full pipeline YAML config path.")
    args = parser.parse_args()
    config = load_yaml(args.config)

    stage1_result = train_stage1(config)
    config.setdefault("knowledge", {})
    config["knowledge"]["selected_features_path"] = str(Path(config["artifacts"]["stage1_dir"]) / "selected_features.json")
    stage2_result = train_stage2(config)

    reports_dir = ensure_dir(config["artifacts"]["reports_dir"])
    save_json(
        Path(reports_dir) / "full_pipeline_summary.json",
        {"stage1": stage1_result, "stage2": stage2_result},
    )
    print("Full pipeline completed.")
    print(f"Stage 1 selected features: {', '.join(stage1_result['selected_features'])}")
    print(f"Stage 2 test AUC: {stage2_result['test_metrics']['auc']:.4f}")


if __name__ == "__main__":
    main()

