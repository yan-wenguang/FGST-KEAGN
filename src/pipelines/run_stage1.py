from __future__ import annotations

import argparse

from src.trainers.stage1_trainer import train_stage1
from src.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FGAT-FS stage 1.")
    parser.add_argument("--config", required=True, help="Stage 1 YAML config path.")
    args = parser.parse_args()
    config = load_yaml(args.config)
    result = train_stage1(config)
    print("Stage 1 completed.")
    print(f"Output dir: {result['output_dir']}")
    print(f"Selected features: {', '.join(result['selected_features'])}")
    print(f"Subset test AUC: {result['subset_test_metrics']['auc']:.4f}")


if __name__ == "__main__":
    main()

