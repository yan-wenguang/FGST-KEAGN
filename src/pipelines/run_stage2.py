from __future__ import annotations

import argparse

from src.trainers.stage2_trainer import train_stage2
from src.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM-KEAGN stage 2.")
    parser.add_argument("--config", required=True, help="Stage 2 YAML config path.")
    args = parser.parse_args()
    config = load_yaml(args.config)
    result = train_stage2(config)
    print("Stage 2 completed.")
    print(f"Output dir: {result['output_dir']}")
    print(f"Test AUC: {result['test_metrics']['auc']:.4f}")


if __name__ == "__main__":
    main()

