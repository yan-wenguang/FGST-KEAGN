from __future__ import annotations

import csv
import io
from pathlib import Path
import random

SEED = 91357
ROW_COUNT = 240
COLUMNS = ["feature_a", "feature_b", "feature_c", "context_level", "label"]
DATA_PATH = Path(__file__).with_name("synthetic.csv")


def synthetic_csv_bytes() -> bytes:
    random_source = random.Random(SEED)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(COLUMNS)
    for _ in range(ROW_COUNT):
        feature_a = random_source.uniform(-1.0, 1.0)
        feature_b = random_source.uniform(-1.0, 1.0)
        feature_c = random_source.uniform(-1.0, 1.0)
        context_level = random_source.uniform(0.0, 1.0)
        noise = random_source.uniform(-0.6, 0.6)
        score = 1.4 * feature_a - 0.9 * feature_b + 0.5 * feature_c + noise
        writer.writerow([
            f"{feature_a:.6f}",
            f"{feature_b:.6f}",
            f"{feature_c:.6f}",
            f"{context_level:.6f}",
            int(score >= 0.0),
        ])
    return buffer.getvalue().encode("utf-8")


def write_synthetic_data(path: Path = DATA_PATH) -> Path:
    content = synthetic_csv_bytes()
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError("Refusing to overwrite a different synthetic fixture.")
    else:
        with path.open("xb") as handle:
            handle.write(content)
    return path


def main() -> None:
    path = write_synthetic_data()
    print(f"Synthetic-only fixture verified: {path.name} ({ROW_COUNT} rows)")


if __name__ == "__main__":
    main()
