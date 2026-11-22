from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: str | Path, default: Any | None = None) -> Any:
    target = Path(path)
    if not target.exists():
        return {} if default is None else default
    with target.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: Any, indent: int = 2) -> None:
    serialized = json.dumps(payload, indent=indent, ensure_ascii=False, allow_nan=False)
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(serialized + "\n", encoding="utf-8")


def save_dataframe(path: str | Path, frame: pd.DataFrame) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    frame.to_csv(target, index=False)
