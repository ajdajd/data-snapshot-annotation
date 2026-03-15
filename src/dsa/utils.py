import json
from pathlib import Path


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    elif x > 1.0:
        return 1.0
    else:
        return x
