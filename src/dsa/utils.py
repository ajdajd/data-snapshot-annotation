import json
from pathlib import Path
from typing import List, Tuple


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


def sanitize_bbox(b: List[float]) -> Tuple[float, float, float, float]:
    if not (isinstance(b, list) and len(b) == 4):
        raise ValueError(f"Invalid bbox: {b}")
    x1, y1, x2, y2 = map(float, b)
    return (clamp01(x1), clamp01(y1), clamp01(x2), clamp01(y2))
