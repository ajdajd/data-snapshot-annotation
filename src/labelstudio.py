import json
from pathlib import Path
from src.constants import ROOT
from typing import Any, Dict, List, Optional, Tuple

# TODO: Move to constants.py
INPUT_JSON_PATH = ROOT / "data/raw_input/project-19-at-2026-02-19-14-26-9aaf565b.json"
OUTPUT_JSON_PATH = ROOT / "data/evaluation_input/ground_truth.json"
PDF_INPUT_DIR = "pdf_input/"


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _ls_rect_to_xyxy_norm(rect_value: Dict[str, Any]) -> List[float]:
    """
    Label Studio rectanglelabels use percentage units:
      x,y,width,height are in [0,100] relative to image width/height.

    Convert to normalized_xyxy in [0,1]:
      [x1, y1, x2, y2]
    """
    x = float(rect_value["x"]) / 100.0
    y = float(rect_value["y"]) / 100.0
    w = float(rect_value["width"]) / 100.0
    h = float(rect_value["height"]) / 100.0

    x1 = _clamp01(x)
    y1 = _clamp01(y)
    x2 = _clamp01(x + w)
    y2 = _clamp01(y + h)

    # Ensure strict ordering
    eps = 1e-9
    if x2 <= x1:
        x2 = _clamp01(x1 + eps)
    if y2 <= y1:
        y2 = _clamp01(y1 + eps)

    return [x1, y1, x2, y2]


def _best_page_dims_for_item(
    results: List[Dict[str, Any]], item_index: int
) -> Optional[Tuple[int, int]]:
    """Infer page image dimensions from any rectangle result on the page."""
    for r in results:
        if r.get("type") == "rectanglelabels" and r.get("item_index") == item_index:
            ow = r.get("original_width")
            oh = r.get("original_height")
            if isinstance(ow, int) and isinstance(oh, int) and ow > 0 and oh > 0:
                return ow, oh
    return None


def convert_labelstudio_export_to_eval_v13(
    input_json_path: str | Path,
    output_json_path: str | Path,
    *,
    dataset_id: str = "labelstudio_export",
    created_at: Optional[str] = None,
    label_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Convert a Label Studio export JSON file (list of tasks) into the unified evaluation
    schema v1.3 JSON format and save it to output_json_path.

    - Keep all rectanglelabels, including non-supported labels like "For review".
      (Filtering happens at evaluation time.)
    - Drop pages that have no rectangle annotations (any label).
    """
    input_json_path = Path(input_json_path)
    output_json_path = Path(output_json_path)

    with input_json_path.open("r", encoding="utf-8") as f:
        tasks = json.load(f)

    if not isinstance(tasks, list) or not tasks:
        raise ValueError(
            "Expected Label Studio export to be a non-empty list of tasks."
        )

    # Keep canonical label_map unless caller overrides it.
    # We still include it top-level because v1.3 expects it, even if export contains extra labels.
    if label_map is None:
        label_map = {"1": "Figure", "2": "Table"}

    documents: List[Dict[str, str]] = []
    page_entries: List[Dict[str, Any]] = []
    seen_docs: set[str] = set()

    for task in tasks:
        meta = task.get("meta") or {}
        doc_name = meta.get("file")
        doc_id = str(doc_name)
        doc_path = PDF_INPUT_DIR + doc_name

        if doc_id not in seen_docs:
            documents.append(
                {"doc_id": doc_id, "doc_name": doc_name, "doc_path": doc_path}
            )
            seen_docs.add(doc_id)

        data = task.get("data") or {}
        pages = data.get("pages") or []
        if not isinstance(pages, list) or not pages:
            continue

        annotations = task.get("annotations") or []
        usable_annotations = [
            a for a in annotations if not a.get("was_cancelled", False)
        ]
        if not usable_annotations:
            continue

        chosen_ann = max(
            usable_annotations,
            key=lambda a: a.get("updated_at") or a.get("created_at") or "",
        )

        results = chosen_ann.get("result") or []
        if not isinstance(results, list) or not results:
            continue

        for page_index, page_path in enumerate(pages):
            objects: List[Dict[str, Any]] = []

            for r in results:
                if r.get("type") != "rectanglelabels":
                    continue
                if r.get("item_index") != page_index:
                    continue

                rect_value = r.get("value") or {}
                rect_labels = rect_value.get("rectanglelabels") or []
                if not rect_labels:
                    continue

                label = str(rect_labels[0])
                bbox = _ls_rect_to_xyxy_norm(rect_value)

                objects.append(
                    {
                        "id": str(
                            r.get("id") or f"{doc_id}::{page_index}::{len(objects)}"
                        ),
                        "label": label,
                        "bbox": bbox,
                    }
                )

            # Drop pages without ANY rectangle annotations (any label)
            if not objects:
                continue

            page_id = f"{doc_id}::p{page_index:03d}"

            dims = _best_page_dims_for_item(results, page_index)
            if dims is None:
                width_px, height_px = 1, 1
            else:
                width_px, height_px = dims

            page_entries.append(
                {
                    "page_id": page_id,
                    "doc_id": doc_id,
                    "page_index": int(page_index),
                    "image": {
                        "width_px": int(width_px),
                        "height_px": int(height_px),
                        "path": str(page_path),
                    },
                    "objects": objects,
                }
            )

    if created_at is None:
        latest = ""
        for t in tasks:
            u = t.get("updated_at") or ""
            if isinstance(u, str) and u > latest:
                latest = u
        created_at = latest or "unknown"

    output_obj: Dict[str, Any] = {
        "label_map": label_map,
        "info": {
            "schema_version": "1.3",
            "type": "ground_truth",
            "dataset_id": dataset_id,
            "created_at": created_at,
            "coordinate_system": {
                "type": "normalized_xyxy",
                "range": [0.0, 1.0],
                "origin": "top_left",
            },
        },
        "documents": documents,
        "predictions": page_entries,
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(output_obj, f, ensure_ascii=False, indent=4)

    return output_obj


def main():
    convert_labelstudio_export_to_eval_v13(INPUT_JSON_PATH, OUTPUT_JSON_PATH)


if __name__ == "__main__":
    main()
