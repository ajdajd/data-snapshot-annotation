"""
Label Studio export adapter -> Unified Evaluation Schema v1.3.

Converts a Label Studio JSON export (list of annotation tasks) into a
ground-truth JSON file that conforms to
``data-snapshot-eval-v1.3.schema.json``.

- Input: Label Studio export JSON file
- Output: single JSON file matching data-snapshot-eval-v1.3.schema.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dsa.constants import LABEL_MAP, ROOT
from dsa.utils import clamp01

# ----------------------------
# Paths / defaults
# ----------------------------

LS_EXPORT_JSON_PATH = (
    ROOT / "data/raw_input/project-22-at-2026-03-12-15-10-03fb0566.json"
)
OUTPUT_JSON_PATH = ROOT / "data/evaluation_input/ground_truth.json"
PDF_INPUT_DIR = "pdf_input/"


# ----------------------------
# Coordinate conversion
# ----------------------------


def _ls_rect_to_xyxy_norm(rect_value: dict[str, Any]) -> list[float]:
    """Convert a Label Studio rectangle annotation to normalized xyxy.

    Label Studio ``rectanglelabels`` use percentage units where ``x``, ``y``,
    ``width``, and ``height`` are in ``[0, 100]`` relative to the image
    dimensions.

    Parameters
    ----------
    rect_value : dict[str, Any]
        The ``"value"`` dict from a Label Studio rectangle result, containing
        ``"x"``, ``"y"``, ``"width"``, and ``"height"`` keys.

    Returns
    -------
    list[float]
        Normalized bounding box as ``[x1, y1, x2, y2]`` in ``[0, 1]``.
    """
    x = float(rect_value["x"]) / 100.0
    y = float(rect_value["y"]) / 100.0
    w = float(rect_value["width"]) / 100.0
    h = float(rect_value["height"]) / 100.0

    x1 = clamp01(x)
    y1 = clamp01(y)
    x2 = clamp01(x + w)
    y2 = clamp01(y + h)

    # Ensure strict ordering.
    eps = 1e-9
    if x2 <= x1:
        x2 = clamp01(x1 + eps)
    if y2 <= y1:
        y2 = clamp01(y1 + eps)

    return [x1, y1, x2, y2]


def _best_page_dims_for_item(
    results: list[dict[str, Any]], item_index: int
) -> tuple[int, int] | None:
    """Infer page image dimensions from rectangle results.

    Scans the annotation results for any ``rectanglelabels`` entry on the
    given page that includes ``original_width`` and ``original_height``.

    Parameters
    ----------
    results : list[dict[str, Any]]
        The ``"result"`` list from a Label Studio annotation.
    item_index : int
        Page index within the task (0-based).

    Returns
    -------
    tuple[int, int] | None
        ``(width_px, height_px)`` if found, ``None`` otherwise.
    """
    for r in results:
        if r.get("type") == "rectanglelabels" and r.get("item_index") == item_index:
            ow = r.get("original_width")
            oh = r.get("original_height")
            if isinstance(ow, int) and isinstance(oh, int) and ow > 0 and oh > 0:
                return ow, oh
    return None


# ----------------------------
# Main adapter
# ----------------------------


def convert_labelstudio_export_to_eval_v13(
    input_json_path: str | Path,
    output_json_path: str | Path,
    *,
    dataset_id: str = "labelstudio_export",
    created_at: str | None = None,
    label_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Convert a Label Studio export to Unified Evaluation Schema v1.3.

    Reads a Label Studio JSON export file (list of annotation tasks) and
    writes a ground-truth JSON file in the unified evaluation format.

    All ``rectanglelabels`` are kept, including non-canonical labels such as
    ``"For review"``.  Filtering happens at evaluation time.
    Pages with no rectangle annotations are dropped.

    Parameters
    ----------
    input_json_path : str | Path
        Path to the Label Studio export JSON file.
    output_json_path : str | Path
        Destination path for the ground-truth JSON.
    dataset_id : str
        Identifier for this dataset in the output metadata.
    created_at : str | None
        ISO 8601 timestamp override.  When ``None``, uses the latest
        ``updated_at`` from the export tasks.
    label_map : dict[str, str] | None
        Canonical label map.  Defaults to ``LABEL_MAP`` from constants.

    Returns
    -------
    dict[str, Any]
        The complete output dictionary that was written to disk.

    Raises
    ------
    ValueError
        If the input JSON is not a non-empty list of tasks.
    """
    input_json_path = Path(input_json_path)
    output_json_path = Path(output_json_path)

    with input_json_path.open("r", encoding="utf-8") as f:
        tasks = json.load(f)

    if not isinstance(tasks, list) or not tasks:
        raise ValueError(
            "Expected Label Studio export to be a non-empty list of tasks."
        )

    if label_map is None:
        label_map = dict(LABEL_MAP)

    documents: list[dict[str, str]] = []
    page_entries: list[dict[str, Any]] = []
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
            objects: list[dict[str, Any]] = []

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

            # Drop pages without any rectangle annotations.
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

    output_obj: dict[str, Any] = {
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


# ----------------------------
# CLI
# ----------------------------


def main() -> None:
    """Run the Label Studio export conversion with default paths."""
    result = convert_labelstudio_export_to_eval_v13(
        LS_EXPORT_JSON_PATH, OUTPUT_JSON_PATH
    )
    n_docs = len(result.get("documents", []))
    n_pages = len(result.get("predictions", []))
    print(f"Done. {n_docs} documents, {n_pages} pages -> {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Convert a Label Studio export JSON to a "
            "Unified Evaluation Schema v1.3 ground-truth file."
        )
    )
    parser.add_argument(
        "--input_json_path",
        type=str,
        default=str(LS_EXPORT_JSON_PATH),
        help="Path to the Label Studio export JSON file.",
    )
    parser.add_argument(
        "--output_json_path",
        type=str,
        default=str(OUTPUT_JSON_PATH),
        help="Destination path for the ground-truth JSON.",
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="labelstudio_export",
        help="Dataset identifier for the output metadata.",
    )
    args = parser.parse_args()

    result = convert_labelstudio_export_to_eval_v13(
        args.input_json_path,
        args.output_json_path,
        dataset_id=args.dataset_id,
    )
    n_docs = len(result.get("documents", []))
    n_pages = len(result.get("predictions", []))
    print(f"Done. {n_docs} documents, {n_pages} pages -> {args.output_json_path}")
