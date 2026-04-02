"""
Create Label Studio annotation tasks with prelabeled predictions from PDF files.

Converts each page of each PDF into a PNG image, then generates a
``tasks_prelabeled.json`` file suitable for importing into Label Studio's
multi-page document annotation workflow with bounding-box predictions
pre-populated from a model prediction JSON file.

Usage::

    python create_tasks_with_prelabeling.py \\
        --input_dir=pdf_input/ \\
        --dataset_name=dataset \\
        --pred_json=data/evaluation_input/DocLayout-YOLO-DocStructBench.json
"""

import argparse
from dotenv import load_dotenv
import json
import uuid
import os
from pathlib import Path

from label_studio_sdk import LabelStudio
from pdf2image import convert_from_path
from tqdm.auto import tqdm

from dsa.constants import ROOT
from dsa.utils import load_json

load_dotenv()
API_KEY = os.getenv("LABELSTUDIO_API_KEY")
LS_BASE_URL = "http://localhost:8080"
BASE_HOST_PATH = "/data/local-files/?d="
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = "labelstudio_data/"


def _bbox_to_ls_result(
    bbox: list[float],
    label: str,
    score: float,
    page_index: int,
) -> dict:
    """Convert a normalized XYXY bounding box to a Label Studio result item.

    Converts from ``[x1, y1, x2, y2]`` normalized coordinates (range [0, 1])
    to Label Studio's percentage-based ``x``, ``y``, ``width``, ``height``
    format, targeting the correct page image via ``item_index``.

    Parameters
    ----------
    bbox : list[float]
        Bounding box in normalized XYXY format: ``[x1, y1, x2, y2]``.
    label : str
        Label name (e.g. ``"Figure"`` or ``"Table"``).
    score : float
        Confidence score in ``[0, 1]``.
    page_index : int
        Zero-based index of the page within the multi-image task,
        used to set ``item_index`` so the annotation targets the correct image.

    Returns
    -------
    dict
        A Label Studio result item dict suitable for inclusion in a
        ``predictions[].result`` list.
    """
    x1, y1, x2, y2 = bbox
    return {
        "id": str(uuid.uuid4())[:8],
        "type": "rectanglelabels",
        "from_name": "rectangles",
        "to_name": "pdf",
        "item_index": page_index,
        "score": score,
        "value": {
            "x": x1 * 100,
            "y": y1 * 100,
            "width": (x2 - x1) * 100,
            "height": (y2 - y1) * 100,
            "rotation": 0,
            "rectanglelabels": [label],
        },
    }


def _build_prediction_index(pred_json: dict) -> dict[str, list[dict]]:
    """Build a lookup from ``doc_id`` to per-page prediction objects.

    Parameters
    ----------
    pred_json : dict
        Loaded prediction JSON conforming to the Unified Evaluation
        Schema v1.3.

    Returns
    -------
    dict[str, list[dict]]
        Mapping of ``doc_id`` → sorted list of page-entry dicts, each
        containing ``page_index`` and ``objects``.
    """
    index: dict[str, list[dict]] = {}
    for page_entry in pred_json.get("predictions", []):
        doc_id = page_entry["doc_id"]
        index.setdefault(doc_id, []).append(page_entry)

    # Sort pages within each document by page_index
    for doc_id in index:
        index[doc_id].sort(key=lambda p: p["page_index"])

    return index


def _build_ls_predictions(
    page_entries: list[dict],
) -> list[dict]:
    """Construct a Label Studio prediction block for a single task (document).

    Iterates over all page entries for the document and converts each
    object's bounding box into a Label Studio result item. The ``item_index``
    on each result item identifies which page image it belongs to.

    Parameters
    ----------
    page_entries : list[dict]
        List of page-entry dicts for a single document, sorted by
        ``page_index``. Each entry must have ``page_index`` and ``objects``
        keys matching the Unified Evaluation Schema v1.3.

    Returns
    -------
    list[dict]
        A list containing a single Label Studio prediction dict with
        all result items collected across all pages. Returns an empty
        list if there are no objects across any page.
    """
    results: list[dict] = []
    total_score = 0.0
    n_objects = 0

    for page_entry in page_entries:
        page_index: int = page_entry["page_index"]
        for obj in page_entry.get("objects", []):
            result_item = _bbox_to_ls_result(
                bbox=obj["bbox"],
                label=obj["label"],
                score=obj.get("score", 0.0),
                page_index=page_index,
            )
            results.append(result_item)
            total_score += obj.get("score", 0.0)
            n_objects += 1

    if not results:
        return []

    avg_score = total_score / n_objects if n_objects > 0 else 0.0
    return [{"result": results, "score": avg_score}]


def _create_project(project_name):
    # Initialize the client
    client = LabelStudio(base_url=LS_BASE_URL, api_key=API_KEY)

    # Load labeling template
    with open(ROOT / "labeling_interface_template.xml", "r", encoding="utf-8") as f:
        label_config = f.read()

    # Create the project
    project = client.projects.create(title=project_name, label_config=label_config)
    print(f"Project created with ID: {project.id}")

    return project.id


def _create_tasks(project_id, task_json_path, dataset_name):
    # Initialize the client
    client = LabelStudio(base_url=LS_BASE_URL, api_key=API_KEY)

    # Import tasks
    payload = load_json(task_json_path)
    client.projects.import_tasks(id=project_id, request=payload)

    # Create local storage
    path = f"/label-studio/data/{dataset_name}"
    client.import_storage.local.create(
        path=path, project=project_id, title=dataset_name
    )

    print(f"Succesfully imported tasks and set up local storage.")

    return None


def main(
    project_name: str,
    dataset_name: str,
    input_pdf_dir: str | Path,
    pred_json_path: str | Path,
) -> None:
    """Convert PDFs to page images and create a prelabeled Label Studio task file.

    Each PDF page is rendered at 300 DPI and saved as a PNG file.
    A ``tasks_prelabeled.json`` file is generated containing references to
    all pages, grouped by source PDF, with model predictions embedded as
    Label Studio prelabels.

    Pages whose ``doc_id`` does not appear in the prediction file are still
    included in the output but will have no prelabeled annotations.

    Parameters
    ----------
    input_dir : str | Path
        Directory containing PDF files to process.
    dataset_name : str
        Name for the output dataset directory under ``labelstudio_data/``.
    pred_json_path : str | Path
        Path to the prediction JSON file conforming to the Unified
        Evaluation Schema v1.3.
    """
    dataset_dir = Path(LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT) / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Build prediction index from pred json file
    pred_json = load_json(pred_json_path)
    pred_index = _build_prediction_index(pred_json)

    task_json = []
    files = list(Path(input_pdf_dir).rglob("*.pdf"))
    for f in tqdm(files):
        # Save pages as PNG files
        images = convert_from_path(pdf_path=f, dpi=300)
        image_list = []
        for idx, image in enumerate(images):
            fname = f"{f.name}_p{idx:03d}.png"
            image.save(dataset_dir / fname, "PNG")
            page = f"{BASE_HOST_PATH}{dataset_name}/{fname}"
            image_list.append(page)

        # Build task json
        doc_id = f.name
        ls_predictions = _build_ls_predictions(pred_index.get(doc_id, []))
        task: dict = {
            "data": {"pages": image_list},
            "meta": {"file": f.name},
        }
        if ls_predictions:
            # TODO: Parameterize bbox_type ("predictions" or "annotations") so we can reuse the function for loading saved progress
            bbox_type = "predictions"
            task[bbox_type] = ls_predictions
        task_json.append(task)

    # Save task json file
    task_json_path = dataset_dir / "tasks_prelabeled.json"
    with open(task_json_path, "w", encoding="utf-8") as f:
        json.dump(task_json, f, indent=2)
    print(f"Wrote {len(task_json)} tasks to {task_json_path}")

    project_id = _create_project(project_name)
    _create_tasks(project_id, task_json_path, dataset_name)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Convert PDFs to page images and create prelabeled Label Studio tasks."
        )
    )
    parser.add_argument(
        "--project_name",
        required=True,
        help="Project name that will appear on Label Studio",
    )
    parser.add_argument(
        "--dataset_name",
        default="dataset",
        help="Dataset name; defines the output directory (default: dataset)",
    )
    parser.add_argument(
        "--input_pdf_dir",
        default="pdf_input/",
        help="Path to the input directory containing PDFs (default: pdf_input/)",
    )
    parser.add_argument(
        "--pred_json_path",
        required=True,
        help="Path to the prediction JSON file",
    )
    args = parser.parse_args()
    main(args.project_name, args.dataset_name, args.input_pdf_dir, args.pred_json_path)
