"""
TF-ID Large adapter -> Unified Evaluation Schema v1.3

- Input: directory of PDFs
- Output: single JSON file matching data-snapshot-eval-v1.3.schema.json
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from pdf2image import convert_from_path
from PIL.Image import Image
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from src.constants import ROOT

# ----------------------------
# Model wrapper (same as yours)
# ----------------------------

# TODO: Move to constants.py
MODEL_ID_DEFAULT = "yifeihu/TF-ID-large"
INPUT_PDF_DIR = ROOT / "pdf_input"
OUTPUT_JSON_PATH = ROOT / "data/evaluation_input/tfid-large.json"


class ExtractSnapshot:
    def __init__(self, model_id: str, device: str):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map=device,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map=device,
        )

    def tf_id_detection(
        self, images: Union[List[Image], Image]
    ) -> List[Dict[str, Any]]:
        if not isinstance(images, list):
            images = [images]

        prompt = ["<OD>"] * len(images)
        inputs = self.processor(text=prompt, images=images, return_tensors="pt")
        inputs.to(self.model.device)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )

        annotations: List[Dict[str, Any]] = []
        for i in range(len(images)):
            annotation = self.processor.post_process_generation(
                generated_text[i],
                task="<OD>",
                image_size=(images[i].width, images[i].height),
            )
            # TF-ID returns a dict keyed by "<OD>"
            annotations.append(annotation["<OD>"])

        return annotations


# ----------------------------
# Schema helpers
# ----------------------------

LABEL_MAP: Dict[str, str] = {
    "1": "Figure",
    "2": "Table",
}

# Common variants we’ve seen in model outputs
_LABEL_NORMALIZATION = {
    "figure": "Figure",
    "fig": "Figure",
    "chart": "Figure",
    "image": "Figure",
    "diagram": "Figure",
    "table": "Table",
    "tbl": "Table",
}

_ALLOWED_LABELS = set(LABEL_MAP.values())


def _utc_now_iso() -> str:
    # ISO8601 with Z
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _coerce_label(label: Any) -> Optional[str]:
    """
    Returns canonical label ("Figure"/"Table") or None if unrecognized.
    """
    if label is None:
        return None

    if isinstance(label, (int, float)):
        # If the model outputs numeric ids (rare), try mapping:
        key = str(int(label))
        return LABEL_MAP.get(key)

    s = str(label).strip()
    if not s:
        return None

    # normalize
    s_low = s.lower()
    if s in _ALLOWED_LABELS:
        return s
    if s_low in _LABEL_NORMALIZATION:
        return _LABEL_NORMALIZATION[s_low]

    # sometimes model outputs e.g. "Figure 1" or "Table:"
    for k, v in _LABEL_NORMALIZATION.items():
        if s_low.startswith(k):
            return v

    return None


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _normalize_bboxes_xyxy(
    bboxes: Sequence[Sequence[float]], width: int, height: int
) -> List[List[float]]:
    """
    Converts absolute xyxy -> normalized_xyxy if needed.
    If bboxes already look normalized, we still clip to [0,1].
    """
    out: List[List[float]] = []
    for bb in bboxes:
        if len(bb) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bb]

        # Heuristic: if any coordinate > 1.5, assume absolute pixels.
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.5:
            x1, x2 = x1 / float(width), x2 / float(width)
            y1, y2 = y1 / float(height), y2 / float(height)

        # enforce ordering (and clip)
        nx1, nx2 = sorted([_clip01(x1), _clip01(x2)])
        ny1, ny2 = sorted([_clip01(y1), _clip01(y2)])

        # If degenerate after sorting/clipping, skip
        if nx2 <= nx1 or ny2 <= ny1:
            continue

        out.append([nx1, ny1, nx2, ny2])
    return out


def _extract_lists(
    annotation: Dict[str, Any],
) -> Tuple[List[List[float]], List[Any], Optional[List[float]]]:
    """
    TF-ID typically returns something like:
      {
        "bboxes": [[x1,y1,x2,y2], ...],
        "labels": ["Figure", "Table", ...]   # or similar
        "scores": [0.93, 0.88, ...]          # may or may not exist depending on post-processing
      }
    We make this defensive.
    """
    bboxes = annotation.get("bboxes") or []
    labels = annotation.get("labels") or annotation.get("classes") or []
    scores = annotation.get("scores")

    # Some toolchains store labels as ints in a "labels" list and names elsewhere; we keep it best-effort.
    if not isinstance(bboxes, list):
        bboxes = []
    if not isinstance(labels, list):
        labels = []

    if scores is not None and not isinstance(scores, list):
        scores = None

    return bboxes, labels, scores


# ----------------------------
# Main adapter
# ----------------------------


@dataclass
class TFIDAdapterConfig:
    model_id: str = MODEL_ID_DEFAULT
    device: str = "cpu"
    dpi: int = 300
    # If you have a preferred way to store doc_path (absolute vs relative), set this:
    store_doc_path_as: str = "relative"  # "relative" or "absolute"


def run_tfid_adapter_directory(
    input_pdf_dir: Union[str, Path],
    output_json_path: Union[str, Path],
    *,
    run_id: Optional[str] = None,
    config: Optional[TFIDAdapterConfig] = None,
) -> Path:
    """
    Runs yifeihu/TF-ID-large on all PDFs under input_pdf_dir and writes ONE
    Unified Evaluation Schema v1.3 prediction file to output_json_path.

    Returns output path.
    """
    input_pdf_dir = Path(input_pdf_dir)
    output_json_path = Path(output_json_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = config or TFIDAdapterConfig()
    extractor = ExtractSnapshot(model_id=cfg.model_id, device=cfg.device)

    pdf_files = sorted(input_pdf_dir.rglob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found under: {input_pdf_dir}")

    run_id = run_id or f"tfid-{uuid.uuid4().hex[:10]}"

    documents: List[Dict[str, str]] = []
    predictions: List[Dict[str, Any]] = []

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        doc_id = pdf_path.name

        if cfg.store_doc_path_as == "absolute":
            doc_path = str(pdf_path.resolve())
        else:
            doc_path = str(pdf_path.resolve().relative_to(ROOT))

        documents.append(
            {
                "doc_id": doc_id,
                "doc_name": pdf_path.name,
                "doc_path": doc_path,
            }
        )

        # Render to images (we do NOT store images in schema for predictions)
        images = convert_from_path(str(pdf_path), dpi=cfg.dpi)

        for page_index, image in enumerate(
            tqdm(images, desc=f"Pages: {pdf_path.name}", leave=False)
        ):
            ann = extractor.tf_id_detection(image)[0]  # per your current code
            bboxes_raw, labels_raw, scores_raw = _extract_lists(ann)

            bboxes = _normalize_bboxes_xyxy(
                bboxes_raw, width=image.width, height=image.height
            )

            # align lengths defensively
            n = len(bboxes)
            labels_raw = list(labels_raw)[:n] + [None] * max(0, n - len(labels_raw))
            if scores_raw is not None:
                scores_raw = list(scores_raw)[:n] + [None] * max(0, n - len(scores_raw))

            page_id = f"{doc_id}::p{page_index:03d}"
            objects: List[Dict[str, Any]] = []

            for i in range(n):
                label = _coerce_label(labels_raw[i])
                if label is None:
                    # If label is unknown, skip (schema requires label consistency with label_map)
                    continue

                score_val = 1.0
                if scores_raw is not None and scores_raw[i] is not None:
                    try:
                        score_val = float(scores_raw[i])
                    except Exception:
                        score_val = 1.0

                # clip score to [0,1]
                if score_val < 0.0:
                    score_val = 0.0
                elif score_val > 1.0:
                    score_val = 1.0

                objects.append(
                    {
                        "id": f"{page_id}:{i}",
                        "label": label,
                        "bbox": bboxes[i],
                        "score": score_val,
                    }
                )

            # Drop pages without ANY rectangle annotations (any label)
            if not objects:
                continue

            predictions.append(
                {
                    "page_id": page_id,
                    "doc_id": doc_id,
                    "page_index": page_index,  # 0-based per schema
                    "objects": objects,
                }
            )

    out = {
        "label_map": LABEL_MAP,
        "info": {
            "schema_version": "1.3",
            "type": "prediction",
            "created_at": _utc_now_iso(),
            "run_id": run_id,
            "model": {
                "name": "yifeihu/TF-ID-large",
                "version": "unknown",
                "notes": f"adapter=tfid; device={cfg.device}; dpi={cfg.dpi}",
            },
            "coordinate_system": {
                "type": "normalized_xyxy",
                "range": [0.0, 1.0],
                "origin": "top_left",
            },
        },
        "documents": documents,
        "predictions": predictions,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=4)

    return output_json_path


# ----------------------------
# Optional CLI-style usage
# ----------------------------

if __name__ == "__main__":
    # Example:
    #   python tfid_adapter.py /path/to/pdfs /path/to/output/tfid_predictions.json --run_id myrun
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pdf_dir", type=str, default=INPUT_PDF_DIR)
    parser.add_argument("--output_json_path", type=str, default=OUTPUT_JSON_PATH)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--store_doc_path_as",
        type=str,
        choices=["relative", "absolute"],
        default="relative",
    )
    parser.add_argument("--model_id", type=str, default=MODEL_ID_DEFAULT)
    args = parser.parse_args()

    cfg = TFIDAdapterConfig(
        model_id=args.model_id,
        device=args.device,
        dpi=args.dpi,
        store_doc_path_as=args.store_doc_path_as,
    )

    out_path = run_tfid_adapter_directory(
        args.input_pdf_dir,
        args.output_json_path,
        run_id=args.run_id,
        config=cfg,
    )
    print(f"Wrote: {out_path}")
