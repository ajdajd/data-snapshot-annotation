"""
DocLayout-YOLO adapter -> Unified Evaluation Schema v1.3

- Input: directory of PDFs
- Output: single JSON file matching data-snapshot-eval-v1.3.schema.json

Model: DocLayout-YOLO fine-tuned on DocStructBench
  - Local  : data/models/doclayout_yolo_docstructbench_imgsz1024.pt
  - Remote : juliozhao/DocLayout-YOLO-DocStructBench  (HuggingFace repo ID)
             -> cached via huggingface_hub.hf_hub_download on first use
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from pdf2image import convert_from_path
from tqdm.auto import tqdm
from doclayout_yolo import YOLOv10

from dsa.constants import ROOT

# ----------------------------
# Paths / defaults
# ----------------------------

MODEL_NAME = "juliozhao/DocLayout-YOLO-DocStructBench"
MODEL_FILENAME = "doclayout_yolo_docstructbench_imgsz1024.pt"
MODEL_PATH_DEFAULT = ROOT / "data" / "models" / MODEL_FILENAME

INPUT_PDF_DIR = ROOT / "pdf_input"
OUTPUT_JSON_PATH = ROOT / "data/evaluation_input/doclayout-yolo.json"


# ----------------------------
# Model loading (with HF cache)
# ----------------------------


def _resolve_model_path(model_path: Union[str, Path]) -> Path:
    """
    Accept either:
      - A local path to a .pt file (returned as-is if it exists).
      - A HuggingFace repo ID (e.g. "juliozhao/DocLayout-YOLO-DocStructBench"),
        which is downloaded and cached via huggingface_hub.hf_hub_download.
    """
    p = Path(model_path)
    if p.exists():
        return p

    # Treat as a HuggingFace repo ID
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download models from HuggingFace. "
            "Install it with:  pip install huggingface-hub"
        ) from exc

    repo_id = str(model_path)
    cached = hf_hub_download(repo_id=repo_id, filename=MODEL_FILENAME)
    return Path(cached)


# def _load_model(model_path: Union[str, Path]):
#     """Load a YOLOv10 model from doclayout_yolo (lazy import)."""
#     try:
#         from doclayout_yolo import YOLOv10  # type: ignore[import-untyped]
#     except ImportError as exc:
#         raise ImportError(
#             "doclayout-yolo is required.  Install it with:\n"
#             "  pip install doclayout-yolo\n"
#             "or add it to your optional deps:\n"
#             "  pip install 'data-snapshot-annotation[doclayout_yolo]'"
#         ) from exc

#     resolved = _resolve_model_path(model_path)
#     return YOLOv10(str(resolved))


# ----------------------------
# Schema helpers
# ----------------------------

LABEL_MAP: Dict[str, str] = {
    "1": "Figure",
    "2": "Table",
}

# DocLayout-YOLO / DocStructBench class names that map to our canonical labels.
# All other classes (title, text, caption, formula, …) are ignored.
_LABEL_NORMALIZATION: Dict[str, str] = {
    "figure": "Figure",
    "fig": "Figure",
    "image": "Figure",
    "chart": "Figure",
    "diagram": "Figure",
    "table": "Table",
    "tbl": "Table",
}

_ALLOWED_LABELS = set(LABEL_MAP.values())


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _coerce_label(raw: Any) -> Optional[str]:
    """Map a raw YOLO class name to a canonical label, or None to skip."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s in _ALLOWED_LABELS:
        return s
    s_low = s.lower()
    if s_low in _LABEL_NORMALIZATION:
        return _LABEL_NORMALIZATION[s_low]
    # Handle "Figure 1", "Table:", etc.
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
    Convert absolute-pixel xyxy bboxes to normalized [0, 1] xyxy.
    Already-normalized coords are clipped and returned as-is.
    Degenerate boxes (zero area) are dropped.
    """
    out: List[List[float]] = []
    for bb in bboxes:
        if len(bb) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in bb)

        # Heuristic: if any coord > 1.5, assume absolute pixels.
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.5:
            x1, x2 = x1 / float(width), x2 / float(width)
            y1, y2 = y1 / float(height), y2 / float(height)

        nx1, nx2 = sorted([_clip01(x1), _clip01(x2)])
        ny1, ny2 = sorted([_clip01(y1), _clip01(y2)])

        if nx2 <= nx1 or ny2 <= ny1:
            continue  # degenerate

        out.append([nx1, ny1, nx2, ny2])
    return out


# ----------------------------
# Main adapter
# ----------------------------


@dataclass
class DocLayoutYOLOAdapterConfig:
    model_path: Union[str, Path] = field(default_factory=lambda: MODEL_PATH_DEFAULT)
    device: str = "cpu"
    dpi: int = 300
    conf: float = 0.2
    imgsz: int = 1024
    store_doc_path_as: str = "relative"  # "relative" or "absolute"


def run_doclayout_yolo_adapter_directory(
    input_pdf_dir: Union[str, Path],
    output_json_path: Union[str, Path],
    *,
    run_id: Optional[str] = None,
    config: Optional[DocLayoutYOLOAdapterConfig] = None,
) -> Path:
    """
    Run DocLayout-YOLO on all PDFs under *input_pdf_dir* and write a single
    Unified Evaluation Schema v1.3 prediction file to *output_json_path*.

    Returns the output path.
    """
    input_pdf_dir = Path(input_pdf_dir)
    output_json_path = Path(output_json_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = config or DocLayoutYOLOAdapterConfig()
    # model = _load_model(cfg.model_path)
    model = YOLOv10(cfg.model_path)

    pdf_files = sorted(input_pdf_dir.rglob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found under: {input_pdf_dir}")

    run_id = run_id or f"doclayout-yolo-{uuid.uuid4().hex[:10]}"

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

        images = convert_from_path(str(pdf_path), dpi=cfg.dpi)

        for page_index, image in enumerate(
            tqdm(images, desc=f"Pages: {pdf_path.name}", leave=False)
        ):
            det_res = model.predict(
                image,
                imgsz=cfg.imgsz,
                conf=cfg.conf,
                device=cfg.device,
                verbose=False,
            )
            result = det_res[0]

            # Extract boxes, scores, class indices from the Results object.
            boxes_tensor = result.boxes.xyxy.cpu().tolist()   # [[x1,y1,x2,y2], ...]
            scores_list = result.boxes.conf.cpu().tolist()     # [float, ...]
            cls_list = result.boxes.cls.cpu().tolist()         # [float, ...]
            names: Dict[int, str] = result.names              # {0: "title", ...}

            bboxes_norm = _normalize_bboxes_xyxy(
                boxes_tensor, width=image.width, height=image.height
            )

            page_id = f"{doc_id}::p{page_index:03d}"
            objects: List[Dict[str, Any]] = []

            for i, bbox in enumerate(bboxes_norm):
                cls_idx = int(cls_list[i])
                raw_name = names.get(cls_idx, "")
                label = _coerce_label(raw_name)
                if label is None:
                    continue  # skip non-Figure/Table classes

                score = float(scores_list[i])
                score = max(0.0, min(1.0, score))

                objects.append(
                    {
                        "id": f"{page_id}:{i}",
                        "label": label,
                        "bbox": bbox,
                        "score": score,
                    }
                )

            if not objects:
                continue

            predictions.append(
                {
                    "page_id": page_id,
                    "doc_id": doc_id,
                    "page_index": page_index,
                    "objects": objects,
                }
            )

    model_path_str = str(cfg.model_path)
    out = {
        "label_map": LABEL_MAP,
        "info": {
            "schema_version": "1.3",
            "type": "prediction",
            "created_at": _utc_now_iso(),
            "run_id": run_id,
            "model": {
                "name": MODEL_NAME,
                "version": MODEL_FILENAME,
                "notes": (
                    f"adapter=doclayout_yolo; model_path={model_path_str}; "
                    f"device={cfg.device}; dpi={cfg.dpi}; "
                    f"conf={cfg.conf}; imgsz={cfg.imgsz}"
                ),
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
# CLI
# ----------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run DocLayout-YOLO over a PDF directory and produce a v1.3 prediction JSON."
    )
    parser.add_argument(
        "--input_pdf_dir", type=str, default=str(INPUT_PDF_DIR),
        help="Directory of PDF files to process.",
    )
    parser.add_argument(
        "--output_json_path", type=str, default=str(OUTPUT_JSON_PATH),
        help="Destination path for the prediction JSON.",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument(
        "--store_doc_path_as",
        type=str,
        choices=["relative", "absolute"],
        default="relative",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(MODEL_PATH_DEFAULT),
        help=(
            "Path to a local .pt file, or a HuggingFace repo ID "
            "(e.g. juliozhao/DocLayout-YOLO-DocStructBench). "
            "When a repo ID is given the model is downloaded and cached via "
            "huggingface_hub."
        ),
    )
    args = parser.parse_args()

    pdf_dir = Path(args.input_pdf_dir)
    print(f"Input PDF dir : {pdf_dir}")
    pdf_files = list(pdf_dir.rglob("*.pdf"))
    print(f"PDFs found    : {len(pdf_files)}")

    cfg = DocLayoutYOLOAdapterConfig(
        model_path=args.model_path,
        device=args.device,
        dpi=args.dpi,
        conf=args.conf,
        imgsz=args.imgsz,
        store_doc_path_as=args.store_doc_path_as,
    )

    out_path = run_doclayout_yolo_adapter_directory(
        args.input_pdf_dir,
        args.output_json_path,
        run_id=args.run_id,
        config=cfg,
    )
    print(f"Wrote: {out_path}")
