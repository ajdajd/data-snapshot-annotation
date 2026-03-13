# snapshot_generator_simple.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from pdf2image import convert_from_path
from PIL import Image
from tqdm.auto import tqdm

from src.constants import ROOT

# TODO: Move to constants.py
CACHE_DIR_PATH = ROOT / "data/pages_cache/"


def _safe_filename(s: str) -> str:
    # Keep it simple and filesystem-safe
    return "".join(ch if (ch.isalnum() or ch in "._-+") else "_" for ch in s)


def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _bbox_to_pixels_xyxy(
    bbox: List[float],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    """
    bbox is normalized [x1,y1,x2,y2] in [0,1], origin top-left.
    Returns integer pixel box for PIL crop: (left, top, right, bottom)
    """
    if not (isinstance(bbox, list) and len(bbox) == 4):
        raise ValueError(f"Invalid bbox: {bbox}")

    x1, y1, x2, y2 = map(float, bbox)

    # clamp to [0,1]
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))

    # convert to pixel coords
    left = int(round(x1 * width))
    top = int(round(y1 * height))
    right = int(round(x2 * width))
    bottom = int(round(y2 * height))

    # clamp to image bounds
    left = max(0, min(width, left))
    right = max(0, min(width, right))
    top = max(0, min(height, top))
    bottom = max(0, min(height, bottom))

    if right <= left or bottom <= top:
        raise ValueError(
            f"Degenerate crop box: {(left, top, right, bottom)} from bbox={bbox}"
        )

    return left, top, right, bottom


def generate_page_cache_on_disk(
    pred_json_path: str | Path,
    cache_dir_path: str | Path,
    *,
    dpi: int = 300,
) -> None:
    """
    Pass 1:
      For each document in JSON, rasterize all pages into cache_dir_path as PNGs:
        cache_dir_path/<pdf_filename>_p000.png, _p001.png, ...
    """
    pred = _load_json(pred_json_path)
    documents: list[Dict] = pred.get("documents")

    cache_dir = Path(cache_dir_path)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for doc in tqdm(documents, desc="Caching pages"):
        doc_id = doc.get("doc_id")
        doc_name = doc.get("doc_name")

        try:
            doc_path = Path(doc.get("doc_path"))
        except TypeError:
            # Infer doc_path if not given or causing errors
            doc_path = ROOT / f"pdf_input/{doc_name}"

        try:
            if not doc_path.exists():
                print(f"[ERROR] PDF missing for doc_id={doc_id}: {doc_path}")
                continue

            images = convert_from_path(pdf_path=str(doc_path), dpi=dpi)

            for idx, image in enumerate(images):
                out_path = cache_dir / f"{doc_name}_p{idx:03d}.png"
                # Save unconditionally (debug-friendly); change if you want caching behavior
                image.save(out_path, "PNG")

        except Exception as e:
            print(f"[ERROR] Failed caching doc_id={doc_id}, pdf={doc_path}: {e}")


def generate_lookup(lst, key, value):
    output = {}
    for x in lst:
        okey = x.get(key)
        oval = x.get(value)
        if okey is not None and oval is not None:
            output[okey] = oval
    return output


def crop_snapshots_from_cache(
    pred_json_path: str | Path,
    cache_dir_path: str | Path,
    output_dir: str | Path,
    *,
    allowed_labels: Optional[List[str]] = None,
) -> None:
    """
    Pass 2:
      For each page prediction entry, open cached page PNG with PIL, crop each bbox, save:
        output_dir/<doc_id>/<label>__<doc_id>_p{page_index:03d}_{k:03d}.png
      where k is the kth label instance in that page (per label, 1-indexed).
    """
    pred = _load_json(pred_json_path)
    doc_name_dict = generate_lookup(pred["documents"], key="doc_id", value="doc_name")

    cache_dir = Path(cache_dir_path)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # If allowed_labels not provided, default to label_map values if present
    if allowed_labels is None:
        label_map = pred.get("label_map") or {}
        if isinstance(label_map, dict) and label_map:
            allowed_labels = [v for v in label_map.values() if isinstance(v, str)]
            # de-dupe preserve order
            seen = set()
            allowed_labels = [
                x for x in allowed_labels if not (x in seen or seen.add(x))
            ]
    allowed_set = set(allowed_labels) if allowed_labels else None

    predictions = pred.get("predictions", []) or []
    for page in tqdm(predictions, desc="Cropping objects"):
        try:
            if not isinstance(page, dict):
                continue
            doc_id = page.get("doc_id")
            page_index = page.get("page_index")
            if not isinstance(doc_id, str) or not isinstance(page_index, int):
                print(f"[ERROR] Bad page entry (missing doc_id/page_index): {page}")
                continue

            pdf_filename = doc_name_dict[doc_id]

            page_img_path = cache_dir / f"{pdf_filename}_p{page_index:03d}.png"
            if not page_img_path.exists():
                print(
                    f"[ERROR] Cached page missing: {page_img_path} (doc_id={doc_id}, page_index={page_index})"
                )
                continue

            try:
                im = Image.open(page_img_path).convert("RGB")
            except Exception as e:
                print(f"[ERROR] Failed to open cached page: {page_img_path}: {e}")
                continue

            width, height = im.size

            objects = page.get("objects", []) or []
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                label = obj.get("label")
                bbox = obj.get("bbox")
                obj_id = obj.get("id")
                if not isinstance(label, str) or not isinstance(bbox, list):
                    print(f"[ERROR] Bad object format obj_id={obj_id}.")
                    continue

                if allowed_set is not None and label not in allowed_set:
                    continue

                try:
                    crop_box = _bbox_to_pixels_xyxy(bbox, width, height)
                    crop = im.crop(crop_box)
                except Exception as e:
                    print(f"[ERROR] Crop failed obj_id={obj_id}: {e}")
                    continue

                doc_dir = out_root / _safe_filename(doc_id)
                doc_dir.mkdir(parents=True, exist_ok=True)

                label_slug = _safe_filename(label)
                out_name = f"{label_slug}__{_safe_filename(obj_id)}.png"
                out_path = doc_dir / out_name

                try:
                    crop.save(out_path, "PNG")
                except Exception as e:
                    print(f"[ERROR] Failed saving crop to {out_path}: {e}")

        except Exception as e:
            print(f"[ERROR] Unexpected failure on page entry: {e}")


def main(
    pred_json_path: str | Path,
    output_dir: str | Path,
    cache_dir_path: str | Path,
    *,
    dpi: int = 300,
    allowed_labels: Optional[List[str]] = None,
) -> None:
    # 1) Cache all page images
    generate_page_cache_on_disk(pred_json_path, cache_dir_path, dpi=dpi)

    # 2) Crop snapshots from the cached page images
    crop_snapshots_from_cache(
        pred_json_path, cache_dir_path, output_dir, allowed_labels=allowed_labels
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_json_path", required=True, help="Path to json file")
    ap.add_argument(
        "--output_dir_path",
        required=True,
        help="Path to store data snapshots",
    )
    ap.add_argument(
        "--cache_dir_path",
        default=CACHE_DIR_PATH,
        help="Path to store full page PNGs",
    )
    ap.add_argument(
        "--dpi", type=int, default=300, help="DPI for pdf2image rasterization"
    )
    ap.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional allowed labels (e.g., Figure Table).",
    )
    args = ap.parse_args()

    main(
        pred_json_path=args.pred_json_path,
        output_dir=args.output_dir_path,
        cache_dir_path=args.cache_dir_path,
        dpi=args.dpi,
        allowed_labels=args.labels,
    )
