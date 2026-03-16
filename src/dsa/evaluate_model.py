from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

from dsa.constants import (
    ROOT,
    GT_JSON_PATH,
    PRED_JSON_PATH,
    OUTPUT_REPORT_PATH,
    IOU_THRESHOLDS,
    LABELS_TO_CONSIDER,
)
from dsa.utils import load_json, sanitize_bbox


def bbox_area(b: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_intersection(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def iou(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> float:
    inter = bbox_intersection(a, b)
    if inter <= 0.0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0.0 else 0.0


def area_recall(
    pred: Tuple[float, float, float, float], gt: Tuple[float, float, float, float]
) -> float:
    """Intersection / GT area (coverage)."""
    inter = bbox_intersection(pred, gt)
    g = bbox_area(gt)
    return inter / g if g > 0.0 else 0.0


def area_precision(
    pred: Tuple[float, float, float, float], gt: Tuple[float, float, float, float]
) -> float:
    """Intersection / Pred area (purity)."""
    inter = bbox_intersection(pred, gt)
    p = bbox_area(pred)
    return inter / p if p > 0.0 else 0.0


# -----------------------------
# Indexing / extraction
# -----------------------------


@dataclass(frozen=True)
class DetObj:
    obj_id: str
    label: str
    bbox: Tuple[float, float, float, float]
    score: Optional[float] = None


def prepare_prediction_objects(
    pred_dict: List[dict], filter_list: List = list()
) -> List[DetObj]:
    """Extract prediction objects."""
    preds = pred_dict.get("predictions", [])
    objs: List[DetObj] = []

    for p in preds:
        if p.get("doc_id") in filter_list:
            continue
        for o in p.get("objects", []):
            label = o.get("label")
            obj_id = str(o.get("id", ""))
            bb = sanitize_bbox(o.get("bbox"))
            score = o.get("score", None)

            objs.append(DetObj(obj_id=obj_id, label=label, bbox=bb, score=score))

    return objs


# -----------------------------
# Matching + metrics
# -----------------------------


@dataclass
class Stats:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    matched: int = 0
    iou_sum: float = 0.0
    area_recall_sum: float = 0.0
    area_precision_sum: float = 0.0

    def add_match(self, pred_box, gt_box) -> None:
        self.tp += 1
        self.matched += 1
        self.iou_sum += iou(pred_box, gt_box)
        self.area_recall_sum += area_recall(pred_box, gt_box)
        self.area_precision_sum += area_precision(pred_box, gt_box)

    def add_fp(self, n: int) -> None:
        self.fp += n

    def add_fn(self, n: int) -> None:
        self.fn += n

    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else math.nan

    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else math.nan

    def mean_iou(self) -> float:
        return self.iou_sum / self.matched if self.matched > 0 else math.nan

    def mean_area_recall(self) -> float:
        return self.area_recall_sum / self.matched if self.matched > 0 else math.nan

    def mean_area_precision(self) -> float:
        return self.area_precision_sum / self.matched if self.matched > 0 else math.nan


def greedy_match(
    gt: List[DetObj], pred: List[DetObj], thr: float
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    Greedy one-to-one matching by IoU descending, using only pairs with IoU >= thr.
    Returns matches as (pred_idx, gt_idx, iou).
    """
    cand: List[Tuple[float, int, int]] = []
    for pi, p in enumerate(pred):
        for gi, g in enumerate(gt):
            v = iou(p.bbox, g.bbox)
            if v >= thr:
                cand.append((v, pi, gi))

    cand.sort(reverse=True, key=lambda t: t[0])

    used_p, used_g = set(), set()
    matches: List[Tuple[int, int, float]] = []
    for v, pi, gi in cand:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matches.append((pi, gi, v))

    unmatched_p = [i for i in range(len(pred)) if i not in used_p]
    unmatched_g = [i for i in range(len(gt)) if i not in used_g]
    return matches, unmatched_p, unmatched_g


def get_doc_ids(pred_dict: dict) -> set[str]:
    docs = pred_dict.get("documents", [])

    ids = set()
    for d in docs:
        if isinstance(d, dict):
            ids.add(d.get("doc_id"))

    return ids


def get_document_mismatch(gt: dict, pred: dict) -> Tuple[List[str], List[str]]:
    gt_docs = get_doc_ids(gt)
    pred_docs = get_doc_ids(pred)

    only_gt = sorted(gt_docs - pred_docs)
    only_pred = sorted(pred_docs - gt_docs)

    if not only_gt and not only_pred:
        print("[OK] documents are consistent: same doc_id set in GT and predictions.")
    if only_gt:
        print(
            f"[WARN] doc_ids present in GT but missing in predictions ({len(only_gt)}):"
        )
        for x in only_gt:
            print(f"  - {x}")
    if only_pred:
        print(
            f"[WARN] doc_ids present in predictions but missing in GT ({len(only_pred)}):"
        )
        for x in only_pred:
            print(f"  - {x}")

    return only_gt, only_pred


# -----------------------------
# Main evaluate
# -----------------------------


def evaluate(
    gt_json_path: str | Path,
    pred_json_path: str | Path,
    *,
    iou_thresholds: Tuple[float, ...] = (0.5, 0.75),
    labels: Tuple[str, ...] = ("Figure", "Table"),
    output_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    # Load files
    gt_json_path = Path(gt_json_path)
    pred_json_path = Path(pred_json_path)
    gt = load_json(gt_json_path)
    pred = load_json(pred_json_path)

    # Prepare files
    only_gt, only_pred = get_document_mismatch(gt, pred)
    gt_objects = prepare_prediction_objects(gt)
    pred_objects = prepare_prediction_objects(pred)

    # Prepare report dict
    schema_version = gt.get("info").get("schema_version")
    label_map = gt.get("label_map")
    label_map = {k: v for k, v in label_map.items() if v in labels}
    doc_mismatch = {"only_gt": only_gt, "only_pred": only_pred}
    report: Dict[str, Any] = {
        "info": {
            "schema_version": schema_version,
            "gt_path": str(gt_json_path.resolve().relative_to(ROOT)),
            "pred_path": str(pred_json_path.resolve().relative_to(ROOT)),
        },
        "label_map": label_map,
        "thresholds": list(iou_thresholds),
        "documents_mismatch": doc_mismatch,
        "metrics": {},
    }

    # Calculate metrics
    for thr in iou_thresholds:
        per_class: Dict[str, Stats] = {lab: Stats() for lab in labels}
        micro = Stats()

        for lab in labels:
            gt_lab = [o for o in gt_objects if o.label == lab]
            pred_lab = [o for o in pred_objects if o.label == lab]

            matches, unmatched_p, unmatched_g = greedy_match(gt_lab, pred_lab, thr)

            st = per_class[lab]
            for pi, gi, _v in matches:
                pred_box = pred_lab[pi].bbox
                gt_box = gt_lab[gi].bbox
                st.add_match(pred_box, gt_box)
                micro.add_match(pred_box, gt_box)

            st.add_fp(len(unmatched_p))
            st.add_fn(len(unmatched_g))
            micro.add_fp(len(unmatched_p))
            micro.add_fn(len(unmatched_g))

        report["metrics"][str(thr)] = {
            "micro": {
                "tp": micro.tp,
                "fp": micro.fp,
                "fn": micro.fn,
                "precision": micro.precision(),
                "recall": micro.recall(),
                "matched": micro.matched,
                "mean_iou": micro.mean_iou(),
                "mean_area_precision": micro.mean_area_precision(),
                "mean_area_recall": micro.mean_area_recall(),
            },
            "per_class": {
                lab: {
                    "tp": st.tp,
                    "fp": st.fp,
                    "fn": st.fn,
                    "precision": st.precision(),
                    "recall": st.recall(),
                    "matched": st.matched,
                    "mean_iou": st.mean_iou(),
                    "mean_area_precision": st.mean_area_precision(),
                    "mean_area_recall": st.mean_area_recall(),
                }
                for lab, st in per_class.items()
            },
        }

    # Save report
    if output_path is not None:
        outp = Path(output_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gt_json_path",
        help="Path to ground truth json file",
        default=GT_JSON_PATH,
    )
    ap.add_argument(
        "--pred_json_path",
        help="Path to prediction json file",
        default=PRED_JSON_PATH,
    )
    ap.add_argument(
        "--output_report_path",
        default=OUTPUT_REPORT_PATH,
        help="Path to save output report json",
    )
    ap.add_argument(
        "--thr",
        nargs="*",
        type=float,
        default=IOU_THRESHOLDS,
        help="IoU thresholds (space separated)",
    )
    ap.add_argument(
        "--labels",
        nargs="*",
        type=str,
        default=LABELS_TO_CONSIDER,
        help="Labels to consider (space separated)",
    )
    args = ap.parse_args()

    rep = evaluate(
        gt_json_path=args.gt_json_path,
        pred_json_path=args.pred_json_path,
        iou_thresholds=tuple(args.thr),
        labels=tuple(args.labels),
        output_path=args.output_report_path,
    )
    print(f"Done! Evaluations report saved at {args.output_report_path}")
