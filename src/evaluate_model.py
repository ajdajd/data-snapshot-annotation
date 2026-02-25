from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


from src.constants import ROOT

# TODO: Move to constants.py
GT_JSON_PATH = ROOT / "data/evaluation_input/ground_truth.json"
PRED_JSON_PATH = ROOT / "data/evaluation_input/tfid-large.json"
# TODO: Reference gt and pred file in report filename
OUTPUT_REPORT_PATH = ROOT / "data/evaluation_output/report.json"


# -----------------------------
# IO
# -----------------------------


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Geometry
# -----------------------------


# TODO: Move to utils.py
def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


# TODO: Move to utils.py
def bbox_sanitize(b: List[float]) -> Tuple[float, float, float, float]:
    if not (isinstance(b, list) and len(b) == 4):
        raise ValueError(f"Invalid bbox: {b}")
    x1, y1, x2, y2 = map(float, b)
    # Robustness: clamp to [0,1], do not reorder
    return (_clamp01(x1), _clamp01(y1), _clamp01(x2), _clamp01(y2))


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


# TODO: Deprecate this; pred files MUST follow page_id naming convention
def index_pages(predictions: List[dict]) -> Dict[Tuple[str, int], List[dict]]:
    """
    Index by (doc_id, page_index). We don't rely on page_id so minor format drift won't break eval.
    """
    out: Dict[Tuple[str, int], List[dict]] = {}
    for p in predictions or []:
        doc_id = p.get("doc_id")
        page_index = p.get("page_index")
        if isinstance(doc_id, str) and isinstance(page_index, int):
            out.setdefault((doc_id, page_index), []).append(p)
    return out


def supported_labels_from_label_map(label_map: Dict[str, str]) -> List[str]:
    # Preserve order of first appearance of values
    seen = set()
    labels: List[str] = []
    for _, v in (label_map or {}).items():
        if isinstance(v, str) and v not in seen:
            labels.append(v)
            seen.add(v)
    return labels


def extract_objects(
    page_entries: List[dict], supported: set[str], expect_score: bool
) -> List[DetObj]:
    """
    Merge objects across duplicate page containers (if any), and filter unsupported labels.
    """
    objs: List[DetObj] = []
    for pe in page_entries or []:
        for o in pe.get("objects", []) or []:
            label = o.get("label")
            if not isinstance(label, str) or label not in supported:
                continue
            obj_id = str(o.get("id", ""))
            bb = bbox_sanitize(o.get("bbox"))
            if expect_score:
                sc = o.get("score", None)
                if sc is None:
                    continue
                objs.append(
                    DetObj(obj_id=obj_id, label=label, bbox=bb, score=float(sc))
                )
            else:
                objs.append(DetObj(obj_id=obj_id, label=label, bbox=bb, score=None))
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


# -----------------------------
# "Validation": document mismatch only
# -----------------------------


def doc_id_set(file_obj: dict) -> set[str]:
    docs = file_obj.get("documents", []) or []
    out = set()
    for d in docs:
        if isinstance(d, dict) and isinstance(d.get("doc_id"), str):
            out.add(d["doc_id"])
    return out


def print_document_mismatch(gt: dict, pred: dict) -> None:
    gt_docs = doc_id_set(gt)
    pred_docs = doc_id_set(pred)

    only_gt = sorted(gt_docs - pred_docs)
    only_pred = sorted(pred_docs - gt_docs)

    if not only_gt and not only_pred:
        print("[OK] documents are consistent: same doc_id set in GT and predictions.")
        return

    if only_gt:
        print(
            f"[WARN] doc_id present in GT but missing in predictions ({len(only_gt)}):"
        )
        for x in only_gt:
            print(f"  - {x}")

    if only_pred:
        print(
            f"[WARN] doc_id present in predictions but missing in GT ({len(only_pred)}):"
        )
        for x in only_pred:
            print(f"  - {x}")


# -----------------------------
# Main evaluate
# -----------------------------


def evaluate(
    ground_truth_path: str | Path,
    prediction_path: str | Path,
    *,
    iou_thresholds: Tuple[float, ...] = (0.5, 0.75),
    output_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    gt = load_json(ground_truth_path)
    pred = load_json(prediction_path)

    # minimal check requested
    print_document_mismatch(gt, pred)

    label_map = gt.get("label_map") or pred.get("label_map") or {}
    labels = supported_labels_from_label_map(label_map)
    supported = set(labels)

    gt_pages = index_pages(gt.get("predictions", []) or [])
    pred_pages = index_pages(pred.get("predictions", []) or [])
    all_keys = set(gt_pages.keys()) | set(pred_pages.keys())

    report: Dict[str, Any] = {
        "info": {
            "schema_version": (pred.get("info", {}) or {}).get("schema_version", "1.3"),
            "gt_path": str(ground_truth_path),
            "pred_path": str(prediction_path),
        },
        "label_map": label_map,
        "thresholds": list(iou_thresholds),
        "metrics": {},
    }

    for thr in iou_thresholds:
        per_class: Dict[str, Stats] = {lab: Stats() for lab in labels}
        micro = Stats()

        for key in sorted(all_keys):
            gt_objs_all = extract_objects(
                gt_pages.get(key, []), supported, expect_score=False
            )
            pred_objs_all = extract_objects(
                pred_pages.get(key, []), supported, expect_score=True
            )

            for lab in labels:
                gt_lab = [o for o in gt_objs_all if o.label == lab]
                pred_lab = [o for o in pred_objs_all if o.label == lab]

                matches, unmatched_p, unmatched_g = greedy_match(gt_lab, pred_lab, thr)

                # TODO: Implement snapshot saving of tp, fp, fn

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

    if output_path is not None:
        outp = Path(output_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    return report


if __name__ == "__main__":
    import argparse

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
        default=[0.5, 0.75],
        help="IoU thresholds (space separated)",
    )
    args = ap.parse_args()

    rep = evaluate(
        args.gt_json_path,
        args.pred_json_path,
        iou_thresholds=tuple(args.thr),
        output_path=args.output_report_path,
    )
    print(json.dumps(rep, indent=4, ensure_ascii=False))
