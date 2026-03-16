from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
PDF_INPUT_DIR = ROOT / "pdf_input"

# evaluate_model.py
GT_JSON_PATH = ROOT / "data/evaluation_input/ground_truth-copy.json"
PRED_JSON_PATH = ROOT / "data/evaluation_input/chatgpt3-copy.json"
OUTPUT_REPORT_PATH = ROOT / "data/evaluation_output/report_chatgpt3-copy.json"
IOU_THRESHOLDS = [0.5, 0.75]
LABELS_TO_CONSIDER = ["Figure", "Table"]
