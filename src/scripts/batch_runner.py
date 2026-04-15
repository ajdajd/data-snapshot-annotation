import logging
from tqdm.auto import tqdm

from dsa.constants import MODELS_DIR
from dsa.adapters.tfid import TFIDConfig, run_tfid_adapter_directory
from dsa.adapters.yolo11 import YOLO11Config, run_yolo11_adapter_directory
from dsa.adapters.yolo26 import YOLO26Config, run_yolo26_adapter_directory
from dsa.adapters.doclayoutyolo import (
    DocLayoutYOLOConfig,
    run_doclayout_yolo_adapter_directory,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="batch_runner.log",
    filemode="a",
)

BATCHES = "pdf_input/unhcr_batch{x}"
OUTPUT_DIR = "data/batch_runs/"


def tfid_batch_runner():
    logging.info("Start TFID runner.")
    for b in tqdm(range(1, 10), desc="Processing batches"):
        path_dir = BATCHES.replace("{x}", str(b))
        out_path = OUTPUT_DIR + f"tfid-large_unhcr_batch{b}.json"
        logging.info(f"Started: {path_dir}")
        cfg = TFIDConfig(
            model_id="yifeihu/TF-ID-large",
            device="cpu",
            dpi=300,
            store_doc_path_as="relative",
            filter_small=False,
        )
        out_path = run_tfid_adapter_directory(
            path_dir,
            out_path,
            run_id=None,
            config=cfg,
        )
        logging.info(f"Wrote: {out_path}")


def yolo11_runner():
    logging.info("Start yolo11 runner.")
    for b in tqdm(range(1, 10), desc="Processing batches"):
        path_dir = BATCHES.replace("{x}", str(b))
        out_path = OUTPUT_DIR + f"yolo11_unhcr_batch{b}.json"
        logging.info(f"Started: {path_dir}")
        cfg = YOLO11Config(
            repo_id="Armaggheddon/yolo11-document-layout",
            filename="yolo11m_doc_layout.pt",
            device="cpu",
            dpi=300,
            conf=0.25,
            iou=0.7,
            imgsz=1280,
            store_doc_path_as="relative",
            filter_small=False,
        )
        out_path = run_yolo11_adapter_directory(
            path_dir,
            out_path,
            run_id=None,
            config=cfg,
        )
        logging.info(f"Wrote: {out_path}")


def yolo26_runner():
    logging.info("Start yolo26 runner.")
    for b in tqdm(range(1, 10), desc="Processing batches"):
        path_dir = BATCHES.replace("{x}", str(b))
        out_path = OUTPUT_DIR + f"yolo26_unhcr_batch{b}.json"
        logging.info(f"Started: {path_dir}")
        cfg = YOLO26Config(
            repo_id="Armaggheddon/yolo26-document-layout",
            filename="yolo26m_doc_layout.pt",
            device="cpu",
            dpi=300,
            conf=0.25,
            iou=0.7,
            imgsz=1280,
            store_doc_path_as="relative",
            filter_small=False,
        )
        out_path = run_yolo26_adapter_directory(
            path_dir,
            out_path,
            run_id=None,
            config=cfg,
        )
        logging.info(f"Wrote: {out_path}")


def doclayoutyolo_runner():
    logging.info("Start doclayoutyolo runner.")
    for b in tqdm(range(1, 10), desc="Processing batches"):
        path_dir = BATCHES.replace("{x}", str(b))
        out_path = OUTPUT_DIR + f"doclayoutyolo_unhcr_batch{b}.json"
        logging.info(f"Started: {path_dir}")
        cfg = DocLayoutYOLOConfig(
            model_path=str(MODELS_DIR / "doclayout_yolo_docstructbench_imgsz1024.pt"),
            device="cpu",
            dpi=300,
            conf=0.2,
            imgsz=1024,
            store_doc_path_as="relative",
            filter_small=False,
        )
        out_path = run_doclayout_yolo_adapter_directory(
            path_dir,
            out_path,
            run_id=None,
            config=cfg,
        )
        logging.info(f"Wrote: {out_path}")


if __name__ == "__main__":
    tfid_batch_runner()
    yolo11_runner()
    yolo26_runner()
    doclayoutyolo_runner()
