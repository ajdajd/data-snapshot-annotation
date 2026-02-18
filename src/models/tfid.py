"""Minimal module for generating results from the yifeihu/TF-ID-large model."""

import json
from pdf2image import convert_from_path
from PIL.Image import Image
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from src.constants import MODEL_OUTPUT_DIR, PDF_INPUT_DIR


MODEL_NAME = "tfid"
MODEL_ID = "yifeihu/TF-ID-large"
DEVICE = "cpu"


class ExtractSnapshot:
    def __init__(self, model_id, device):
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

    def tf_id_detection(self, images: list[Image] | Image) -> list[dict]:
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

        annotations = []
        for i in range(len(images)):
            annotation = self.processor.post_process_generation(
                generated_text[i],
                task="<OD>",
                image_size=(images[i].width, images[i].height),
            )
            annotations.append(annotation["<OD>"])

        return annotations


def normalize_annotation(annotation, width, height):
    bboxes = annotation["bboxes"]

    normalized_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        normalized_bboxes.append([x1 / width, y1 / height, x2 / width, y2 / height])

    normalized_annotation = {k: v for k, v in annotation.items()}
    normalized_annotation["bboxes"] = normalized_bboxes

    return normalized_annotation


def save_annotation(annotation, file_name, page, output_dir=MODEL_OUTPUT_DIR):
    output_dir = output_dir / file_name
    output_dir.mkdir(parents=True, exist_ok=True)

    fp = output_dir / f"{MODEL_NAME}_{page:03d}.json"
    with open(fp, "w") as f:
        json.dump(annotation, f)

    return None


def main():
    extractor = ExtractSnapshot(model_id=MODEL_ID, device=DEVICE)

    pdf_files = sorted(PDF_INPUT_DIR.rglob("*.pdf"))
    for pdf in tqdm(pdf_files, desc="Processing PDFs"):
        images = convert_from_path(pdf, dpi=300)

        for page, image in enumerate(tqdm(images, desc="Processing pages"), start=1):
            annotation = extractor.tf_id_detection(image)[0]
            annotation = normalize_annotation(annotation, image.width, image.height)
            save_annotation(annotation, pdf.name, page)


if __name__ == "__main__":
    main()
