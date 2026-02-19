import argparse
import json
from pathlib import Path
from pdf2image import convert_from_path
from tqdm.auto import tqdm

BASE_HOST_PATH = "/data/local-files/?d="
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = "labelstudio_data/"


def main(input_dir, dataset_name):
    output_dir = Path(LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT) / dataset_name
    task_json = []

    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(Path(input_dir).rglob("*.pdf"))
    for f in tqdm(files):
        images = convert_from_path(pdf_path=f, dpi=300)

        image_list = []
        for idx, image in enumerate(images, start=1):
            # Save png file
            fname = f"{f.name}_page_{idx:03d}.png"
            image.save(Path(output_dir) / fname, "PNG")

            # Compile to task json
            page = f"{BASE_HOST_PATH}{dataset_name}/{fname}"
            image_list.append(page)

        task_json.append(
            {
                "data": {"pages": image_list},
                "meta": {"file": f.name},
            }
        )

    with open(output_dir / "tasks.json", "w") as f:
        json.dump(task_json, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="pdf_input/",
        help="Path to the input directory (default: pdf_input/)",
    )
    parser.add_argument(
        "--dataset_name",
        default="dataset",
        help="Dataset name; defines the output directory (default: dataset)",
    )
    args = parser.parse_args()
    main(args.input_dir, args.dataset_name)
