import argparse
from pathlib import Path
from pdf2image import convert_from_path
from tqdm.auto import tqdm


def main(input_dir, output_dir):
    files = list(Path(input_dir).rglob("*.pdf"))
    for f in tqdm(files):
        images = convert_from_path(pdf_path=f, dpi=300)

        for idx, image in enumerate(images):
            output_path = Path(output_dir) / f"{f.name}_p{idx:03d}.png"
            image.save(output_path, "PNG")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="pdf_input/",
        help="Path to the input directory (default: pdf_input/)",
    )
    parser.add_argument(
        "--output_dir",
        default="pages_output/",
        help="Path to the output directory (default: pages_output/)",
    )
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
