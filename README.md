# data-snapshot-annotation
Label Studio for data-snapshot

# First time setup

1. Clone the repository.
    ```shell
    git clone git@github.com:ajdajd/data-snapshot-annotation.git
    ```
2. Go inside the repo and start the service.
    ```shell
    cd data-snapshot-annotation
    docker compose up
    ```
3. If you encounter permission issues, run
    ```shell
    # Note: Make your cwd is the project directory!
    sudo chmod -R 777 .
    ```
    then retry.
    ```shell
    docker compose up
    ```
4. Open `http://localhost:8080/` on a web browser and create a login.

# Setting up an annotation project

## 1. Pre-requisites

1. Install dependencies.
    ```shell
    pip install .
    ```
2. Install Poppler.
    ```shell
    sudo apt-get install poppler-utils
    ```

## 2. Converting PDFs to images and creating tasks for Label Studio
1. Add PDF files to the `pdf_input` directory.
2. Run `python create_tasks.py --dataset_name=={dataset}`. The `dataset_name` parameter may be set into any string.
3. This will generate the following files into the `labelstudio_data/{dataset}` directory:
    - Individual PNG files for each page of each PDF
    - A `tasks.json` file.

## 3. Creating an annotation project
1. Setup the project.
    1. Open Label Studio and click `Create Project`.
    2. Fill out Project Name page.
    3. In Data Import, click `Upload Files` and select the `tasks.json` generated in the previous section.
    4. In Labeling Setup, select `Multi-page document annotation`.
    5. Create label names. This can be edited later.
    6. Click `Save`.
2. Setup the dataset.
    1. Go to the project's settings.
    8. Select `Cloud Storage` > `Add Source Storage` > `Local Files` > `Next`.
    9. Add a Storage Title.
    10. In Absolute local path, replace `/label-studio/data/your-subdirectory` with `/label-studio/data/{dataset}`.
    11. Click `Test Connection` > `Next`.
    12. Import Method: `Tasks - Treat each JSON, JSONL, or Parquet...`
    13. Click `Next` > `Save`. (Important: Do NOT click `Save & Sync`.)
3. Go to the project tab. Each row (called a "task") should correspond to a PDF file to annotate.

# Evaluating models (WIP)

1. Install additional dependencies.
    ```shell
    pip install -e .["evaluation"]
    ```
1. Generate ground truth labels `ground_truth.json`.
    1. Export annotations from Label Studio. Use the JSON format.
    2. Run `python src/labelstudio.py`. Make sure to point `INPUT_JSON_PATH` to the file generated from the previous step.
2. Generate prediction file(s).
    1. Run `python src/tfid.py` to generate `tfid-large.json`.
3. Run `python src/evaluate_model.py --gt_json_path=path/to/ground_truth.json --pred_json_path=path/to/pred.json`

# Troubleshooting

- `PermissionError: [Errno 13] Permission denied: '/label-studio/data/media'` when setting up Label Studio
  - Solution: Give writeable permission to the project directory.
    ```shell
    # Note: Make your cwd is the project directory!
    sudo chmod -R 777 .
    ```
