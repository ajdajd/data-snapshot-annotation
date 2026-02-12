# data-snapshot-annotation
Label Studio for data-snapshot

# First time setup

1. Setup Label Studio
    ```shell
    mkdir data-snapshot-annotation
    cd data-snapshot-annotation
    # copy docker-compose.yml here or git clone
    docker compose up
    ```
2. If you encounter permission issues, run
    ```shell
    sudo chmod -R 777 .
    ```
    then retry.
    ```shell
    docker compose up
    ```
3. Open `http://localhost:8080/` on a web browser and create a login.

# Setting up a project

## Pre-requisites

Install `requirements.txt` and Poppler.
```shell
pip install -r requirements.txt
sudo apt-get install poppler-utils
```

## Convert PDFs to images and creating tasks
1. Add PDF files to the `pdf_input` directory.
2. Run `python create_tasks.py --dataset_name==dataset`. The `dataset_name` parameter may be set into any string.
3. This will generate the following files into the `labelstudio_data/dataset` directory:
    - Individual PNG files for each page of each PDF
    - A `tasks.json` file.

## Creating an annotation project
1. Open Label Studio and click `Create Project`.
2. Fill out Project Name page.
3. In Data Import, click `Upload Files` and select the `tasks.json` generated in the previous section.
4. In Labeling Setup, select `Multi-page document annotation`.
5. Create label names. This can be edited later.
6. Click `Save`.
7. Go to the project's settings.
8. Select `Cloud Storage` > `Add Source Storage` > `Local Files` > `Next`.
9. Add a Storage Title (e.g., "Dataset").
10. In Absolute local path, replace `/label-studio/data/your-subdirectory` with `/label-studio/data/dataset`.
11. Click `Test Connection` > `Next`.
12. Import Method: `Tasks - Treat each JSON, JSONL, or Parquet...`
13. Click `Next` > `Save`. (Important: Do NOT click `Save & Sync`.)

Go to the project tab. Each row (called a "task") should correspond to a PDF file to annotate.

# Troubleshooting

- `PermissionError: [Errno 13] Permission denied: '/label-studio/data/media'`
  - Solution: Give writeable permission.
    ```shell
    sudo chmod -R 777 .
    ```
