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

# Creating an annotation project

1. Click `Create Project`.
2. Fill out Project Name page.
3. Skip Data Import.
4. In Labeling Setup, select `Object Detection with Bounding Boxes`.
5. Create label names. This can be edited later.
6. Click `Save`.

# Adding data to annotate
1. Create a folder in the `labelstudio_data` folder (e.g., `project1_data`) and put the files to annotate there. (See [Converting PDF files to images](#converting-pdf-files-to-images) section for handling PDF files.)
2. Go to the project's settings.
3. Select `Cloud Storage` > `Add Source Storage` > `Local Files` > `Next`.
4. Add a Storage Title.
5. In Absolute local path, replace `/label-studio/data/your-subdirectory` with the folder you created in Step 1 (e.g., `/label-studio/data/project1_data`).
6. Click `Test Connection` > `Next`.
7. Import Method: `Files - Automatically creates a task...`
8. Click `Load Preview`. It show the files in the data folder.
9. Click `Next` > `Save & Sync`.

The files should show up in the project tab. New files will appear as new tasks or by manually syncing in the Cloud Storage settings.

# Converting PDF files to images

1. Install `requirements.txt` and Poppler.
    ```shell
    pip install -r requirements.txt
    sudo apt-get install poppler-utils
    ```
2. Add PDF files to the `pdf_input` directory.
3. Run `python extract_pages.py`.
4. Pages will be saves PNG files in the `pages_output` directory.

# Troubleshooting

- `PermissionError: [Errno 13] Permission denied: '/label-studio/data/media'`
  - Solution: Give writeable permission.
    ```shell
    sudo chmod -R 777 .
    ```
