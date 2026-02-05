# data-snapshot-annotation
Label Studio for data-snapshot

# Setup
```shell
mkdir data-snapshot-annotation
cd data-snapshot-annotation
# copy docker-compose.yml here or git clone
docker compose up
```

Then enter `localhost:8000` on a web browser to start annotating!

# Troubleshooting

- `PermissionError: [Errno 13] Permission denied: '/label-studio/data/media'`
  - Solution: Give writeable permission.
    ```shell
    sudo chmod -R 777 label-studio/.
    ```
