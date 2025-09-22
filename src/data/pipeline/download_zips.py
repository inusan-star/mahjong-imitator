import logging
import requests
from tqdm import tqdm

import src.config as config


def run(year: int):
    config.LOG_ZIPS_DIR.mkdir(parents=True, exist_ok=True)

    file_name = config.TENHO_LOG_ZIP_FILENAME_FORMAT.format(year=year)
    file_path = config.LOG_ZIPS_DIR / file_name
    url = config.TENHO_LOG_ZIP_URL_FORMAT.format(year=year)

    if file_path.exists():
        logging.info(f"File '{file_name}' already exists. Skipping download.")
        return

    logging.info(f"Downloading '{file_name}' from {url}... .")

    try:
        response = requests.get(url, headers=config.TENHO_HEADERS, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(file_path, "wb") as f, tqdm(
            desc=file_name,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)

        logging.info(f"Successfully downloaded '{file_name}'.")

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download {file_name}. Cleaning up.")

        if file_path.exists():
            file_path.unlink()

        raise
