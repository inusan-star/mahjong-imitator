import logging
import random
import time

import requests
from tqdm.rich import tqdm

import src.config as global_config
import src.data.config as data_config


def run(year: int):
    """Download log zips."""
    global_config.LOG_ZIPS_DIR.mkdir(parents=True, exist_ok=True)

    file_name = data_config.TENHO_LOG_ZIP_FILENAME_FORMAT.format(year=year)
    file_path = global_config.LOG_ZIPS_DIR / file_name
    url = data_config.TENHO_LOG_ZIP_URL_FORMAT.format(year=year)

    if file_path.exists():
        logging.info("File '%s' already exists. Skipping.", file_name)
        return

    logging.info("Downloading '%s' from %s ...", file_name, url)

    try:
        time.sleep(random.uniform(global_config.REQUEST_SLEEP_MIN, global_config.REQUEST_SLEEP_MAX))
        response = requests.get(url, headers=data_config.TENHO_HEADERS, stream=True, timeout=global_config.REQUESTS_TIMEOUT)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(file_path, "wb") as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)

        logging.info("Successfully downloaded '%s'.", file_name)

    except requests.exceptions.RequestException:
        logging.error("Failed to download '%s'. Cleaning up. Halting.", file_name)

        if file_path.exists():
            file_path.unlink()

        raise
