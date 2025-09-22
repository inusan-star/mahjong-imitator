import logging

import requests
from tqdm import tqdm

import src.config as config


def run(year: int):
    """Download log zip."""
    config.LOG_ZIPS_DIR.mkdir(parents=True, exist_ok=True)

    file_name = config.TENHO_LOG_ZIP_FILENAME_FORMAT.format(year=year)
    file_path = config.LOG_ZIPS_DIR / file_name
    url = config.TENHO_LOG_ZIP_URL_FORMAT.format(year=year)

    if file_path.exists():
        logging.info("File '%s' already exists. Skipping download.", file_name)
        return

    logging.info("Downloading '%s' from %s...", file_name, url)

    try:
        response = requests.get(url, headers=config.TENHO_HEADERS, stream=True, timeout=config.REQUESTS_TIMEOUT)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(file_path, "wb") as f, tqdm(
            desc=f"Downloading ZIP ({year})",
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)

        logging.info("Successfully downloaded '%s'.", file_name)

    except requests.exceptions.RequestException as _:
        logging.error("Failed to download '%s'. Cleaning up.", file_name)

        if file_path.exists():
            file_path.unlink()

        raise
