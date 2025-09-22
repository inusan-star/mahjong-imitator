import logging
import pathlib
import shutil
import zipfile

from tqdm import tqdm

import src.config as config


def run(year: int):
    """Decompress log archive."""
    zip_filename = config.TENHO_LOG_ZIP_FILENAME_FORMAT.format(year=year)
    zip_filepath = config.LOG_ZIPS_DIR / zip_filename
    output_dir = config.GZIPPED_LOGS_DIR / str(year)

    if not zip_filepath.exists():
        logging.warning("File '%s' not found. Skipping decompression.", zip_filename)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Decompressing '%s' to '%s'...", zip_filename, output_dir)

    try:
        with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
            member_list = [
                member for member in zip_ref.infolist() if member.filename.endswith(".gz") and not member.is_dir()
            ]

            for member in tqdm(member_list, desc=f"Decompressing ZIP ({year})"):
                basename = pathlib.Path(member.filename).name
                target_path = output_dir / basename

                if target_path.exists():
                    continue

                with zip_ref.open(member) as source_file, open(target_path, "wb") as target_file:
                    shutil.copyfileobj(source_file, target_file)

        logging.info("Successfully decompressed '%s'.", zip_filename)

    except zipfile.BadZipFile as _:
        logging.error("Failed to decompress '%s'.", zip_filename)
        raise
