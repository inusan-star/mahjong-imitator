import gzip
import logging
import pathlib
import shutil
import zipfile

from tqdm.rich import tqdm

import src.config as config


def run(year: int):
    """Decompress log archives."""
    zip_filename = config.TENHO_LOG_ZIP_FILENAME_FORMAT.format(year=year)
    zip_filepath = config.LOG_ZIPS_DIR / zip_filename
    gz_output_dir = config.GZIPPED_LOGS_DIR / str(year)

    if not zip_filepath.exists():
        logging.info("File '%s' not found. Skipping decompression.", zip_filename)
        return

    gz_output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Decompressing '%s' to '%s' ...", zip_filename, gz_output_dir)

    try:
        with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
            member_list = [
                member for member in zip_ref.infolist() if member.filename.endswith(".html.gz") and not member.is_dir()
            ]

            for member in tqdm(member_list, desc="Decompressing ZIP"):
                basename = pathlib.Path(member.filename).name
                target_path = gz_output_dir / basename

                if target_path.exists():
                    continue

                with zip_ref.open(member) as source_file, open(target_path, "wb") as target_file:
                    shutil.copyfileobj(source_file, target_file)

        logging.info("Successfully decompressed '%s'.", zip_filename)

    except zipfile.BadZipFile as _:
        logging.error("Failed to decompress '%s'.", zip_filename)
        raise

    txt_output_dir = config.TEXT_LOGS_DIR / str(year)
    gz_files = sorted(list(gz_output_dir.glob("*.html.gz")))

    if not gz_files:
        logging.info("No *.gz files found in '%s'. Skipping decompression.", gz_output_dir)
        return

    txt_output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Decompressing *.gz files to '%s' ...", txt_output_dir)

    try:
        for gz_file in tqdm(gz_files, desc="Decompressing GZ"):
            txt_filename = gz_file.name.replace(".html.gz", ".txt")
            txt_filepath = txt_output_dir / txt_filename

            if txt_filepath.exists():
                continue

            with gzip.open(gz_file, "rb") as f_gz, open(txt_filepath, "wt", encoding="utf-8") as f_txt:
                f_txt.write(f_gz.read().decode("utf-8"))

        logging.info("Successfully decompressed *.gz files.")

    except gzip.BadGzipFile as _:
        logging.error("Failed to decompress *.gz files.")
        raise
