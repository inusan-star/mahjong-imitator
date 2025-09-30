from datetime import datetime
import logging
import re

from sqlalchemy.exc import SQLAlchemyError
from tqdm.rich import tqdm

import src.config as config
from src.db.log import LogRepository
from src.db.session import get_db_session


def _bulk_insert_logs(logs: list[dict]):
    """Bulk insert logs."""
    if not logs:
        return

    try:
        with get_db_session() as session:
            log_repo = LogRepository(session)
            log_repo.bulk_insert(logs)
    except SQLAlchemyError:
        logging.error("Failed to insert logs. Halting.")
        raise


def run(year: int):
    """Extract source IDs."""
    txt_dir = config.TEXT_LOGS_DIR / str(year)

    if not txt_dir.exists():
        logging.info("Directory '%s' not found. Skipping.", txt_dir)
        return

    txt_files = sorted(list(txt_dir.glob("*.txt")))

    if not txt_files:
        logging.info("No *.txt files found in '%s'. Skipping.", txt_dir)
        return

    logging.info("Extracting source IDs from *.txt files in '%s' & Inserting logs into the database ...", txt_dir)

    logs_to_insert = []

    for txt_file in tqdm(txt_files, desc="Extracting & Inserting", unit="file"):
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                for line in f:
                    log_parts = line.strip().split("|")

                    if len(log_parts) < 4:
                        continue

                    time_str_match = re.match(re.compile(config.TENHO_LOG_TIME_REGEX), log_parts[0].strip())
                    source_id_match = re.search(re.compile(config.TENHO_LOG_ID_REGEX), log_parts[3].strip())

                    if not time_str_match or not source_id_match:
                        continue

                    time_str = time_str_match.group(0)
                    source_id = source_id_match.group(1)
                    date_str = source_id[:8]

                    log_record = {
                        "source_id": source_id,
                        "played_at": datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H:%M"),
                    }
                    logs_to_insert.append(log_record)

        except (OSError, ValueError, IndexError):
            logging.warning("Failed to extract source IDs. Skipping. File: %s", txt_file)
            continue

        if len(logs_to_insert) >= config.DB_BATCH_SIZE or (txt_file == txt_files[-1] and logs_to_insert):
            _bulk_insert_logs(logs_to_insert)
            logs_to_insert.clear()

    logging.info("Successfully extracted source IDs & inserting logs.")
