import logging
import random
import requests
import time
import xml.dom.minidom

from sqlalchemy import extract
from sqlalchemy.exc import SQLAlchemyError
from tqdm.rich import tqdm

import src.config as config
from src.db.log import Log, LogRepository
from src.db.session import get_db_session


def _bulk_update_logs(logs: list[dict]):
    """Bulk update logs."""
    if not logs:
        return

    try:
        with get_db_session() as session:
            log_repo = LogRepository(session)
            log_repo.bulk_update(logs)
    except SQLAlchemyError:
        logging.error("Failed to update logs. Halting.")
        raise


def run(year: int):
    """Fetch mjlogs."""
    mjlog_output_dir = config.MJLOGS_DIR / str(year)

    logging.info("Finding unprocessed logs from database ...")

    with get_db_session() as session:
        log_repo = LogRepository(session)
        filters = (
            Log.mjlog_status == 0,
            extract("year", Log.played_at) == year,
        )
        logs_to_fetch = [(log.id, log.source_id) for log in log_repo.find(*filters)]

    if not logs_to_fetch:
        logging.info("No unprocessed logs found from database. Skipping.")
        return

    mjlog_output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Successfully found unprocessed logs.")
    logging.info("Fetching mjlogs to '%s' & Updating logs in the database ...", mjlog_output_dir)

    log_to_updates = []

    for (log_id, source_id) in tqdm(logs_to_fetch, desc="Fetching & Updating", unit="log"):
        url = config.TENHO_LOG_URL_FORMAT.format(source_id=source_id)
        mjlog_filename = config.TENHO_MJLOG_FILENAME_FORMAT.format(source_id=source_id)
        mjlog_filepath = mjlog_output_dir / mjlog_filename
        relative_path = mjlog_filepath.relative_to(config.PROJECT_ROOT)

        log_to_update = {"id": log_id, "mjlog_status": 2}

        try:
            time.sleep(random.uniform(config.REQUEST_SLEEP_MIN, config.REQUEST_SLEEP_MAX))
            response = requests.get(url, headers=config.TENHO_HEADERS, timeout=config.REQUESTS_TIMEOUT)
            response.raise_for_status()

            formatted_mjlog = xml.dom.minidom.parseString(response.content).toprettyxml(indent="  ")

            with open(mjlog_filepath, "wb") as f:
                f.write(formatted_mjlog.encode("utf-8"))

            log_to_update["mjlog_status"] = 1
            log_to_update["mjlog_file_path"] = str(relative_path)

        except requests.exceptions.RequestException:
            logging.error("Failed to fetch mjlog. Source ID: %s", source_id)

        log_to_updates.append(log_to_update)

        if len(log_to_updates) >= config.DB_BATCH_SIZE or ((log_id, source_id) == logs_to_fetch[-1] and log_to_updates):
            _bulk_update_logs(log_to_updates)
            log_to_updates.clear()

    logging.info("Successfully fetched mjlogs & updated logs.")
