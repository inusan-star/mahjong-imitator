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
        logs_to_fetch = log_repo.find(*filters)

        if not logs_to_fetch:
            logging.info("No unprocessed logs found from database. Skipping fetching.")
            return

        mjlog_output_dir.mkdir(parents=True, exist_ok=True)

        logging.info("Successfully found unprocessed logs.")
        logging.info("Fetching mjlogs to '%s' ...", mjlog_output_dir)

        log_to_updates = []

        for log in tqdm(logs_to_fetch, desc="Fetching MJLOG", unit="log"):
            url = config.TENHO_LOG_URL_FORMAT.format(source_id=log.source_id)
            mjlog_filename = config.TENHO_MJLOG_FILENAME_FORMAT.format(source_id=log.source_id)
            mjlog_filepath = mjlog_output_dir / mjlog_filename
            relative_path = mjlog_filepath.relative_to(config.PROJECT_ROOT)

            log_to_update = {"id": log.id, "mjlog_status": 2}

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
                logging.warning("Failed to fetch mjlog. External ID: %s", log.source_id)

            log_to_updates.append(log_to_update)

        if not log_to_updates:
            logging.info("No mjlogs fetched. Skipping updating.")
            return

        logging.info("Successfully fetched mjlogs.")
        logging.info("Updating logs in the database ...")

        try:
            log_repo.bulk_update(log_to_updates)
            logging.info("Successfully updated logs.")
        except SQLAlchemyError as _:
            logging.error("Failed to update logs.")
            raise
