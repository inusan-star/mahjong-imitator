import logging
import subprocess

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
    """Convert mjlogs to jsons."""
    json_output_dir = config.JSON_LOGS_DIR / str(year)

    logging.info("Finding unprocessed logs from database ...")

    with get_db_session() as session:
        log_repo = LogRepository(session)
        filters = (
            Log.mjlog_status == 1,
            Log.json_status == 0,
            extract("year", Log.played_at) == year,
        )
        logs = log_repo.find(*filters, order_by=[Log.played_at.asc()])
        logs_to_convert = [(log.id, log.source_id, log.mjlog_file_path) for log in logs]

    if not logs_to_convert:
        logging.info("No unprocessed logs found from database. Skipping.")
        return

    json_output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Successfully found unprocessed logs.")
    logging.info("Converting mjlogs to jsons at '%s' & Updating logs in the database ...", json_output_dir)

    log_to_updates = []

    for log_id, source_id, mjlog_file_path in tqdm(logs_to_convert, desc="Converting & Updating", unit="log"):
        json_filepath = json_output_dir / f"{source_id}.json"
        relative_path = json_filepath.relative_to(config.PROJECT_ROOT)

        log_to_update = {"id": log_id, "json_status": 2}

        try:
            try:
                mjlog_text = (config.PROJECT_ROOT / mjlog_file_path).read_text(encoding="shift_jis")
            except UnicodeDecodeError:
                mjlog_text = (config.PROJECT_ROOT / mjlog_file_path).read_text(encoding="utf-8")

            process = subprocess.run(
                ["mjxc", "convert", "--to-mjxproto"],
                input=mjlog_text,
                capture_output=True,
                check=True,
                text=True,
                timeout=config.SUBPROCESS_TIMEOUT,
            )

            with open(json_filepath, "w", encoding="utf-8") as f:
                f.write(process.stdout.strip())

            log_to_update["json_status"] = 1
            log_to_update["json_file_path"] = str(relative_path)

        except subprocess.CalledProcessError:
            logging.error("Failed to convert mjlog to json. Source ID: %s", source_id)

        except subprocess.TimeoutExpired:
            logging.error("Timeout converting mjlog to json. Source ID: %s", source_id)

        log_to_updates.append(log_to_update)

        if len(log_to_updates) >= config.DB_BATCH_SIZE or (
            (log_id, source_id, mjlog_file_path) == logs_to_convert[-1] and log_to_updates
        ):
            _bulk_update_logs(log_to_updates)
            log_to_updates.clear()

    logging.info("Successfully converted mjlogs to jsons & updated logs.")
