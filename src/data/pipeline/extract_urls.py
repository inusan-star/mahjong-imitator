from datetime import datetime
import logging
import re

from sqlalchemy.exc import SQLAlchemyError
from tqdm import tqdm

import src.config as config
from src.db.game import GameRepository
from src.db.session import get_db_session


def run(year: int):
    """Extract log URLs."""
    txt_dir = config.TEXT_LOGS_DIR / str(year)

    if not txt_dir.exists():
        logging.info("Directory '%s' not found. Skipping extraction.", txt_dir)
        return

    txt_files = sorted(list(txt_dir.glob("*.txt")))

    if not txt_files:
        logging.info("No *.txt files found in '%s'. Skipping extraction.", txt_dir)
        return

    logging.info("Extracting URLs from *.txt files in '%s'...", txt_dir)

    games_to_insert = []
    log_id_pattern = re.compile(config.TENHO_LOG_ID_REGEX)

    for txt_file in tqdm(txt_files, desc=f"Extracting URL ({year})"):
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    parts = line.strip().split("|")

                    if len(parts) < 4:
                        continue

                    match = log_id_pattern.search(parts[3])

                    if not match:
                        continue

                    log_id = match.group(1)
                    time_str = parts[1].strip()
                    date_str = log_id[:8]

                    game = {
                        "log_url": config.TENHO_LOG_URL_FORMAT.format(log_id=log_id),
                        "played_at": datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H:%M"),
                    }
                    games_to_insert.append(game)

                except (ValueError, IndexError):
                    continue

    if not games_to_insert:
        logging.info("No URLs found in '%s'.", txt_dir)
        return

    logging.info("Successfully extracted URLs.")
    logging.info("Inserting URLs into the database...")

    try:
        with get_db_session() as session:
            game_repo = GameRepository(session)
            game_repo.bulk_insert(games_to_insert)

        logging.info("Successfully inserted URLs.")

    except SQLAlchemyError as _:
        logging.error("Failed to insert URLs.")
        raise
