from datetime import datetime
import logging
from pathlib import Path
import random
import warnings

import mjx
from mjx import State
import numpy as np
from rich.logging import RichHandler
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

import src.config as global_config
from src.db.log import Log
from src.db.session import get_db_session

from src.yaku.exp1 import config as yaku_config
from src.yaku.exp1.feature.obs_encoder import ObservationEncoder
from src.yaku.exp1.feature.yaku_encoder import YakuEncoder


def setup_logging():
    """Set up logging configuration."""
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                log_time_format="[%Y-%m-%d %H:%M:%S]",
            )
        ],
    )


def _save_batch(batch_count: int, input_history: list, output_history: list, parent_dir: Path):
    """Save batch to the specified directory."""
    input_path = parent_dir / yaku_config.INPUT_NAME / f"history_{batch_count:03d}.npy"
    output_path = parent_dir / yaku_config.OUTPUT_NAME / f"history_{batch_count:03d}.npy"

    np.save(input_path, np.array(input_history, dtype=np.int32))
    np.save(output_path, np.array(output_history, dtype=np.float32))


def run():
    """Create yaku prediction dataset."""
    random.seed(yaku_config.SEED)
    np.random.seed(yaku_config.SEED)
    input_history, output_history = [], []
    game_count, round_count, batch_count, data_count = 0, 0, 0, 0
    last_processed_at = None

    for parent_dir in [yaku_config.TRAIN_DIR, yaku_config.VALID_DIR, yaku_config.TEST_DIR]:
        (parent_dir / yaku_config.INPUT_NAME).mkdir(parents=True, exist_ok=True)
        (parent_dir / yaku_config.OUTPUT_NAME).mkdir(parents=True, exist_ok=True)

    obs_encoder = ObservationEncoder()
    yaku_encoder = YakuEncoder()

    with get_db_session() as session:
        logging.info("Finding logs from database.")

        logs = (
            session.query(Log)
            .filter(Log.json_status == 1)
            .filter(Log.played_at >= datetime(yaku_config.START_YEAR, 1, 1))
            .order_by(Log.played_at.asc())
            .all()
        )

        session.expunge_all()

    if not logs:
        logging.info("No logs found. Skipping.")
        return

    target_batch_total = yaku_config.TRAIN_BATCH_COUNT + yaku_config.VALID_BATCH_COUNT + yaku_config.TEST_BATCH_COUNT
    total_target = yaku_config.TARGET_BATCH_SIZE * target_batch_total

    logging.info("Starting dataset creation.")

    with tqdm(total=total_target, desc="Creating", unit="data") as pbar:
        for log in logs:
            if batch_count >= target_batch_total:
                break

            json_path = global_config.PROJECT_ROOT / log.json_file_path

            if not json_path.exists():
                continue

            game_count += 1
            last_processed_at = log.played_at

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    round_lines = f.readlines()

            except (OSError, ValueError):
                continue

            for round_line in round_lines:
                if batch_count >= target_batch_total:
                    break

                state = State(round_line.strip())
                terminal = state.to_proto().round_terminal

                if not terminal.wins:
                    continue

                for win in terminal.wins:
                    yaku_vector = yaku_encoder.encode(win.yakus)

                    if np.sum(yaku_vector) == 0:
                        continue

                    round_count += 1

                    decisions = [
                        (obs, act) for obs, act in state.past_decisions()
                        if obs.who() == win.who and act.type() in [mjx.ActionType.DISCARD, mjx.ActionType.TSUMOGIRI]
                    ]

                    if not decisions:
                        continue

                    if batch_count >= yaku_config.TRAIN_BATCH_COUNT + yaku_config.VALID_BATCH_COUNT:
                        decisions = [random.choice(decisions)]

                    for obs, _ in decisions:
                        input_history.append(obs_encoder.encode(obs))
                        output_history.append(yaku_vector)
                        data_count += 1
                        pbar.update(1)

                        if data_count % yaku_config.TARGET_BATCH_SIZE == 0:
                            batch_count += 1

                            if batch_count <= yaku_config.TRAIN_BATCH_COUNT:
                                target_dir = yaku_config.TRAIN_DIR

                            elif batch_count <= yaku_config.TRAIN_BATCH_COUNT + yaku_config.VALID_BATCH_COUNT:
                                target_dir = yaku_config.VALID_DIR

                            else:
                                target_dir = yaku_config.TEST_DIR

                            _save_batch(batch_count, input_history, output_history, target_dir)
                            input_history.clear()
                            output_history.clear()

                            if batch_count >= target_batch_total:
                                break

    logging.info(
        "Dataset creation completed. Games: %d, Rounds: %d, Datas: %d, Last Processed At: %s",
        game_count,
        round_count,
        data_count,
        last_processed_at,
    )


if __name__ == "__main__":
    setup_logging()
    run()