import logging
from pathlib import Path
import warnings

from mjx import State
import numpy as np
from rich.logging import RichHandler
from sqlalchemy.orm import joinedload
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

import src.config as global_config
from src.db.game import Game
from src.db.game_player import GamePlayer, GamePlayerRepository
from src.db.log import Log
from src.db.player import Player, PlayerRepository
from src.db.session import get_db_session

from src.yaku import config as yaku_config
from src.yaku.feature.obs_encoder import ObservationEncoder
from src.yaku.feature.yaku_encoder import YakuEncoder


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
    input_history, output_history = [], []
    game_count, round_count, batch_count, data_count = 0, 0, 0, 0
    last_processed_at = None

    for parent_dir in [yaku_config.TRAIN_DIR, yaku_config.VALID_DIR]:
        (parent_dir / yaku_config.INPUT_NAME).mkdir(parents=True, exist_ok=True)
        (parent_dir / yaku_config.OUTPUT_NAME).mkdir(parents=True, exist_ok=True)

    obs_encoder = ObservationEncoder()
    yaku_encoder = YakuEncoder()

    with get_db_session() as session:
        target_filter = Player.game_count < yaku_config.MAX_PLAYER_GAME_COUNT

        logging.info("Finding target players.")

        player_repo = PlayerRepository(session)
        target_players = player_repo.find(target_filter)
        target_player_ids = {player.id for player in target_players}

        logging.info("Finding logs for target players.")

        logs = (
            session.query(Log)
            .options(joinedload(Log.game))
            .join(Game, Game.log_id == Log.id)
            .join(GamePlayer, GamePlayer.game_id == Game.id)
            .join(Player, Player.id == GamePlayer.player_id)
            .filter(Log.json_status == 1)
            .filter(target_filter)
            .distinct()
            .order_by(Log.played_at.asc())
            .all()
        )

        session.expunge_all()

    if not target_player_ids or not logs:
        logging.info("No players or logs found. Skipping.")
        return

    target_batch_total = yaku_config.TRAIN_BATCH_COUNT + yaku_config.VALID_BATCH_COUNT
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

            with get_db_session() as session:
                game_player_repo = GamePlayerRepository(session)
                game_players = game_player_repo.find(GamePlayer.game_id == log.game.id)
                seat_to_player = {game_player.seat_index: game_player.player_id for game_player in game_players}

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
                    if seat_to_player.get(win.who) not in target_player_ids:
                        continue

                    round_count += 1
                    yaku_vector = yaku_encoder.encode(win.yakus)

                    for obs, _ in state.past_decisions():
                        if obs.who() != win.who:
                            continue

                        input_history.append(obs_encoder.encode(obs))
                        output_history.append(yaku_vector)
                        data_count += 1
                        pbar.update(1)

                        if data_count % yaku_config.TARGET_BATCH_SIZE == 0:
                            batch_count += 1

                            if batch_count <= yaku_config.TRAIN_BATCH_COUNT:
                                target_dir = yaku_config.TRAIN_DIR

                            else:
                                target_dir = yaku_config.VALID_DIR

                            _save_batch(batch_count, input_history, output_history, target_dir)
                            input_history.clear()
                            output_history.clear()

                            if batch_count >= (yaku_config.TRAIN_BATCH_COUNT + yaku_config.VALID_BATCH_COUNT):
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
