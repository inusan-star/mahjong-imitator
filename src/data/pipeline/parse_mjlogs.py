import logging
import re
from typing import cast, Optional
import urllib.parse

from sqlalchemy import extract
from sqlalchemy.exc import SQLAlchemyError
from tqdm.rich import tqdm

import src.config as global_config
import src.data.config as data_config
from src.db.game import Game
from src.db.game_player import GamePlayer
from src.db.log import Log, LogRepository
from src.db.player import Player, PlayerRepository
from src.db.session import get_db_session


def _is_target_rule(game_type: int) -> bool:
    """
    Determine whether the rule is a target.
    Target: Red tiles, Kuitan, South round, 4-player.
    """
    return (game_type & 0x02) == 0 and (game_type & 0x04) == 0 and (game_type & 0x08) != 0 and (game_type & 0x10) == 0


def _parse_ranks(scores: list[int]) -> list[int]:
    """Calculate ranks from scores."""
    score_with_seats = []

    for i, score in enumerate(scores):
        score_with_seats.append((score, i))

    sorted_scores = sorted(score_with_seats, key=lambda x: x[0], reverse=True)

    ranks = [0] * 4

    for rank, (_, seat_index) in enumerate(sorted_scores, start=1):
        ranks[seat_index] = rank

    return ranks


def _save_game_player_metadata(games_data: list[dict]):
    """Save game and player metadata."""
    if not games_data:
        return

    try:
        with get_db_session() as session:
            player_repo = PlayerRepository(session)
            player_cache: dict[str, Player] = {}

            for data in games_data:
                game = Game(
                    log_id=data["log_id"],
                    game_type=data["game_type"],
                    lobby_id=data["lobby_id"],
                )
                session.add(game)
                session.flush()

                played_at = data["played_at"]

                for i, p_data in enumerate(data["players"]):
                    name = p_data["name"]
                    gender = p_data["gender"]
                    dan = p_data["dan"]
                    rate = p_data["rate"]
                    score = p_data["score"]
                    rank = p_data["rank"]

                    player: Optional[Player] = None

                    if name in player_cache:
                        player = player_cache[name]

                    else:
                        found_players = player_repo.find(Player.name == name)

                        if found_players:
                            player = found_players[0]
                            player = cast(Player, player)
                            player_cache[name] = player

                    if not player:
                        player = Player(
                            name=name,
                            gender=gender,
                            last_dan=dan,
                            last_rate=rate,
                            max_dan=dan,
                            max_rate=rate,
                            last_played_at=played_at,
                            game_count=0,
                            first_count=0,
                            second_count=0,
                            third_count=0,
                            fourth_count=0,
                        )
                        session.add(player)
                        session.flush()
                        player_cache[name] = player

                    if played_at >= player.last_played_at:
                        player.last_dan = dan
                        player.last_rate = rate
                        player.last_played_at = played_at

                    if dan > player.max_dan:
                        player.max_dan = dan

                    if rate > player.max_rate:
                        player.max_rate = rate

                    player.game_count += 1

                    if rank == 1:
                        player.first_count += 1

                    elif rank == 2:
                        player.second_count += 1

                    elif rank == 3:
                        player.third_count += 1

                    elif rank == 4:
                        player.fourth_count += 1

                    game_player = GamePlayer(
                        game_id=game.id,
                        player_id=player.id,
                        seat_index=i,
                        dan=dan,
                        rate=rate,
                        score=score,
                        rank=rank,
                    )
                    session.add(game_player)

    except SQLAlchemyError:
        logging.error("Failed to save game and player metadata. Halting.")
        raise


def _parse_mjlog(log_id: int, mjlog_file_path: str, played_at) -> Optional[dict]:
    """Parse a single mjlog file."""
    if not mjlog_file_path:
        return None

    mjlog_path = global_config.PROJECT_ROOT / mjlog_file_path

    if not mjlog_path.exists():
        logging.info("MJLOG file not found: %s", mjlog_path)
        return None

    try:
        try:
            content = mjlog_path.read_text(encoding="shift_jis")

        except UnicodeDecodeError:
            content = mjlog_path.read_text(encoding="utf-8")

    except Exception:
        logging.error("Failed to read MJLOG file: %s", mjlog_path)
        return None

    # Parse Header
    go_match = re.search(data_config.MJLOG_GO_REGEX, content)

    if not go_match:
        return None

    game_type = int(go_match.group(1))
    lobby_id = int(go_match.group(2))

    if not _is_target_rule(game_type):
        return None

    un_match = re.search(data_config.MJLOG_UN_REGEX, content)

    if not un_match:
        return None

    raw_names = [un_match.group(1), un_match.group(2), un_match.group(3), un_match.group(4)]
    dans = [int(x) for x in un_match.group(5).split(",")]
    rates = [float(x) for x in un_match.group(6).split(",")]
    sxs = un_match.group(7).split(",")

    # Parse Footer
    owari_matches = list(re.finditer(data_config.MJLOG_OWARI_REGEX, content))

    if not owari_matches:
        return None

    owari_values = owari_matches[-1].group(1).split(",")

    if len(owari_values) < 8:
        return None

    scores = [int(owari_values[i * 2]) * 100 for i in range(4)]
    ranks = _parse_ranks(scores)

    players_data = []

    for i in range(4):
        players_data.append(
            {
                "name": urllib.parse.unquote(raw_names[i]),
                "gender": sxs[i],
                "dan": dans[i],
                "rate": rates[i],
                "score": scores[i],
                "rank": ranks[i],
            }
        )

    return {
        "log_id": log_id,
        "played_at": played_at,
        "game_type": game_type,
        "lobby_id": lobby_id,
        "players": players_data,
    }


def run(year: int):
    """Parse mjlogs."""
    logging.info("Finding unprocessed logs from database ...")

    with get_db_session() as session:
        log_repo = LogRepository(session)
        logs = log_repo.find(
            Log.mjlog_status == 1,
            Game.id.is_(None),
            extract("year", Log.played_at) == year,
            outer_joins=[Game],
            order_by=[Log.played_at.asc()],
        )
        logs_to_process = [(log.id, log.source_id, log.mjlog_file_path, log.played_at) for log in logs]

    if not logs_to_process:
        logging.info("No logs to process from database. Skipping.")
        return

    logging.info("Successfully found logs to process.")
    logging.info("Parsing mjlogs & Inserting games and players metadata into the database ...")

    game_player_to_insert = []

    for log_id, source_id, mjlog_file_path, played_at in tqdm(logs_to_process, desc="Parsing & Inserting", unit="log"):
        game_player_data = _parse_mjlog(log_id, mjlog_file_path, played_at)

        if game_player_data:
            game_player_to_insert.append(game_player_data)

        if len(game_player_to_insert) >= global_config.DB_BATCH_SIZE or (
            (log_id, source_id, mjlog_file_path, played_at) == logs_to_process[-1] and game_player_to_insert
        ):
            _save_game_player_metadata(game_player_to_insert)
            game_player_to_insert.clear()

    logging.info("Successfully processed logs & inserted games and players metadata.")
