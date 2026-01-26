import json
import logging
import random
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple

import duckdb
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

import src.config as global_config
from src.yaku.common import config as common_config


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


def _load_data_with_duckdb() -> Tuple[Dict[int, int], Dict[str, dict], int, Dict[int, str]]:
    """Load data using DuckDB."""
    dumps_dir = global_config.DUMPS_DIR

    logs_parquet = f"{dumps_dir}/logs/*.parquet"
    games_parquet = f"{dumps_dir}/games/*.parquet"
    rounds_parquet = f"{dumps_dir}/rounds/*.parquet"
    round_yaku_parquet = f"{dumps_dir}/round_yaku/*.parquet"
    yaku_parquet = f"{dumps_dir}/yaku/*.parquet"

    connect = duckdb.connect()

    yaku_df = connect.execute(f"SELECT id, name FROM read_parquet('{yaku_parquet}')").df()
    yaku_id_to_name = dict(zip(yaku_df["id"], yaku_df["name"]))
    excluded_yaku_ids = set(yaku_df[yaku_df["name"].isin(common_config.EXCLUDED_YAKU_NAMES)]["id"])

    query = f"""
        SELECT
            r.id as round_id,
            l.json_file_path,
            r.round_index,
            r.honba,
            list(DISTINCT ry.yaku_id) as yaku_ids
        FROM read_parquet('{logs_parquet}') l
        JOIN read_parquet('{games_parquet}') g ON l.id = g.log_id
        JOIN read_parquet('{rounds_parquet}') r ON g.id = r.game_id
        JOIN read_parquet('{round_yaku_parquet}') ry ON r.id = ry.round_id
        WHERE l.json_status = 1 AND r.is_agari = 1
        GROUP BY r.id, l.json_file_path, r.round_index, r.honba
    """

    round_df = connect.execute(query).df()
    total_available_rounds = len(round_df)

    yaku_threshold_count = total_available_rounds * common_config.YAKU_THRESHOLD_RATIO

    temp_counts: Dict[int, int] = defaultdict(int)

    for row in round_df.itertuples(index=False):
        for yaku_id in row.yaku_ids:
            temp_counts[yaku_id] += 1

    valid_yaku_ids = {
        yaku_id
        for yaku_id, count in temp_counts.items()
        if yaku_id not in excluded_yaku_ids and count >= yaku_threshold_count
    }

    game_data_map: Dict[str, dict] = {}
    yaku_counts: Dict[int, int] = defaultdict(int)

    for row in tqdm(round_df.itertuples(index=False), total=len(round_df), desc="Mapping", unit="round"):
        filtered_ids = [yaku_id for yaku_id in row.yaku_ids if yaku_id in valid_yaku_ids]

        if filtered_ids:
            game_data_map[str(row.round_id)] = {
                "yaku_ids": filtered_ids,
                "path": row.json_file_path,
                "round_index": row.round_index,
                "honba": row.honba,
            }

            for yaku_id in filtered_ids:
                yaku_counts[yaku_id] += 1

    connect.close()

    return dict(yaku_counts), game_data_map, total_available_rounds, yaku_id_to_name


def _calculate_yaku_rarity_rank(yaku_counts: Dict[int, int]) -> Dict[int, int]:
    """Calculate yaku rarity rank."""
    yaku_ids = sorted(yaku_counts.keys(), key=lambda x: yaku_counts[x])

    return {yaku_id: i for i, yaku_id in enumerate(yaku_ids)}


def _get_yaku_split_targets(yaku_counts: Dict[int, int], total_available_rounds: int) -> Dict[int, Dict[str, float]]:
    """Get yaku split targets."""
    splits = {"train": common_config.TRAIN_RATIO, "valid": common_config.VALID_RATIO, "test": common_config.TEST_RATIO}
    yaku_split_targets: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for yaku_id, yaku_count in yaku_counts.items():
        for split, ratio in splits.items():
            yaku_split_targets[yaku_id][split] = (
                (yaku_count / total_available_rounds) * common_config.TOTAL_EXTRACT_ROUNDS * ratio
            )

    return {key: dict(value) for key, value in yaku_split_targets.items()}


def _allocate_games(
    game_data_map: Dict[str, dict],
    yaku_rarity_rank: Dict[int, int],
    split_targets: Dict[str, int],
    yaku_split_targets: Dict[int, Dict[str, float]],
) -> Tuple[Dict[str, dict], Dict[str, int], Dict[int, Dict[str, int]]]:
    """Allocate rounds to splits."""
    splits: List[str] = ["train", "valid", "test"]
    game_allocation_map: Dict[str, dict] = {}
    current_split_count: Dict[str, int] = {split: 0 for split in splits}
    yaku_split_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    yaku_satisfied: Dict[int, bool] = defaultdict(bool)

    def _get_rarest_rank(round_id: str) -> float:
        yaku_ids = game_data_map[round_id]["yaku_ids"]

        if not yaku_ids:
            return float("inf")

        return min(yaku_rarity_rank[yaku_id] for yaku_id in yaku_ids)

    def _get_satisfaction_ratio(split: str, yaku_id: int) -> float:
        target = yaku_split_targets[yaku_id][split]

        if target == 0 or current_split_count[split] >= split_targets[split]:
            return float("inf")

        return yaku_split_counts[yaku_id][split] / target

    round_ids = list(game_data_map.keys())
    random.shuffle(round_ids)

    with tqdm(total=common_config.TOTAL_EXTRACT_ROUNDS, desc="Allocating", unit="round") as pbar:
        for round_id in round_ids:
            if all(yaku_satisfied.values()) and len(yaku_satisfied) > 0:
                logging.info("All yaku targets satisfied. Early stopping.")
                break

            if pbar.n >= common_config.TOTAL_EXTRACT_ROUNDS:
                break

            yaku_ids: List[int] = game_data_map[round_id]["yaku_ids"]

            for yaku_id in yaku_ids:
                total_assigned = sum(yaku_split_counts[yaku_id][split] for split in splits)
                total_target = sum(yaku_split_targets[yaku_id].values())

                if total_assigned >= total_target:
                    yaku_satisfied[yaku_id] = True

            unsatisfied_yaku_ids = [yaku_id for yaku_id in yaku_ids if not yaku_satisfied[yaku_id]]

            if not unsatisfied_yaku_ids:
                continue

            representative_yaku_id = min(unsatisfied_yaku_ids, key=lambda x: yaku_rarity_rank[x])

            best_split: str = min(splits, key=lambda s: _get_satisfaction_ratio(s, representative_yaku_id))

            if _get_satisfaction_ratio(best_split, representative_yaku_id) == float("inf"):
                available_splits = [split for split in splits if current_split_count[split] < split_targets[split]]

                if not available_splits:
                    continue

                best_split = min(available_splits, key=lambda s: _get_satisfaction_ratio(s, representative_yaku_id))

            game_allocation_map[round_id] = {
                "path": game_data_map[round_id]["path"],
                "round_index": game_data_map[round_id]["round_index"],
                "honba": game_data_map[round_id]["honba"],
                "split": best_split,
            }
            current_split_count[best_split] += 1

            for yaku_id in yaku_ids:
                yaku_split_counts[yaku_id][best_split] += 1

            pbar.update(1)

    return game_allocation_map, current_split_count, {key: dict(value) for key, value in yaku_split_counts.items()}


def main():
    """Split rounds into train, valid, and test sets."""
    setup_logging()

    random.seed(common_config.SEED)

    common_config.YAKU_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Loading data with DuckDB ...")

    yaku_counts, game_data_map, total_available_rounds, id_to_name = _load_data_with_duckdb()

    logging.info("Successfully loaded data.")
    logging.info("Calculating yaku rarity rank ...")

    yaku_rarity_rank = _calculate_yaku_rarity_rank(yaku_counts)

    logging.info("Successfully calculated yaku rarity rank.")
    logging.info("Getting yaku split targets ...")

    yaku_split_targets = _get_yaku_split_targets(yaku_counts, total_available_rounds)

    logging.info("Successfully got yaku split targets.")
    logging.info("Allocating rounds to splits ...")

    split_targets = {
        "train": int(common_config.TOTAL_EXTRACT_ROUNDS * common_config.TRAIN_RATIO),
        "valid": int(common_config.TOTAL_EXTRACT_ROUNDS * common_config.VALID_RATIO),
        "test": int(common_config.TOTAL_EXTRACT_ROUNDS * common_config.TEST_RATIO),
    }

    game_allocation_map, current_split_count, yaku_distribution_stats = _allocate_games(
        game_data_map, yaku_rarity_rank, split_targets, yaku_split_targets
    )

    for split in ["train", "valid", "test"]:
        split_total = current_split_count[split]

        for yaku_id, counts in yaku_distribution_stats.items():
            count = counts.get(split, 0)
            ratio = count / split_total if split_total > 0 else 0

            if ratio < common_config.YAKU_THRESHOLD_RATIO:
                logging.warning(
                    "[%s] %s distribution is low: %.3f%% (n=%d)", split.upper(), id_to_name[yaku_id], ratio * 100, count
                )

    logging.info("Successfully allocated rounds to splits.")

    with open(common_config.SPLITS_FILE, "w", encoding="utf-8") as f:
        json.dump(game_allocation_map, f, indent=2, ensure_ascii=False)

    console = Console()
    table = Table(title=f"Dataset Yaku Distribution ({len(yaku_counts)} Classes)")
    table.add_column("Yaku Name", justify="left", style="cyan")
    table.add_column("Ideal Ratio (%)", justify="right", style="yellow")
    table.add_column("Total Count", justify="right", style="magenta")
    table.add_column("Train (Count/%)", justify="right")
    table.add_column("Valid (Count/%)", justify="right")
    table.add_column("Test (Count/%)", justify="right")
    table.add_column("Dataset Ratio (%)", justify="right", style="green")

    for yaku_id, counts in sorted(yaku_distribution_stats.items(), key=lambda x: sum(x[1].values()), reverse=True):
        ideal_percentage = (yaku_counts[yaku_id] / total_available_rounds) * 100
        total_in_dataset = sum(counts.values())
        percentage = (total_in_dataset / common_config.TOTAL_EXTRACT_ROUNDS) * 100
        train_percentage = (
            (counts["train"] / current_split_count["train"] * 100) if current_split_count["train"] > 0 else 0
        )
        valid_percentage = (
            (counts["valid"] / current_split_count["valid"] * 100) if current_split_count["valid"] > 0 else 0
        )
        test_percentage = (counts["test"] / current_split_count["test"] * 100) if current_split_count["test"] > 0 else 0

        table.add_row(
            id_to_name[yaku_id],
            f"{ideal_percentage:.2f}%",
            f"{total_in_dataset:,}",
            f"{counts['train']:,} ({train_percentage:.2f}%)",
            f"{counts['valid']:,} ({valid_percentage:.2f}%)",
            f"{counts['test']:,} ({test_percentage:.2f}%)",
            f"{percentage:.2f}%",
        )

    console.print(table)

    logging.info(
        "Split completed. Train: %d, Valid: %d, Test: %d",
        current_split_count["train"],
        current_split_count["valid"],
        current_split_count["test"],
    )


if __name__ == "__main__":
    main()
