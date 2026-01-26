import csv
import json
import logging
import multiprocessing as mp
from pathlib import Path
import random
from typing import List, Tuple, Dict, Any
import warnings

import mjx
from mjx import State
import numpy as np
from rich.logging import RichHandler
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

import src.config as global_config
import src.yaku.common.config as common_config
import src.yaku.exp4.config as exp4_config  # exp1から変更
from src.yaku.common.yaku_encoder import YakuEncoder
from src.yaku.exp4.feature.obs_encoder import ObservationEncoder  # exp4用のエンコーダに変更

worker_obs_encoder: ObservationEncoder = None  # type: ignore
worker_yaku_encoder: YakuEncoder = None  # type: ignore


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


def worker_init():
    """Initialize encoders in each worker process."""
    global worker_obs_encoder, worker_yaku_encoder
    worker_obs_encoder = ObservationEncoder()
    worker_yaku_encoder = YakuEncoder()


def _save_chunk(chunk_count: int, input_list: List[np.ndarray], output_list: List[np.ndarray], parent_directory: Path):
    """Save a chunk of data."""
    input_directory = parent_directory / exp4_config.INPUT_FILENAME_PREFIX
    output_directory = parent_directory / exp4_config.OUTPUT_FILENAME_PREFIX

    input_directory.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(parents=True, exist_ok=True)

    input_path = input_directory / f"chunk_{chunk_count:03d}.npy"
    output_path = output_directory / f"chunk_{chunk_count:03d}.npy"

    # Transformer用入力は (SEQ_LEN, FEATURE_DIM) の形状。float32で保存
    np.save(input_path, np.array(input_list, dtype=np.float32))
    np.save(output_path, np.array(output_list, dtype=np.float32))


def _process_round(args: Tuple[str, Dict[str, Any]]) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Process single round (Worker function)."""
    round_id, info = args
    split = info["split"]
    json_path = global_config.PROJECT_ROOT / info["path"]

    if not json_path.exists():
        logging.error("JSON path does not exist: %s", json_path)
        return []

    try:
        with open(json_path, "r", encoding="utf-8") as file:
            round_lines = file.readlines()

        round_index = info["round_index"]
        honba = info["honba"]
        target_line = ""

        for line in round_lines:
            line_json = json.loads(line.strip())
            init_score = line_json.get("publicObservation", {}).get("initScore", {})

            if init_score.get("round", 0) == round_index and init_score.get("honba", 0) == honba:
                target_line = line
                break

        if not target_line:
            logging.error("Target line not found for round_id: %s", round_id)
            return []

        state = State(target_line.strip())
        terminal = state.to_proto().round_terminal

        if not terminal:
            logging.error("Round terminal not found for round_id: %s", round_id)
            return []

        results = []

        for win in terminal.wins:
            if not win.yakus and not win.yakumans:
                continue

            all_yakus = list(win.yakus) + list(win.yakumans)
            yaku_vector = worker_yaku_encoder.encode(all_yakus)

            if np.sum(yaku_vector.numpy()) == 0:
                continue

            # [cite_start]既存研究のターゲットアクション [cite: 185, 186]
            target_action_types = [
                mjx.ActionType.DISCARD,
                mjx.ActionType.TSUMOGIRI,
                mjx.ActionType.RIICHI,
                mjx.ActionType.CHI,
                mjx.ActionType.PON,
                mjx.ActionType.OPEN_KAN,
                mjx.ActionType.CLOSED_KAN,
                mjx.ActionType.ADDED_KAN,
            ]

            decisions = [
                (observation, action)
                for observation, action in state.past_decisions()
                if observation.who() == win.who and action.type() in target_action_types
            ]

            if not decisions:
                continue

            for observation, _ in decisions:
                results.append(
                    (
                        split,
                        worker_obs_encoder.encode(observation),  # (35, 22) 形式を生成
                        yaku_vector.numpy(),
                    )
                )

        if not results:
            logging.error("No valid results for round_id: %s", round_id)

        return results

    except Exception as error:
        logging.error("Error processing round %s: %s", round_id, error)
        return []


def main():
    """Create dataset based on splits using multiprocessing."""
    random.seed(common_config.SEED)
    np.random.seed(common_config.SEED)

    if not common_config.SPLITS_FILE.exists():
        logging.error("Splits file not found.")
        return

    with open(common_config.SPLITS_FILE, "r", encoding="utf-8") as file:
        game_allocation_map = json.load(file)

    logging.info("Starting dataset creation for experiment 4 (Transformer) ...")

    split_directory_map = {"train": exp4_config.TRAIN_DIR, "valid": exp4_config.VALID_DIR, "test": exp4_config.TEST_DIR}
    buffers = {split: {"input": [], "output": []} for split in split_directory_map}
    chunk_counts = {split: 0 for split in split_directory_map}
    split_data_counts = {split: 0 for split in split_directory_map}
    total_data_count = 0
    yaku_distribution = {split: None for split in split_directory_map}

    with mp.Pool(processes=mp.cpu_count(), initializer=worker_init) as pool:
        with tqdm(total=len(game_allocation_map), desc="Creating", unit="round") as progress_bar:
            for results in pool.imap_unordered(_process_round, game_allocation_map.items()):
                for split, input_data, output_data in results:
                    buffers[split]["input"].append(input_data)
                    buffers[split]["output"].append(output_data)

                    total_data_count += 1
                    split_data_counts[split] += 1

                    yaku_np = output_data.astype(np.int64)

                    if yaku_distribution[split] is None:
                        yaku_distribution[split] = yaku_np.copy()
                    else:
                        yaku_distribution[split] += yaku_np

                    if len(buffers[split]["input"]) >= exp4_config.CHUNK_SIZE:
                        chunk_counts[split] += 1
                        _save_chunk(
                            chunk_counts[split],
                            buffers[split]["input"],
                            buffers[split]["output"],
                            split_directory_map[split],
                        )
                        buffers[split]["input"].clear()
                        buffers[split]["output"].clear()

                progress_bar.update(1)

    for split, directory in split_directory_map.items():
        if buffers[split]["input"]:
            chunk_counts[split] += 1
            _save_chunk(chunk_counts[split], buffers[split]["input"], buffers[split]["output"], directory)

    logging.info("Dataset creation completed.")

    exp4_config.RESULT_DIR.mkdir(parents=True, exist_ok=True)

    final_yaku_encoder = YakuEncoder()
    yaku_names = final_yaku_encoder.labels

    with open(exp4_config.DATA_STATS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Statistic", "Train", "Valid", "Test", "Total"])
        writer.writerow(
            [
                "Data Count",
                split_data_counts["train"],
                split_data_counts["valid"],
                split_data_counts["test"],
                total_data_count,
            ]
        )
        writer.writerow(
            [
                "Chunk Count",
                chunk_counts["train"],
                chunk_counts["valid"],
                chunk_counts["test"],
                sum(chunk_counts.values()),
            ]
        )
        writer.writerow([])
        writer.writerow(["Yaku Name", "Train", "Valid", "Test"])

        num_yaku = len(yaku_distribution["train"]) if yaku_distribution["train"] is not None else 0
        for i in range(num_yaku):
            writer.writerow(
                [
                    yaku_names[i],
                    yaku_distribution["train"][i] if yaku_distribution["train"] is not None else 0,
                    yaku_distribution["valid"][i] if yaku_distribution["valid"] is not None else 0,
                    yaku_distribution["test"][i] if yaku_distribution["test"] is not None else 0,
                ]
            )


if __name__ == "__main__":
    setup_logging()
    main()
