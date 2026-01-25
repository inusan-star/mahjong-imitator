import csv
from collections import defaultdict
from datetime import datetime
import logging
from pathlib import Path
import warnings

import mjx
from mjx import State
from rich.logging import RichHandler
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.console import Group
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

import src.config as global_config
from src.db.log import Log
from src.db.session import get_db_session
from src.db.yaku import YakuRepository
from src.yaku.exp1 import config as yaku_config


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


def generate_status_view(meta_counts, yaku_counts, yaku_map, ordered_yaku_ids):
    """全役の進捗をID順に表示するUIを生成する。"""

    # 1. 基本統計パネル
    stats_table = Table(title="📊 処理統計", box=None, expand=True)
    stats_table.add_column("項目", style="cyan")
    stats_table.add_column("現在の累計", justify="right", style="magenta")
    stats_table.add_row("試合数 (Games)", f"{meta_counts['games']['total']:,}")
    stats_table.add_row("ラウンド数 (Rounds)", f"{meta_counts['rounds']['total']:,}")
    stats_table.add_row("局面数 (Datas)", f"{meta_counts['datas']['total']:,}")

    # 2. 全役進捗テーブルを生成する関数
    def create_yaku_table(ids):
        table = Table(box=None, expand=True, header_style="bold green")
        table.add_column("ID", justify="right", style="dim", width=3)
        table.add_column("役名", style="white")
        table.add_column("出現数", justify="right", style="yellow")
        for yid in ids:
            name = yaku_map.get(yid, "Unknown")
            count = yaku_counts[yid]["total"]
            table.add_row(str(yid), name, f"{count:,}")
        return table

    # 55役を半分に分ける
    mid = (len(ordered_yaku_ids) + 1) // 2
    left_ids = ordered_yaku_ids[:mid]
    right_ids = ordered_yaku_ids[mid:]

    # グリッドの構築
    yaku_grid = Table.grid(expand=True)
    # 修正ポイント: expand=True を ratio=1 に変更
    yaku_grid.add_column(ratio=1)
    yaku_grid.add_column(width=2)  # 隙間
    yaku_grid.add_column(ratio=1)
    yaku_grid.add_row(create_yaku_table(left_ids), "", create_yaku_table(right_ids))

    return Group(
        Panel(stats_table, border_style="blue", title="[bold]Overall Progress[/]"),
        Panel(yaku_grid, border_style="green", title="[bold]Yaku Counts (Order by ID)[/]"),
    )


def run():
    """Analyze yaku distribution and export to CSV."""
    setup_logging()

    # 1. DBから役の定義を取得
    yaku_map = {}
    ordered_yaku_ids = []

    with get_db_session() as session:
        yaku_repo = YakuRepository(session)
        all_yaku = yaku_repo.find()
        for yaku in all_yaku:
            if yaku.name == "人和":
                continue
            yaku_map[yaku.id] = yaku.name
            ordered_yaku_ids.append(yaku.id)

    # 2. ログの取得
    year = yaku_config.START_YEAR
    with get_db_session() as session:
        logs = (
            session.query(Log)
            .filter(Log.json_status == 1)
            .filter(Log.played_at >= datetime(year, 1, 1))
            .filter(Log.played_at < datetime(year + 1, 1, 1))
            .order_by(Log.played_at.asc())
            .all()
        )
        session.expunge_all()

    if not logs:
        return

    # 3. 集計
    yaku_counts = defaultdict(lambda: defaultdict(int))
    meta_counts = defaultdict(lambda: defaultdict(int))

    logging.info(f"Analyzing {len(logs)} logs for {year}...")

    # ライブ表示
    with Live(generate_status_view(meta_counts, yaku_counts, yaku_map, ordered_yaku_ids), refresh_per_second=4) as live:
        for i, log in enumerate(logs):
            json_path = global_config.PROJECT_ROOT / log.json_file_path
            if not json_path.exists():
                continue

            month = log.played_at.month
            meta_counts["games"][month] += 1
            meta_counts["games"]["total"] += 1

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    round_lines = f.readlines()

                for round_line in round_lines:
                    state = State(round_line.strip())
                    terminal = state.to_proto().round_terminal
                    if not terminal.wins:
                        continue

                    meta_counts["rounds"][month] += 1
                    meta_counts["rounds"]["total"] += 1

                    for win in terminal.wins:
                        has_target_yaku = False
                        for mjx_yaku_id in win.yakus:
                            db_yaku_id = mjx_yaku_id + 1
                            if db_yaku_id in yaku_map:
                                yaku_counts[db_yaku_id][month] += 1
                                yaku_counts[db_yaku_id]["total"] += 1
                                has_target_yaku = True

                        if has_target_yaku:
                            decisions = [
                                (obs, act)
                                for obs, act in state.past_decisions()
                                if obs.who() == win.who
                                and act.type() in [mjx.ActionType.DISCARD, mjx.ActionType.TSUMOGIRI]
                            ]
                            meta_counts["datas"][month] += len(decisions)
                            meta_counts["datas"]["total"] += len(decisions)
            except:
                continue

            # 表示更新（パフォーマンスのため50試合ごと）
            if i % 50 == 0:
                live.update(generate_status_view(meta_counts, yaku_counts, yaku_map, ordered_yaku_ids))

    # 4. CSV出力
    output_dir = Path("src/yaku/exp1")
    output_file = output_dir / f"yaku_distribution_{year}.csv"
    months = list(range(1, 13))

    with open(output_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Item"] + [f"{m}月" for m in months] + ["Total"])
        metrics = [("試合数", "games"), ("ラウンド数", "rounds"), ("局面数", "datas")]
        for label, key in metrics:
            writer.writerow(["", label] + [meta_counts[key][m] for m in months] + [meta_counts[key]["total"]])
        writer.writerow([])
        for yid in ordered_yaku_ids:
            writer.writerow([yid, yaku_map[yid]] + [yaku_counts[yid][m] for m in months] + [yaku_counts[yid]["total"]])

    logging.info(f"Done. Results exported to {output_file}")


if __name__ == "__main__":
    run()
