import logging
import math
import re
import warnings

import pandas as pd
from rich.logging import RichHandler
from sqlalchemy import Engine, inspect, text
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

import src.config as global_config
from src.db.session import engine


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


def _export_table(target_engine: Engine, table_name: str) -> None:
    """Export a table to Parquet format."""
    with target_engine.connect() as connect:
        result = connect.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        row_count = result.scalar() or 0

    output_dir = global_config.DUMPS_DIR / table_name
    output_dir.mkdir(parents=True, exist_ok=True)

    total_chunks = math.ceil(row_count / global_config.DB_CHUNK_SIZE)

    with target_engine.connect().execution_options(stream_results=True) as connect:
        chunks = pd.read_sql(text(f"SELECT * FROM {table_name}"), connect, chunksize=global_config.DB_CHUNK_SIZE)

        for index, chunk in enumerate(
            tqdm(chunks, total=total_chunks, desc=f"Exporting '{table_name}'", unit="chunk", leave=False)
        ):
            chunk_path = output_dir / f"part_{index:03d}.parquet"
            chunk.to_parquet(chunk_path, index=False, engine="pyarrow")


def main() -> None:
    """Database to Parquet export execution."""
    setup_logging()

    logging.info("🀄🀄🀄 Starting database export to Parquet 🀄🀄🀄")

    global_config.DUMPS_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Finding tables in the database ...")

    all_tables = inspect(engine).get_table_names()
    exclude_pattern = re.compile(r"^(flyway|__).*", re.IGNORECASE)
    tables = [table for table in all_tables if not exclude_pattern.match(table)]

    logging.info("Successfully found tables.")
    logging.info("Exporting tables to Parquet in '%s' ...", global_config.DUMPS_DIR)

    for table_name in tqdm(tables, desc="Exporting", unit="table"):
        try:
            logging.info("Running export table: '%s' ...", table_name)

            _export_table(engine, table_name)

            logging.info("Successfully exported table")

        except Exception:
            logging.error("Failed to export table. Skipping. Table: '%s'", table_name)
            continue

    logging.info("🀄🀄🀄 Finished database export 🀄🀄🀄")


if __name__ == "__main__":
    main()
