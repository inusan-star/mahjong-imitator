import argparse
import logging

from src.data.pipeline import download_zips
from src.data.pipeline import decompress_archives


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    """Run data collection pipeline."""
    setup_logging()

    pipeline_steps = {
        "download_zips": download_zips.run,
        "decompress_archives": decompress_archives.run,
    }

    parser = argparse.ArgumentParser(description="Data collection pipeline.")
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        required=True,
        help="List of years to process (e.g., --years 2023 2024)",
    )
    args = parser.parse_args()

    for year in args.years:
        logging.info("✨✨✨ Starting data collection pipeline for year: %s ✨✨✨", year)

        for step_name, step_func in pipeline_steps.items():
            logging.info("Running step: '%s' for year %s...", step_name, year)

            try:
                step_func(year)
                logging.info("Successfully completed step: '%s' for year %s.", step_name, year)

            except Exception as _:
                logging.error(
                    "Failed at step: '%s' for year %s. Halting process for this year.", step_name, year, exc_info=True
                )
                break

        logging.info("✨✨✨ Finished data collection pipeline for year: %s ✨✨✨", year)


if __name__ == "__main__":
    main()
