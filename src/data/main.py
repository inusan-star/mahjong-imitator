import argparse
import logging
import warnings

from rich.logging import RichHandler
from tqdm import TqdmExperimentalWarning

from src.data.pipeline import download_zips, decompress_archives, extract_source_ids, fetch_mjlogs


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


def main():
    """Run data collection pipeline."""
    setup_logging()

    pipeline_steps = {
        "download_zips": download_zips.run,
        "decompress_archives": decompress_archives.run,
        "extract_source_ids": extract_source_ids.run,
        "fetch_mjlogs": fetch_mjlogs.run,
    }

    parser = argparse.ArgumentParser(
        description="Data collection pipeline.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        required=True,
        help="List of years to process (e.g., --years 2023 2024)",
    )
    parser.add_argument(
        "--from-step",
        type=str,
        choices=pipeline_steps.keys(),
        default=list(pipeline_steps.keys())[0],
        help="The step to start the pipeline from",
    )
    args = parser.parse_args()

    all_step_names = list(pipeline_steps.keys())
    start_step_index = all_step_names.index(args.from_step)
    steps_to_run = {name: pipeline_steps[name] for name in all_step_names[start_step_index:]}

    for year in args.years:
        logging.info("🀄🀄🀄 Starting data collection pipeline for year: %s 🀄🀄🀄", year)

        for step_name, step_func in steps_to_run.items():
            logging.info("Running step: '%s' ...", step_name)

            try:
                step_func(year)
                logging.info("Successfully completed step: '%s'.", step_name)

            except Exception as _:
                logging.error("Failed at step: '%s'. Halting process.", step_name, exc_info=True)
                break

        logging.info("🀄🀄🀄 Finished data collection pipeline for year: %s 🀄🀄🀄", year)


if __name__ == "__main__":
    main()
