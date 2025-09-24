import argparse
import logging

from src.data.pipeline import download_zip, decompress_archives, extract_urls


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
        "download_zip": download_zip.run,
        "decompress_archives": decompress_archives.run,
        "extract_urls": extract_urls.run,
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
        logging.info("✨✨✨ Starting data collection pipeline for year: %s ✨✨✨", year)

        for step_name, step_func in steps_to_run.items():
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
