import argparse
import logging

from src.data.pipeline import download_zips


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    setup_logging()

    pipeline_steps = {
        "download_zips": download_zips.run,
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
        logging.info(f"✨✨✨ Starting data collection pipeline for year: {year} ✨✨✨")

        for step_name, step_func in pipeline_steps.items():
            logging.info(f"Running step: '{step_name}' for year {year}... .")

            try:
                step_func(year)
                logging.info(f"Successfully completed step: '{step_name}' for year {year}.")

            except Exception as e:
                logging.error(
                    f"Failed at step: '{step_name}' for year {year}. Halting process for this year.", exc_info=True
                )
                break

        logging.info(f"✨✨✨ Finished data collection pipeline for year: {year} ✨✨✨")


if __name__ == "__main__":
    main()
