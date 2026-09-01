import datetime
import logging
import os
import sys


def configure_experiment_logging(
    log_directory: str,
    run_count: int,
) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(log_directory, exist_ok=True)

    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                os.path.join(
                    log_directory,
                    f"{run_count}runs_{timestamp}.log",
                )
            ),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
