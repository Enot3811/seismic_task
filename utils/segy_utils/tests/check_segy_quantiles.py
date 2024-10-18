"""Check performance of quantiles calculating with different parameters."""


import time
from pathlib import Path
import argparse
from loguru import logger
import sys

sys.path.append(str(Path(__file__).parents[3]))
from utils.segy_utils.segy_functions import calculate_segy_quantiles


def main(segy_path: Path):
    start = time.time()
    logger.info(
        calculate_segy_quantiles(segy_path, batch_size=32, n_processes=8))
    logger.info(time.time() - start)

    start = time.time()
    logger.info(
        calculate_segy_quantiles(segy_path, batch_size=16, n_processes=8))
    logger.info(time.time() - start)

    start = time.time()
    logger.info(
        calculate_segy_quantiles(segy_path, batch_size=8, n_processes=8))
    logger.info(time.time() - start)

    start = time.time()
    logger.info(
        calculate_segy_quantiles(segy_path, batch_size=1, n_processes=8))
    logger.info(time.time() - start)

    start = time.time()
    logger.info(
        calculate_segy_quantiles(segy_path, batch_size=32, n_processes=1))
    logger.info(time.time() - start)

    start = time.time()
    logger.info(
        calculate_segy_quantiles(segy_path, batch_size=1, n_processes=1))
    logger.info(time.time() - start)



def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('segy_path', type=Path,
                        help='Path to the .sgy file to load.')
    args = parser.parse_args(['../data/seismic/seismic.sgy'])

    return args


if __name__ == '__main__':
    args = parse_args()
    main(segy_path=args.segy_path)
