"""Check SegySliceLoader. Iterate along it and show slices.

Press esc to end iteration. Press enter to switch to the next axis.
"""

import sys
from pathlib import Path
import argparse
from typing import List, Tuple, Optional

from loguru import logger
from cv2 import destroyAllWindows
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[3]))
from utils.data_utils.data_functions import show_images_cv2
from utils.segy_utils.segy_slice_loader import SegySliceLoader
from utils.argparse_utils import (
    non_negative_int, natural_int, required_length)


def main(
    segy_path: Path,
    axes_to_load: List[int] = (0, 1),
    clip_values: Optional[Tuple[int, int]] = None,
    n_slices: int = 1
):
    loader = SegySliceLoader(segy_path)
    logger.info(f'{loader.n_ilines=}, {loader.n_xlines=}, {loader.n_samples=}')
    to_next_axis = False

    for axis, axis_len in zip(axes_to_load, (loader.n_ilines,
                                             loader.n_xlines,
                                             loader.n_samples)):
        logger.info(f'Iterate over axis {axis}...')

        pbar = tqdm(range(axis_len))
        for i in range(0, axis_len, n_slices):

            if n_slices == 1:
                idx = i
            else:
                idx = (i, min(i + n_slices, axis_len))
            slices = loader.get_slices(idx, axis)

            if i == 0:
                logger.info(f'Axis shape: {slices.shape[:2]}')

            normalized_slices = (slices + 32767) / (32767 * 2)  # Normalize
            
            # Titles for cv2 windows
            titles = [
                f'Axis {axis}. Press esc to exit. Press enter for next axis.']
            if clip_values:
                titles.append('Clipped slice')

            # Iterate along slices
            for j in range(slices.shape[-1]):
                pbar.update(1)

                images = [normalized_slices[..., j]]

                # Clipped slice if needed
                if clip_values:
                    clip_slice = slices[..., j].copy()
                    clip_slice[clip_slice < clip_values[0]] = clip_values[0]
                    clip_slice[clip_slice > clip_values[1]] = clip_values[1]
                    clip_slice = ((clip_slice - clip_values[0]) /
                                  (clip_values[1] - clip_values[0]))
                    images.append(clip_slice)

                key = show_images_cv2(images, titles, destroy_windows=False)
                if key == 27:  # esc
                    destroyAllWindows()
                    return
                elif key == 13:
                    destroyAllWindows()
                    to_next_axis = True
                    break
            if to_next_axis:
                to_next_axis = False
                break
        else:
            destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('segy_path', type=Path,
                        help='Path to the .sgy file to load.')
    parser.add_argument('--axes_to_load', type=non_negative_int, nargs='+',
                        default=(0, 1), action=required_length(1, 3),
                        help='Axes along which slices will be loaded.')
    parser.add_argument('--clip_values', type=float, nargs=2,
                        default=None,
                        help=('Min/max values to clip slices. '
                              'If not passed, clip is not performed'))
    parser.add_argument('--n_slices', type=natural_int, default=1,
                        help='Number of slices loaded at once.')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(segy_path=args.segy_path, axes_to_load=args.axes_to_load,
         clip_values=args.clip_values, n_slices=args.n_slices)
