"""Check SegySliceLoader. Iterate along it and show slices.

Press esc to end iteration. Press enter to switch to the next axis.
"""

import sys
from pathlib import Path
import argparse
from typing import List, Tuple, Optional

import torch
from torch.utils.data import DataLoader
from cv2 import destroyAllWindows
from tqdm import tqdm
import albumentations as A

sys.path.append(str(Path(__file__).parents[3]))
from utils.data_utils.data_functions import show_images_cv2
from utils.segy_utils.segy_slice_loader import SegySliceLoader
from utils.argparse_utils import (
    non_negative_int, natural_int, required_length)
from utils.segy_utils.segy_slice_dataset import SegySliceDataset


def main(
    segy_path: Path,
    axes_to_load: List[int] = (0, 1),
    clip_values: Optional[Tuple[int, int]] = (-32767, 32767),
    crop_size: Optional[Tuple[int, int]] = None,
    mean_std: Optional[Tuple[float, float]] = None,
    batch_size: int = 1,
    random_seed: int = 42
):
    # Configure random
    torch.random.manual_seed(random_seed)

    # Prepare augmentations if needed
    if crop_size or mean_std:
        transforms = []
        if crop_size:
            transforms.append(A.RandomCrop(*crop_size))
        if mean_std:
            transforms.append(A.Normalize(*mean_std, max_pixel_value=1))
        transforms = A.Compose(transforms=transforms)
    else:
        transforms=None

    # Create dataset and data loader
    dset = SegySliceDataset(
        segy_path, axes_to_use=axes_to_load, values_range=clip_values,
        transforms=transforms)
    
    dloader = DataLoader(
        dset, batch_size, shuffle=True, collate_fn=SegySliceDataset.collate_fn)

    # Iterate over data loader and check the samples
    for batch in tqdm(dloader, 'Iterate over DataLoader'):
        for slice in batch:

            # Restore slice to original form
            if mean_std:
                raise NotImplementedError()  # TODO
            np_slice = slice.permute(1, 2, 0).numpy()

            # Show
            key = show_images_cv2(np_slice, destroy_windows=False)
            if key == 27:  # esc
                destroyAllWindows()
                return
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
                        default=(-32767, 32767),
                        help=("Min/max values to clip slices' values."))
    parser.add_argument('--crop_size', type=natural_int, nargs=2, default=None,
                        help=('Size for slice cropping. '
                              'If not given then cropping does not perform.'))
    parser.add_argument('--mean_std', type=float, nargs=2, default=None,
                        help=('Mean and std for normalization. If not given '
                              'then perform min/max normalization.'))
    parser.add_argument('--batch_size', type=natural_int, default=1,
                        help='Batch size for DataLoader.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for dataset shuffling.')
    args = parser.parse_args([
        '../data/seismic/seismic.sgy',
        '--batch_size', '16',
        '--clip_values', '-5800', '5800',
        '--crop_size', '224', '224',
        '--mean_std', 
    ])

    if args.clip_values[0] > args.clip_values[1]:
        raise ValueError()  # TODO

    return args


if __name__ == '__main__':
    args = parse_args()
    main(segy_path=args.segy_path, axes_to_load=args.axes_to_load,
         clip_values=args.clip_values, crop_size=args.crop_size,
         mean_std=args.mean_std, batch_size=args.batch_size,
         random_seed=args.random_seed)
