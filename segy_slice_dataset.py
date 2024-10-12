from pathlib import Path
from typing import Tuple, Optional, Union, List

import torch
from torch.utils.data import Dataset
import albumentations as A

from segy_slice_loader import SegySliceLoader


class SegySliceDataset(Dataset):
    """Dataset class for loading and cropping seg-y slices."""

    def __init__(
        self,
        segy_path: Union[str, Path],
        axes_to_use: Tuple[int] = (0, 1),
        values_range: Tuple[float, float] = (-32767.0, 32767),
        transforms: Optional[A.Compose] = None
    ):
        """Initialize the dataset.

        Parameters
        ----------
        segy_path : Path
            Path to the .sgy file.
        axes_to_use : Tuple[int], optional
            Axes along which to load slices. By default is `(0, 1)`.
        values_range : Tuple[float, float], optional
            Allowable value range for slice pixels.
            All values that are out of this range will be clipped.
            By default is `(-32767, 32767)`.
        transforms : Optional[A.Compose], optional
            Transforms to apply to the slice.
        """
        self.loader = SegySliceLoader(segy_path)
        
        slices_per_axis = {
            0: self.loader.n_ilines,
            1: self.loader.n_xlines,
            2: self.loader.n_samples
        }
        self.indexes = []
        for axis in axes_to_use:
            self.indexes += [(axis, i) for i in range(slices_per_axis[axis])]

        self.min_value, self.max_value = values_range
        self.transforms = transforms

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx: int):
        axis, idx = self.indexes[idx]
        slice = self.loader.get_slice(idx, axis)

        # Clip and normalize slice values
        slice[slice < self.min_value] = self.min_value
        slice[slice > self.max_value] = self.max_value
        slice = (slice - self.min_value) / (self.max_value - self.min_value)
        
        if self.transforms:
            slice = self.transforms(image=slice)['image']

        # Convert and add channel dim
        slice_tensor = torch.tensor(slice, dtype=torch.float32)[None, ...]

        
        return slice_tensor
    
    @staticmethod
    def collate_fn(batch: List[torch.Tensor]):
        raise NotImplementedError()


if __name__ == '__main__':
    # Check dataset
    import random
    import matplotlib.pyplot as plt
    random.seed = 42

    path = '../data/seismic/seismic.sgy'
    dset = SegySliceDataset(path, values_range=(-5800, 5800))
    print(f'Dataset size: {len(dset)=}')

    for _ in range(3):
        idx = random.randint(0, len(dset))
        sample = dset[idx]
        print(f'Sample {idx}: {sample.shape}, {sample.max()}, {sample.min()}')

        plt.imshow(sample.squeeze(), cmap='Greys_r')
        plt.show()

    transforms = A.Compose(transforms=[
        A.RandomCrop(224, 224),
        # A.Normalize(mean=(-2.0068,), std=(2434.5066,), max_pixel_value=32767)
        # A.Normalize(mean=(8.5735,), std=(2250.4207,), max_pixel_value=7021)
    ])
    dset = SegySliceDataset(path, transforms=transforms)
    for _ in range(3):
        idx = random.randint(0, len(dset))
        sample = dset[idx]
        print(f'Sample {idx}: {sample.shape}, {sample.max()}, {sample.min()}')

        plt.imshow(sample.squeeze(), cmap='Greys_r')
        plt.show()
