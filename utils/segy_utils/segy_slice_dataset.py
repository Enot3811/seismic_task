from pathlib import Path
from typing import Tuple, Optional, Union, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import albumentations as A
from utils.segy_utils.segy_slice_loader import SegySliceLoader

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
            self.indexes += [(i, axis) for i in range(slices_per_axis[axis])]

        self.min_value, self.max_value = values_range
        self.transforms = transforms

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx: int):
        slice = self.loader.get_slices(*self.indexes[idx])

        # Clip and normalize slice values
        slice[slice < self.min_value] = self.min_value
        slice[slice > self.max_value] = self.max_value
        slice = (slice - self.min_value) / (self.max_value - self.min_value)
        
        if self.transforms:
            slice = self.transforms(image=slice)['image']

        # Convert and add channel dim
        slice_tensor = torch.tensor(slice, dtype=torch.float32).permute(2, 0, 1)

        return slice_tensor
    
    @staticmethod
    def collate_fn(batch: List[torch.Tensor]):
        """Make all the slices to have the same size if necessary.

        Parameters
        ----------
        batch : List[torch.Tensor]
            List of slices that can have different size.
        """
        min_h = min_w = None
        for slice in batch:
            h, w = slice.shape[1:]
            if min_h is None or min_h > h:
                min_h = h
            if min_w is None or min_w > w:
                min_w = w
        # b, c, h, w
        for i in range(len(batch)):
            if batch[i].shape[1] != min_h or batch[i].shape[2] != min_w:
                batch[i] = F.interpolate(
                    batch[i][None, ...], size=(min_h, min_w), 
                    mode='bilinear', align_corners=False).squeeze(0)
        return torch.stack(batch)
