"""SEG-Y slice loader module."""


from pathlib import Path
from typing import Union, Tuple

from numpy.typing import NDArray
import segfast
import segyio


class SegySliceLoader:
    """Class wrapper over `segfast` loader for loading slices of data."""

    def __init__(
        self, segy_path: Union[str, Path], engine: str = 'memmap'
    ):
        """Initialize slice loader.

        Parameters
        ----------
        segy_path : Path
            Path to ".sgy" file.
        engine : str, optional
            Engine name for `segfast.Loader`.
            It can be either `"memmap"` or `"segyio"`.
            By default is `"memmap"`.
        """
        with segyio.open(segy_path, strict=True, ignore_geometry=False) as f:
            self.n_ilines, self.n_xlines = len(f.ilines), len(f.xlines)
        self.segy_file = segfast.open(segy_path, engine)
        self.n_samples = self.segy_file.n_samples
        self.n_traces = self.segy_file.n_traces
        
    def get_slices(
        self, idx: Union[int, Tuple[int, int]], axis: int
    ) -> NDArray:
        """Get one or multiple slices of data by index(es) and an axis.

        Parameters
        ----------
        idx : int
            The index or the pair of indices. If pair is given
            then get multiple slices as `all_slices[idx[0], idx[1]]`.
        axis : int
            The axis to get the slice(s) from.

        Returns
        -------
        NDArray
            2D or 3D slice array, depending on whether multiple slices
            are requested.

        Raises
        ------
        ValueError
            Raise if the axis is not in range from 0 to 2.
        ValueError
            Raise if idx has wrong signature.
        ValueError
            Raise if idx is a pair and first one is larger than the second one.
        """
        if isinstance(idx, int):
            st_idx = idx
            end_idx = idx + 1
        elif isinstance(idx, tuple) and len(idx) == 2:
            st_idx, end_idx = idx
            if st_idx > end_idx:
                raise ValueError(
                    'First index in the pair must be larger '
                    'than the second one.')
        else:
            raise ValueError(
                f'idx must be either int or tuple[int, int] but got {idx}')
        n_slices = end_idx - st_idx

        if axis == 0:
            slices = self.segy_file.load_traces(
                range(st_idx * self.n_xlines, end_idx * self.n_xlines))
            slices = slices.reshape(n_slices, self.n_xlines, self.n_samples)
            return slices.transpose(1, 2, 0)
        elif axis == 1:
            indices = []
            for i in range(st_idx, end_idx):
                indices += list(range(i, self.n_traces, self.n_xlines))
            # indices.sort()
            slices = self.segy_file.load_traces(indices)
            slices = slices.reshape(n_slices, self.n_ilines, self.n_samples)
            return slices.transpose(1, 2, 0)
            # return self.segy_file.load_traces(
            #     range(idx, self.n_traces, self.n_xlines))
        elif axis == 2:
            slices = self.segy_file.load_depth_slices(range(st_idx, end_idx))
            slices = slices.reshape(n_slices, self.n_ilines, self.n_xlines)
            return slices.transpose(1, 2, 0)
            # return (self.segy_file.load_depth_slices(range(st_idx, end_idx))
            #         .reshape(self.n_ilines, self.n_xlines))
        else:
            raise ValueError(
                f'Axis must be in the range from 0 to 2 but got {axis}.')
