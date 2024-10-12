"""Seg-y slice loader module."""


from pathlib import Path
from typing import Union

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
        
    def get_slice(self, idx: int, axis: int) -> NDArray:
        """Get a slice of data by an index and an axis.

        Parameters
        ----------
        idx : int
            The index of slice along the axis.
        axis : int
            The axis to get the slice from.

        Returns
        -------
        NDArray
            2D slice array.

        Raises
        ------
        ValueError
            Raise if the axis is not in range from 0 to 2.
        """
        if axis == 0:
            return self.segy_file.load_traces(
                range(idx * self.n_xlines, (idx + 1) * self.n_xlines))
        elif axis == 1:
            return self.segy_file.load_traces(
                range(idx, self.n_traces, self.n_ilines))
        elif axis == 2:
            return self.segy_file.load_depth_slices([idx]).reshape(
                self.n_ilines, self.n_xlines)
        else:
            raise ValueError(
                f'Axis must be in the range from 0 to 2 but got {axis}.')
        

def calculate_quantile_values(loader: SegySliceLoader):
    for axis, length in enumerate([loader.n_ilines,
                                   loader.n_xlines,
                                   loader.n_samples]):
        maxes = []
        mins = []
        for i in range(length):
            slice = loader.get_slice(i, axis)
            vmin, vmax = np.quantile(slice, (0.01, 0.99))
            maxes.append(vmax)
            mins.append(vmin)
        vmin = np.mean(vmin)
        vmax = np.mean(vmax)
        print(f'{axis=}: {vmin=}, {vmax=}')
        

if __name__ == '__main__':
    # Check slice loader
    import matplotlib.pyplot as plt
    import numpy as np

    path = '../data/seismic/seismic.sgy'
    loader = SegySliceLoader(path)
    print(loader.n_ilines, loader.n_xlines, loader.n_samples)

    slice = loader.get_slice(50, 2)
    print(slice.shape)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    img = ax1.imshow(slice.T, cmap='Greys_r', vmin=slice.min(), vmax=slice.max())
    vmin, vmax = np.quantile(slice, (0.01, 0.99))
    img = ax2.imshow(slice.T, cmap='Greys_r', vmin=vmin, vmax=vmax)
    plt.show()

    calculate_quantile_values(loader)
