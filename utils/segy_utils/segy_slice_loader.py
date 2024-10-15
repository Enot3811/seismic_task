"""SEG-Y slice loader module."""


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
                range(idx, self.n_traces, self.n_xlines))
        elif axis == 2:
            return self.segy_file.load_depth_slices([idx]).reshape(
                self.n_ilines, self.n_xlines)
        else:
            raise ValueError(
                f'Axis must be in the range from 0 to 2 but got {axis}.')
