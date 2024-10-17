"""Module that contains tool functions for working with SEG-Y data."""


from typing import Union, Tuple
from pathlib import Path
import multiprocessing as mp
from functools import partial

import numpy as np

from utils.segy_utils.segy_slice_loader import SegySliceLoader


def calculate_quantile(
    idx: int,
    loader: SegySliceLoader,
    axis: int,
    min_quantile: float,
    max_quantile: float
):
    """Function for mp.Pool"""
    return np.quantile(
        loader.get_slices(idx, axis), (min_quantile, max_quantile))


def calculate_segy_quantiles(
    sgy_path: Union[str, Path],
    axes_to_load: Tuple[int, ...] = (0, 1),
    min_quantile: float = 0.01,
    max_quantile: float = 0.99,
    batch_size: int = 1,
    n_processes: int = 1,
) -> Tuple[float, float]:
    """Calculate quantiles values of .sgy file.

    Using a larger batch size and more parallel processes
    can speed up execution.

    Parameters
    ----------
    sgy_path : Union[str, Path]
        Path to 
    axes_to_load : Tuple[int, ...], optional
        _description_, by default (0, 1)
    min_quantile : float, optional
        _description_, by default 0.01
    max_quantile : float, optional
        _description_, by default 0.99
    batch_size : int, optional
        _description_, by default 1
    n_processes : int, optional
        _description_, by default 1

    Returns
    -------
    Tuple[float, float]
        _description_
    """
    # TODO docstring
    loader = SegySliceLoader(sgy_path)

    for axis, axis_len in zip(axes_to_load, (loader.n_ilines,
                                     loader.n_xlines,
                                     loader.n_samples)):
        # Make indices for loader
        slices_indices = []
        for i in range(0, axis_len, batch_size):
            slices_indices.append((i, min(i + batch_size, axis_len)))

        quantile_func = partial(
            calculate_quantile, axis=axis, loader=loader,
            min_quantile=min_quantile, max_quantile=max_quantile)

        if n_processes > 1:
            with mp.Pool(processes=n_processes) as pool:
                results = pool.map(quantile_func, slices_indices)
        else:
            results = []
            for idx in slices_indices:
                results.append(quantile_func(idx))
        min_v, max_v = np.mean(np.array(results), axis=0)
        return min_v, max_v
