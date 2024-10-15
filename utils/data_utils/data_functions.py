"""Module that contain tool functions for working with images."""


from typing import List, Union, Optional, Tuple
import matplotlib.axes
import numpy as np
from numpy.typing import NDArray
import cv2
import matplotlib
import matplotlib.pyplot as plt


def show_images_cv2(
    images: Union[NDArray, List[NDArray]],
    window_title: Union[str, List[str]] = 'image',
    destroy_windows: bool = True,
    delay: int = 0,
    rgb_to_bgr: bool = True
) -> int:
    """Display one or a few images by cv2.

    Press any key to return from function. Key's code will be returned.
    If `destroy_windows` is `True` then windows will be closed.

    Parameters
    ----------
    images : Union[NDArray, Iterable[NDArray]]
        Image array or list of image arrays.
    window_title : Union[str, Iterable[str]], optional
        Image window's title. If List is provided it must have the same length
        as the list of images. By default is `'image'`.
    destroy_windows : bool, optional
        Whether to close windows after function's end. By default is `True`.
    delay : int, optional
        Time in ms to wait before window closing. If `0` is passed then window
        won't be closed before any key is pressed. By default is `0`.
    rgb_to_bgr : bool, optional
        Whether to convert input images from RGB to BGR before showing.
        By default is `True`. If your images are already in BGR then sign it
        as `False`. Ignore if image has only one channel.

    Returns
    -------
    int
        Pressed key code.
    """
    key_code = -1
    if isinstance(images, (List, tuple)):
        if isinstance(window_title, str):
            one_title = True
        elif (isinstance(window_title, list) and
                len(window_title) == len(images)):
            one_title = False
        else:
            raise TypeError(
                '"window_title" must be str or List[str] with the same '
                'length as the list of images.')
        for i, image in enumerate(images):
            if one_title:
                title = f'{window_title}_{i}'
            else:
                title = window_title[i]
            if rgb_to_bgr and image.shape[-1] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow(title, image)
    elif isinstance(images, np.ndarray):
        if rgb_to_bgr and images.shape[-1] == 3:
            images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_title, images)
    else:
        raise TypeError('"images" must be NDArray or List of NDArrays, '
                        f'but got {type(images)}')
    key_code = cv2.waitKey(delay)
    if destroy_windows:
        cv2.destroyAllWindows()
    return key_code


def show_image_plt(
    img: NDArray,
    ax: Optional[matplotlib.axes.Axes] = None,
    figsize: Tuple[int, int] = (16, 8),
    plt_show: bool = False
) -> matplotlib.axes.Axes:
    """Display an image on a matplotlib figure.

    Parameters
    ----------
    img : NDArray
        An image to display with shape `(h, w, c)` in RGB.
    ax : Optional[matplotlib.axes.Axes], optional
        Axes for image showing. If not given then a new Figure and Axes
        will be created.
    figsize : Tuple[int, int], optional
        Figsize for pyplot figure. By default is `(16, 8)`.
    plt_show : bool, optional
        Whether to make `plt.show()` in this function's calling.
        By default is `False`.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with showed image.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    if plt_show:
        plt.show()
    return ax
