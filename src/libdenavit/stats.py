import numpy as np
import numpy.typing as npt
from scipy.stats import norm


def qq_coordinates(
    data: npt.ArrayLike,
    dist=norm,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return theoretical and sample quantiles for a Q-Q plot.

    Parameters
    ----------
    data : array_like
        One-dimensional sample data.
    dist : scipy.stats distribution, optional
        The theoretical probability distribution used to compute the
        expected quantiles. The distribution must provide a ``ppf`` method.
        The default is the standard normal distribution.

    Returns
    -------
    x_qq : numpy.ndarray
        Theoretical quantiles from ``dist``.
    y_qq : numpy.ndarray
        Sorted sample data.

    Notes
    -----
    The plotting positions are calculated as

        p_i = (i + 0.5) / n

    for ``i = 0, ..., n - 1``.
    """
    data = np.asarray(data, dtype=float)

    if data.ndim != 1:
        raise ValueError("data must be one-dimensional")

    if data.size == 0:
        raise ValueError("data must contain at least one value")

    n = data.size
    probabilities = (np.arange(n) + 0.5) / n

    x_qq = dist.ppf(probabilities)
    y_qq = np.sort(data)

    return x_qq, y_qq
