import numpy as np
import pandas as pd
import sys
from ..sarracen_dataframe import SarracenDataFrame


def _get_mass(data: 'SarracenDataFrame'):
    if data.mcol is None:
        if 'mass' not in data.params:
            raise KeyError("'mass' column does not exist in this "
                           "SarracenDataFrame.")
        return data.params['mass']

    return data[data.mcol]


def _get_origin(origin: list) -> list:
    if origin is None:
        return [0.0, 0.0, 0.0]
    else:
        return origin


def _bin_particles_by_radius(data: 'SarracenDataFrame',
                             r_in: float = None,
                             r_out: float = None,
                             bins: int = 300,
                             log: bool = False,
                             geometry: str = 'cylindrical',
                             origin: list = None):
    """
    Utility function to bin particles in discrete intervals by radius.

    Parameters
    ----------
    data: SarracenDataFrame
        The particle dataset.
    r_in : float, optional
        Inner radius of the disc. Defaults to the minimum r value.
    r_out : float, optional
        Outer radius of the disc. Defaults to the maximum r value.
    bins : int, optional
        Defines the number of equal-width bins in the range [r_in, r_out].
        Default is 300.
    log : bool, optional
        Whether to bin in log scale or not. Defaults to False.
    geometry : str, optional
        Coordinate system to use to calculate the particle radii. Can be
        either *spherical* or *cylindrical*. Defaults to *cylindrical*.
    origin : array-like, optional
        The x, y and z centre point around which to compute radii. Defaults to
        [0, 0, 0].

    Returns
    -------
    rbins: Series
        The radial bin to which each particle belongs.
    bin_edges: ndarray
        Locations of the bin edges.
    """

    if geometry == 'spherical':
        r = np.sqrt((data[data.xcol] - origin[0]) ** 2
                    + (data[data.ycol] - origin[1]) ** 2
                    + (data[data.zcol] - origin[2]) ** 2)
    elif geometry == 'cylindrical':
        r = np.sqrt((data[data.xcol] - origin[0]) ** 2
                    + (data[data.ycol] - origin[1]) ** 2)
    else:
        raise ValueError("geometry should be either 'cylindrical' or "
                         "'spherical'")

    # should we add epsilon here?
    if r_in is None:
        r_in = r.min() - sys.float_info.epsilon
    if r_out is None:
        r_out = r.max() + sys.float_info.epsilon

    if log:
        bin_edges = np.logspace(np.log10(r_in), np.log10(r_out), bins+1)
    else:
        bin_edges = np.linspace(r_in, r_out, bins+1)
    rbins = pd.cut(r, bin_edges)

    return rbins, bin_edges

def _bin_particles_2D(data: 'SarracenDataFrame',
                      r_in: float = None,
                      r_out: float = None,
                      th_max: float = None,
                      rbins: int = 300,
                      thbins: int = 300,
                      log: bool = False,
                      origin: list = None):
    """
    Utility function to bin particles in discrete intervals by radius
    and polar angle in spherical coordinates.

    Parameters
    ----------
    data: SarracenDataFrame
        The particle dataset.
    r_in : float, optional
        Inner radius of the disc. Defaults to the minimum r value.
    r_out : float, optional
        Outer radius of the disc. Defaults to the maximum r value.
    th_max: float, optional
        Maximum polar angle with respect to the midplane. Defaults to pi/2.
    rbins : int, optional
        Defines the number of equal-width bins in the range [r_in, r_out].
        Default is 300.
    thbins : int, optional
        Defines the number of equal-width bins in the range [pi/2 - zmax, pi/2 + zmax].
        Default is 300.
    log : bool, optional
        Whether to bin in log scale or not. Defaults to False.
    origin : array-like, optional
        The x, y and z centre point around which to compute radii. Defaults to
        [0, 0, 0].

    Returns
    -------
    rbins: Series
        The radial bin to which each particle belongs.
    thbins: Series
        The polar bin to which each particle belongs.
    r_bin_edges: ndarray
        Locations of the radial bin edges.
    th_bin_edges: ndarray
        Locations of the polar bin edges.
    """

    r = np.sqrt((data[data.xcol] - origin[0]) ** 2
                + (data[data.ycol] - origin[1]) ** 2
                + (data[data.zcol] - origin[2]) ** 2)
    
    th = np.arccos(data[data.zcol]/r)

    if r_in is None:
        r_in = r.min() - sys.float_info.epsilon
    if r_out is None:
        r_out = r.max() + sys.float_info.epsilon

    if th_max is None:
        th_max = np.pi/2 - sys.float_info.epsilon
    elif (th_max<=0) or (th_max>np.pi/2):
        raise ValueError("th_max should be 0 < th_max <= pi/2")

    if log:
        r_bin_edges = np.logspace(np.log10(r_in), np.log10(r_out), rbins+1)
    else:
        r_bin_edges = np.linspace(r_in, r_out, rbins+1)
    
    th_bin_edges = np.linspace(np.pi/2-th_max, np.pi/2 + th_max, thbins + 1)
    rbins = pd.cut(r, r_bin_edges)
    thbins = pd.cut(th, th_bin_edges)

    return rbins, thbins, r_bin_edges, th_bin_edges

def _get_bin_midpoints(bin_edges: np.ndarray,
                       log: bool = False) -> np.ndarray:
    """
    Calculate the midpoint of bins given their edges.

    Parameters
    ----------
    bin_edges: ndarray
        Locations of the bin edges.
    log : bool, optional
        Whether to bin in log scale or not. Defaults to False.
    """

    if log:
        return np.sqrt(bin_edges[:-1] * bin_edges[1:])
    else:
        return 0.5 * (bin_edges[1:] - bin_edges[:-1]) + bin_edges[:-1]
