import numpy as np
import torch

from typing import Dict

from deepnado.common import constants


def get_shape(d):
    """
    infers image shape from data in dict d
    """
    # use whatever variable is available
    k = list(set(d.keys()) & set(constants.ALL_VARIABLES))  # assumes this is non-empty!
    return d[k[0]].shape


def add_coordinates(d, min_range_m=2125.0, include_az=True, tilt_last=True):
    c = compute_coordinates(d, min_range_m=min_range_m, include_az=include_az, tilt_last=tilt_last)
    d["coordinates"] = c
    return d


def compute_coordinates(d, min_range_m=2125.0, include_az=True, tilt_last=True):
    """
    Add coordinate tensors r, rinv to data dict d.
    If include_az is True, also add theta.

    Coordinates are stacked along the "tilt" dimension, which is assumed
    to be the final dimension if tilt_last=True.  If tilt_last=False,
    coordinates are concatenated along axis=0.

    min_range_m is minimum possible range of radar data in meters

    """
    full_shape = get_shape(d)
    shape = full_shape[-3:-1] if tilt_last else full_shape[-2:]

    # "250" is the resolution of NEXRAD
    # "1e-5" is scaling applied for normalization
    SCALE = 1e-5  # used to scale range field for CNN
    rng_lower = (d["rng_lower"] + 250) * SCALE  # [1,]
    rng_upper = (d["rng_upper"] - 250) * SCALE  # [1,]
    min_range_m *= SCALE

    # Get az range,  convert to math convention where 0 deg is x-axis
    az_lower = d["az_lower"]
    az_lower = (90 - az_lower) * np.pi / 180  # [1,]
    az_upper = d["az_upper"]
    az_upper = (90 - az_upper) * np.pi / 180  # [1,]

    # create mesh grids
    az = torch.linspace(az_lower, az_upper, shape[0])
    rg = torch.linspace(rng_lower, rng_upper, shape[1])
    R, A = torch.meshgrid(rg, az, indexing="xy")

    # limit to minimum range of radar
    R = torch.where(R >= min_range_m, R, min_range_m)

    Rinv = 1 / R

    cat_axis = -1 if tilt_last else 0
    if include_az:
        c = torch.stack((R, A, Rinv), axis=cat_axis)
    else:
        c = torch.stack((R, Rinv), axis=cat_axis)
    return c


def split_x_y(d: Dict[str, np.ndarray]):
    """
    Splits dict into X,y, where y are tornado labels
    """
    y = d["label"]
    return d, y


def remove_time_dim(d):
    """
    Removes time dimension from data by taking last available frame
    """
    for v in d:
        d[v] = d[v][-1]
    return d


def compute_sample_weight(x, y, wN=1.0, w0=1.0, w1=1.0, w2=1.0, wW=0.5):
    """
    Assigns sample weights to samples in x,y based on
    ef_number of tornado

    category,  weight
    -----------
    random      wN
    warnings    wW
    0           w0
    1           w1
    2+          w2
    """
    weights = torch.ones_like(y, dtype=float)
    ef = x["ef_number"]
    warn = x["category"] == 2  # warnings

    weights = torch.where(ef == -1, wN, weights)  # set all nulls to wN
    weights = torch.where(warn, wW, weights)  # set warns to wW
    weights = torch.where(ef == 0, w0, weights)
    weights = torch.where(ef == 1, w1, weights)
    weights = torch.where(ef > 1, w2, weights)

    return x, y, weights
