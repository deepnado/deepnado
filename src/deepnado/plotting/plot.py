import matplotlib.pyplot as plt
import numpy as np

from typing import Dict, List, Any

from deepnado.common import constants
from deepnado.plotting.legacy import get_cmap


def get_label(key):
    return constants.VARIABLES_TO_LABELS.get(key, "")


def plot_radar(
    data: Dict[str, Any],
    channels: List[str] = ["DBZ", "VEL"],
    fig: plt.Figure = None,
    batch_idx: int = 0,
    time_idx: int = -1,
    sweep_idx: List = None,
    include_cbar: bool = False,
    include_title: bool = True,
    n_rows: int = None,
    n_cols: int = None,
):
    """
    Creates a visualization of radar data.

    data is a disctionary generated with one of the data loaders under src/data

    Each channel in data will be a tensor with shape of either [time,height,width,n_sweeps]
    """
    if fig is None:
        fig = plt.figure()
    x = data[channels[0]]  # grab sample image
    if len(x.shape) == 4:  # no batch
        batch_idx = None
    # if sweep index not provided assume lowest sweep
    if sweep_idx is None:
        sweep_idx = len(channels) * [0]
    bidx = lambda a: a[time_idx] if batch_idx is None else a[batch_idx, time_idx]  # noqa: E731

    az_min = np.float64(bidx(data["az_lower"])) * np.pi / 180
    az_max = np.float64(bidx(data["az_upper"])) * np.pi / 180
    rmin = np.float64(bidx(data["rng_lower"])) / 1e3
    rmax = np.float64(bidx(data["rng_upper"])) / 1e3

    na, nr = (x.shape[1], x.shape[2]) if batch_idx is None else (x.shape[2], x.shape[3])
    T = np.linspace(az_min, az_max, na)
    R = np.linspace(rmin, rmax, nr)
    R, T = np.meshgrid(R, T)

    for k, c in enumerate(channels):
        Z = np.float64(bidx(data[c])[..., sweep_idx[k]])  # [H,W]

        if n_rows is None:
            ax = fig.add_subplot(1, len(channels), k + 1, polar=True)
        else:
            ax = fig.add_subplot(n_rows, n_cols, k + 1, polar=True)
        ax.set_theta_zero_location("N")  # radar convention
        ax.set_theta_direction(-1)
        ax.grid(False)

        ax.set_rorigin(-rmin)
        ax.set_thetalim([az_min, az_max])
        rt = np.linspace(0, rmax - rmin, 6)
        ax.set_rgrids(rt, labels=(rt + rmin).astype(np.int64))
        ax.set_xticklabels([])  # turns off ticks
        ax.set_yticklabels([])

        if include_cbar:
            cmap, norm = get_cmap(c)
            im = ax.pcolormesh(T, R - rmin, Z, shading="nearest", cmap=cmap, norm=norm)
            fig.colorbar(im, location="right", shrink=0.5, label=get_label(c))

        if include_title:
            ax.set_title(c)
