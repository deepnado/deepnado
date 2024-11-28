import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd

from deepnado.common.constants import ALL_VARIABLES
from deepnado.data.dataset import TornadoDataset
from deepnado.plotting.plot import plot_radar


current_file = os.path.abspath(__file__)
current_folder = os.path.dirname(current_file)
resource_dir = os.path.join(current_folder, "resources")
DATA_ROOT = resource_dir


@staticmethod
def read_catalog(data_root) -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(data_root, "catalog.csv"), parse_dates=["start_time", "end_time"]
    )


def generate_plot():
    # list which variables to plot
    vars_to_plot = ALL_VARIABLES

    catalog = read_catalog(DATA_ROOT)
    file_list = [os.path.join(DATA_ROOT, f) for f in catalog.filename]

    # Grab a single sample using data loader
    dindx = 0
    n_frames = 1
    data_set = TornadoDataset(file_list, variables=ALL_VARIABLES, n_frames=n_frames)
    data = data_set[dindx]

    fig = plt.figure(figsize=(12, 6))

    plot_radar(
        data,
        fig=fig,
        channels=vars_to_plot,
        include_cbar=True,
        time_idx=-1,  # show last frame
        n_rows=2,
        n_cols=3,
    )

    # Add a caption (optional)
    fig.text(
        0.5,
        0.05,
        os.path.basename(data_set.file_list[dindx]) + " EF=" + str(data["ef_number"][0]),
        ha="center",
    )

    return fig


if __name__ == "__main__":
    matplotlib.use("QtAgg")
    figure = generate_plot()
    plt.show()
