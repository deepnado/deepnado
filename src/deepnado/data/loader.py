import numpy as np

from torch import from_numpy
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Dict

from deepnado.common import constants
from deepnado.data import preprocess
from deepnado.data.dataset import TornadoDataset
from deepnado.data.utils import query_catalog


class TornadoDataLoader:

    def __init__(self) -> None:
        pass

    def _numpy_to_torch(self, d: Dict[str, np.ndarray]):
        for key, val in d.items():
            d[key] = from_numpy(np.array(val))
        return d

    def get_dataloader(
        self,
        data_root: str,
        data_type: str = "train",  # or 'test'
        years: list = list(range(2013, 2023)),
        batch_size: int = 128,
        weights: Dict = None,
        include_az: bool = False,
        random_state: int = 1234,
        select_keys: list = None,
        tilt_last: bool = True,
        workers: int = 8,
    ):
        """
        Initializes torch.utils.data.DataLoader for training CNN Tornet baseline.

        data_root - location of TorNet
        data_Type - 'train' or 'test'
        years     - list of years btwn 2013 - 2022 to draw data from
        batch_size - batch size
        weights - optional sample weights, see note below
        include_az - if True, coordinates also contains az field
        random_state - random seed for shuffling files
        workers - number of workers to use for loading batches
        select_keys - Only generate a subset of keys from each tornet sample
        tilt_last - If True (default), order of dimensions is left as [batch,azimuth,range,tilt]
                    If False, order is permuted to [batch,tilt,azimuth,range]

        weights is optional, if provided must be a dict of the form
        weights={'wN':wN,'w0':w0,'w1':w1,'w2':w2,'wW':wW}
        where wN,w0,w1,w2,wW are numeric weights assigned to random,
        ef0, ef1, ef2+ and warnings samples, respectively.

        After loading TorNet samples, this does the following preprocessing:
        - Optionally permutes order of dimensions to not have tilt last
        - Takes only last time frame
        - adds 'coordinates' variable used by CoordConv layers. If include_az is True, this
        includes r, r^{-1} (and az if include_az is True)
        - Splits sample into inputs,label
        - If weights is provided, returns inputs,label,sample_weights
        """

        file_list = query_catalog(data_root, data_type, years, random_state)

        transform_list = [
            lambda d: self._numpy_to_torch(d),
            lambda d: preprocess.remove_time_dim(d),
            lambda d: preprocess.add_coordinates(d, include_az=include_az, tilt_last=tilt_last),
            lambda d: preprocess.split_x_y(d),
        ]

        if weights:
            transform_list.append(lambda xy: preprocess.compute_sample_weight(*xy, **weights))

        if select_keys is not None:
            transform_list.append(
                lambda xy: (preprocess.select_keys(xy[0], keys=select_keys),) + xy[1:]
            )

        # Dataset, with preprocessing
        transform = transforms.Compose(transform_list)

        dataset = TornadoDataset(
            file_list,
            variables=constants.ALL_VARIABLES,
            n_frames=1,
            tilt_last=tilt_last,
            transform=transform,
        )

        loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers)

        return loader
