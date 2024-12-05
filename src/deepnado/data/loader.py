import numpy as np

from torch import from_numpy
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Dict

from deepnado.common import constants
from deepnado.data import preprocess
from deepnado.data.dataset import TornadoDataset
from deepnado.data.utils import query_catalog

def numpy_to_torch(d: Dict[str, np.ndarray]):
    for key, val in d.items():
        d[key] = from_numpy(np.array(val))
    return d

def remove_time_dim(data):
    return preprocess.remove_time_dim(data)

def add_coordinates(data, include_az, tilt_last):
    return preprocess.add_coordinates(data, include_az=include_az, tilt_last=tilt_last)

def split_x_y(data):
    return preprocess.split_x_y(data)

def compute_sample_weight_transform(xy, weights):
    return preprocess.compute_sample_weight(*xy, **weights)

def select_keys_transform(xy, select_keys):
    x_selected = preprocess.select_keys(xy[0], keys=select_keys)
    return (x_selected,) + xy[1:]


# Wrapper functions for arguments requiring closures
class TransformAddCoordinates:
    def __init__(self, include_az, tilt_last):
        self.include_az = include_az
        self.tilt_last = tilt_last

    def __call__(self, data):
        return add_coordinates(data, self.include_az, self.tilt_last)

class TransformSampleWeight:
    def __init__(self, weights):
        self.weights = weights

    def __call__(self, xy):
        return compute_sample_weight_transform(xy, self.weights)

class TransformSelectKeys:
    def __init__(self, select_keys):
        self.select_keys = select_keys

    def __call__(self, xy):
        return select_keys_transform(xy, self.select_keys)


class TornadoDataLoader:

    def __init__(self) -> None:
        pass
    
    def select_keys(self, d, select_keys):
        return preprocess.select_keys(d[0], keys=select_keys), + d[1:]

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
            numpy_to_torch,
            remove_time_dim,
            TransformAddCoordinates(include_az=include_az, tilt_last=tilt_last),
            split_x_y,
        ]

        if weights:
            transform_list.append(TransformSampleWeight(weights))

        if select_keys is not None:
            transform_list.append(TransformSelectKeys(select_keys))

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
