import os
import pandas as pd
import xarray as xr

from deepnado.data.dataset import TornadoDataset


class TestDataSet:

    DATA_ROOT = ""

    @staticmethod
    def read_catalog(data_root) -> pd.DataFrame:
        return pd.read_csv(
            os.path.join(data_root, "catalog.csv"), parse_dates=["start_time", "end_time"]
        )

    @classmethod
    def setup_class(cls):
        current_file = os.path.abspath(__file__)
        current_folder = os.path.dirname(current_file)
        resource_dir = os.path.join(current_folder, "resources")
        cls.DATA_ROOT = resource_dir

    @classmethod
    def teardown_class(cls):
        pass

    def test_read_catalog(self):
        TestDataSet.read_catalog(self.DATA_ROOT)

    def test_read_file(self):
        catalog = TestDataSet.read_catalog(self.DATA_ROOT)
        file_list = [os.path.join(self.DATA_ROOT, f) for f in catalog.filename]
        # ds=xr.open_dataset(file_list[0])

        data_set = TornadoDataset(file_list)

        # random access
        data = data_set[0]
