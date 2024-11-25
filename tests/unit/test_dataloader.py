import os
import pandas as pd

from torch.utils.data import DataLoader

from deepnado.data.loader import TornadoDataLoader


class TestDataLoader:

    DATA_ROOT = ""

    @classmethod
    def setup_class(cls):
        current_file = os.path.abspath(__file__)
        current_folder = os.path.dirname(current_file)
        resource_dir = os.path.join(current_folder, "resources")
        cls.DATA_ROOT = resource_dir

    @classmethod
    def teardown_class(cls):
        pass

    def test_data_loader(self):

        data_loader_obj = TornadoDataLoader()
        data_loader = data_loader_obj.get_dataloader(self.DATA_ROOT, "test", [2020], workers=1)

        assert isinstance(data_loader, DataLoader)
