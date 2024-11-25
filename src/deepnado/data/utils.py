import os
import pandas as pd


@staticmethod
def query_catalog(
    data_root: str,
    data_type: str,
    years: list[int],
    random_state: int,
    catalog: pd.DataFrame = None,
) -> list[str]:
    """Obtain file names that match criteria.
    If catalog is not provided, this loads and parses the
    default catalog.

    Inputs:
    data_root: location of data
    data_type: train or test
    years: list of years between 2013 - 2022 to draw data from
    random_state: random seed for shuffling files
    catalog:  Preloaded catalog, optional
    """
    if catalog is None:
        catalog_path = os.path.join(data_root, "catalog.csv")
        if not os.path.exists(catalog_path):
            raise RuntimeError("Unable to find catalog.csv at " + data_root)
        catalog = pd.read_csv(catalog_path, parse_dates=["start_time", "end_time"])

    catalog = catalog[catalog["type"] == data_type]
    catalog = catalog[catalog.start_time.dt.year.isin(years)]
    catalog = catalog.sample(frac=1, random_state=random_state)  # shuffle file list

    file_list = [os.path.join(data_root, f) for f in catalog.filename]

    return file_list
