"""
Utilities for working with training data
"""
from typing import List, Optional

import geopandas as gpd
import numpy as np


def filter_dataframe(
    gdf: gpd.GeoDataFrame,
    query: Optional[str] = None,
    frac: float = 1.0,
    exclude_eops: Optional[List[str]] = None,
    seed: int = 42,
):
    if query is not None:
        gdf = gdf.query(query)
    if exclude_eops is not None:
        gdf = gdf[~gdf.EOPATCH_NAME.isin(exclude_eops)]
    return gdf.sample(frac=frac, random_state=seed)


def train_test_val_split(
    gdf: gpd.GeoDataFrame, fraction_train: float = 0.6, fraction_test: float = 0.2, fraction_val: float = 0.2
):
    assert (fraction_train + fraction_val + fraction_test) == 1, "Fractions of train, test, val must sum up to 1."

    eopatches = gdf.EOPATCH_NAME.unique()
    n_eops = len(eopatches)
    n_train, n_val = int(n_eops * fraction_train), int(n_eops * fraction_val)
    n_test = n_eops - n_train - n_val

    train_eops = np.random.choice(eopatches, n_train, replace=False)
    test_eops = np.random.choice(list(set(eopatches) - set(train_eops)), n_test, replace=False)
    val_eops = np.random.choice(list(set(eopatches) - set(train_eops) - set(test_eops)), n_val, replace=False)

    assert len(train_eops) + len(test_eops) + len(val_eops) == n_eops
    assert set(train_eops).union(set(test_eops)).union(set(val_eops)) == set(eopatches)

    def split(x, train_set, test_set, val_set):
        if x in train_set:
            return "train"
        if x in test_set:
            return "test"
        if x in val_set:
            return "val"
        raise ValueError(f"Could not find a subset for eopatch: {x}")

    gdf["SUBSET"] = gdf.EOPATCH_NAME.apply(lambda x: split(x, train_eops, test_eops, val_eops))
    return gdf
