import functools
import json

import numpy as np
from shapely.wkt import loads

from eolearn.core import EOTask, get_filesystem
from sentinelhub import SHConfig


class ExportRasterDataTask(EOTask):
    def __init__(self, raster_feature, grid_feature, path, config=None):
        self.raster_feature = raster_feature
        self.grid_feature = grid_feature
        self.path = path
        self.config = config or SHConfig()

    def execute(self, eopatch):
        data = eopatch[self.raster_feature]
        grid = eopatch[self.grid_feature]

        filesystem = get_filesystem(self.path, config=self.config)
        for sample, name in zip(data, grid.NAME):
            with filesystem.openbin(f"{name}.npy", "w") as file_handle:
                np.save(file_handle, sample)

        return eopatch


class ExportGeometriesAndLabelsTask(EOTask):
    def __init__(self, reference_feature, path, config=None):
        self.reference_feature = reference_feature
        self.path = path
        self.config = config or SHConfig()

    @staticmethod
    def _export_to_json(tile_df, filesystem):
        payload = [
            {
                "label": "building",
                "geometry": list(loads(polygon).exterior.coords)[:-1]
                if isinstance(polygon, str)
                else list(polygon.exterior.coords)[:-1],
            }
            for polygon in tile_df.pixel_bbox.values
        ]

        filename = f"{tile_df.NAME.values[0]}.json"
        with filesystem.open(filename, "w") as file_handle:
            json.dump(payload, file_handle, indent=2)

    def execute(self, eopatch):
        reference_data = eopatch[self.reference_feature]
        reference_data = reference_data[["pixel_bbox", "NAME"]]

        filesystem = get_filesystem(self.path, config=self.config)
        export_function = functools.partial(self._export_to_json, filesystem=filesystem)
        reference_data.groupby("NAME").apply(export_function)

        return eopatch
