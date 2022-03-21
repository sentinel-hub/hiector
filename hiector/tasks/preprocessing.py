import functools
from typing import Optional

import shapely.ops
from shapely.wkt import loads as loads_wkt
import geopandas as gpd

from eolearn.core import EOTask, MapFeatureTask
from sentinelhub import utm_to_pixel

from ..utils.preprocessing import calculate_bbox_ratio, calculate_hbb_and_obb, round_point_coords


class ReprojectReferenceTask(EOTask):
    def __init__(self, reference_feature):
        self.reference_feature = reference_feature

    def execute(self, eopatch):
        new_ref = gpd.GeoDataFrame(data=[], geometry=[], crs=eopatch.bbox.crs.pyproj_crs())
        if self.reference_feature in eopatch:
            ref = eopatch[self.reference_feature]
            target_crs = eopatch.bbox.crs.pyproj_crs()
            new_ref = ref.to_crs(target_crs)
        eopatch[self.reference_feature] = new_ref
        return eopatch


class DropDuplicatePolygonsTask(MapFeatureTask):
    """Creates bounding boxes around each polygon"""

    def map_method(self, dataframe, *, column):
        return dataframe.drop_duplicates(subset=column)


class RemoveInvalidGeometryTask(MapFeatureTask):
    """There are some LineString geometries, we remove them here"""

    def map_method(self, dataframe):
        dataframe["geometry"] = dataframe["geometry"].buffer(0)
        return dataframe[dataframe.is_valid]


class CreatePolygonBBoxesTask(MapFeatureTask):
    """Creates bounding boxes around each polygon"""

    def map_method(self, dataframe):
        return calculate_hbb_and_obb(dataframe)


class FilterEmptyGeometriesTask(MapFeatureTask):
    """Filters geometries with 0 area"""

    def map_method(self, dataframe, *, column):
        return dataframe[~dataframe[column].is_empty]


class TransformToPixelsCoordTask(EOTask):
    """Transform bounding boxes into pixel coordinates"""

    def __init__(
        self,
        input_feature,
        output_feature,
        bbox_column: str,
        resolution: float,
        round_decimals: Optional[int] = None,
    ):
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.bbox_column = bbox_column
        self.resolution = resolution
        self.round_decimals = round_decimals

    def execute(self, eopatch):
        transform = eopatch.bbox.get_transform_vector(self.resolution, self.resolution)
        dataframe = eopatch[self.input_feature]

        def utm_to_pixel_transformer(east, north):
            row, column = utm_to_pixel(east, north, transform=transform, truncate=False)
            return column, row

        bboxes = dataframe[self.bbox_column]

        new_bbox_column_name = "pixel_bbox"
        dataframe[new_bbox_column_name] = bboxes.apply(
            lambda bbox_polygon: shapely.ops.transform(utm_to_pixel_transformer, loads_wkt(bbox_polygon)).wkt
        )

        if self.round_decimals is not None:
            rounder = functools.partial(round_point_coords, decimals=self.round_decimals)
            dataframe[new_bbox_column_name] = dataframe[new_bbox_column_name].apply(
                lambda bbox_polygon: shapely.ops.transform(rounder, loads_wkt(bbox_polygon)).wkt
            )

        eopatch[self.output_feature] = dataframe
        return eopatch


class CalculateBBoxRatioTask(MapFeatureTask):
    """For each bounding box it calculates ratios between its larger and smaller sides"""

    def map_method(self, dataframe, bbox_column):
        bboxes = dataframe[bbox_column]

        ratio_column_name = f"{bbox_column}_ratio"
        dataframe[ratio_column_name] = bboxes.apply(calculate_bbox_ratio)

        return dataframe
