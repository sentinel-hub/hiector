"""
Module with utilities for preprocessing data that will be used for training
"""
from typing import Tuple

from geopandas import GeoDataFrame
from shapely.geometry import LineString, Polygon


def calculate_hbb_and_obb(dataframe: GeoDataFrame, as_wkt: bool = True) -> GeoDataFrame:
    """Calculates horizontal and oriented bounding boxes for geometries in dataframe.

    :param dataframe: A dataframe with geometries
    :param as_wkt: If True geometries will be added as WKT strings, otherwise they will be added as shapely objects.
    :return: A dataframe with two added columns, 'hbb' and 'obb' for horizontal and oriented bounding boxes.
    """
    dataframe["hbb"] = dataframe.geometry.envelope
    dataframe["obb"] = dataframe.geometry.apply(lambda x: x.minimum_rotated_rectangle)

    if as_wkt:
        dataframe["hbb"] = dataframe["hbb"].apply(lambda x: x.wkt)
        dataframe["obb"] = dataframe["obb"].apply(lambda x: x.wkt)

    return dataframe


def round_point_coords(x: float, y: float, decimals: int) -> Tuple[float, float]:
    """Rounds coordinates of a point"""
    return round(x, decimals), round(y, decimals)


def calculate_bbox_ratio(bbox_polygon: Polygon) -> float:
    """Calculate a ratio between larger and smaller sides of a bounding box polygon"""
    size1 = LineString(list(bbox_polygon.exterior.coords)[:2]).length
    size2 = LineString(list(bbox_polygon.exterior.coords)[1:3]).length

    small_size = min(size1, size2)
    large_size = max(size1, size2)

    area = bbox_polygon.area
    relative_error = abs(small_size * large_size - area) / area
    if relative_error > 1e-6:
        raise ValueError(f"It seems that this polygon is not a bounding box: {bbox_polygon.wkt}")

    ratio = large_size / small_size
    return ratio
