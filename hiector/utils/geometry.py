from typing import Callable

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon


def merge_bboxes(
    df: gpd.GeoDataFrame,
    iou_method: Callable[[Polygon, gpd.GeoSeries], pd.Series],
    iou_thr: float = 0.4,
    sorting_col: str = "area",
) -> gpd.GeoDataFrame:
    """Merges bounding boxes in DF based on some threshold over score calculated with iou_method.

    Args:
        df (gpd.GeoDataFrame): DataFrame with oriented bouding boxes.
        iou_method (Callable[[Polygon, gpd.GeoSeries], pd.Series]): IOU-type function to calculate intersection between
            bounding boxes.
        iou_thr (float, optional): Threshold to determine which bounding boxes to discard.  Defaults to .4.
        sorting_col (str, optional): Property over which to sort (prioritize) bounding boxes.  Defaults to 'area'.

    Returns:
        gpd.GeoDataFrame: Filtered dataset of bounding boxes.
    """
    assert sorting_col in df.columns, f"Sorting column:  {sorting_col} is not present in the dataframe."

    # This is needed because joining on dataframes with CRS is significantly (15x) slower than if dataframe doesn't
    # have CRS, since we do a join for each bounding boxes and geopandas asserts that CRSes match during each join.
    crs = df.crs
    df.crs = None
    df["merging_idx"] = df.index.values

    remaining = df.sort_values(by=sorting_col, ascending=False)

    # It's cheaper to do spatial index in the beginning over the whole dataframe and consequently check a few more
    # candidates than it is to build a new index for each new iteration view.
    spatial_idx = df.sindex
    final_idxs = []

    while len(remaining):
        first = remaining.iloc[0]
        candidates = df.iloc[spatial_idx.query(first.geometry)]

        ious = iou_method(first.geometry, candidates.geometry)
        toremove = candidates[ious >= iou_thr].merging_idx

        remaining = remaining[~remaining.merging_idx.isin(toremove)]
        final_idxs.append(first.merging_idx)

    df.crs = crs
    return df[df.merging_idx.isin(final_idxs)]


def close_holes(poly: Polygon, thr: float) -> Polygon:
    """Close polygon holes that are lower than the threshold.

    Args:
        poly (Polygon): Input shapely polygon.
        thr (float): Hole area threshold.

    Returns:
        Polygon: Shapely polygon with holes below the threshold sized removed.
    """
    if poly.interiors:
        return Polygon(
            shell=list(poly.exterior.coords), holes=[hole for hole in poly.interiors if hole.envelope.area > thr]
        )
    return poly


def merge_predictions(predictions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Dissolve overlapping predictions and explode into connected components.

    Args:
        predictions (gpd.GeoDataFrame): Dataframe with building predictions.
    Returns:
        gpd.GeoDataFrame: GeoDataFrame with only Geometry column where each row represents one spot area.
    """
    predictions["dissolve_col"] = 1
    predictions_exploded = predictions.dissolve(by="dissolve_col").explode()
    return predictions_exploded[["geometry"]]
