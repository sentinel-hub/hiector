import math
from typing import Callable

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import Polygon

from eolearn.core import EOPatch, EOTask, FeatureType, MapFeatureTask
from eolearn.io import VectorImportTask

from ..utils.geometry import merge_bboxes


class PrepareMRRTask(EOTask):
    """This task will prepare the geopandas dataframe with minimum rotated bounding boxes from
    reference geometries (buildings).
    """

    @staticmethod
    def _calculate_mrr(g: shapely.geometry.Polygon, minsize: float) -> shapely.geometry.Polygon:
        # Calculate minimal rotated rectangle for a geometry. If the MRR is below a certain size
        # it is scaled to minsize.
        _mrr = g.minimum_rotated_rectangle
        if _mrr.area < minsize:
            ratio = math.sqrt(minsize / _mrr.area)
            _mrr = shapely.affinity.scale(_mrr, xfact=ratio, yfact=ratio, origin="centroid")
        return _mrr

    def _prepare_df(
        self, df: gpd.GeoDataFrame, erode_buffer: float = 2, minsize: float = 3 * 4 * 100
    ) -> gpd.GeoDataFrame:
        _df = df.copy()
        _df["dissolve_col"] = 1
        # Building geometries are buffered and then merged/dissolved so that blocks of buildings that are inseparable
        # by S-2 are joined into one geometry / MRR
        _df.geometry = _df.buffer(erode_buffer)
        _df = _df.dissolve(by="dissolve_col").explode(ignore_index=True)
        # The geometries are unbuffered to reflect the original size
        _df.geometry = _df.buffer(-1 * erode_buffer)
        # Remove empty geometries
        _df = _df[~_df.geometry.is_empty]
        # minimal rotated rectangle is calculated for each of these joined geometries
        _df = _df.geometry.apply(lambda g: self._calculate_mrr(g, minsize))
        return _df.reset_index(drop=True)

    def __init__(
        self,
        input_feature: FeatureType,
        output_feature: FeatureType,
        closing_buffer: int = 2,
        minsize: int = 3 * 4 * 100,
    ):
        """
        Args:
            input_feature (FeatureType): input vector feature (with reference geometries)
            output_feature (FeatureType): output vector feature (with minimum rotated rectangle geometries)
            closing_buffer (float): buffer used to join "touching" reference geometries
            minsize (float): minimum area size for the minimum rotated rectangle geometries
        """
        self.input_feature = self.parse_feature(
            input_feature, allowed_feature_types=[FeatureType.VECTOR, FeatureType.VECTOR_TIMELESS]
        )

        self.output_feature = self.parse_feature(
            output_feature, allowed_feature_types=[FeatureType.VECTOR, FeatureType.VECTOR_TIMELESS]
        )

        self.closing_buffer = closing_buffer
        self.minsize = minsize

    def execute(self, eopatch: EOPatch):
        gdf = gpd.GeoDataFrame({"geometry": [], "area": None, "merging_idx": None}, crs=None)
        if not eopatch[self.input_feature].empty:
            gdf = self._prepare_df(eopatch[self.input_feature], self.closing_buffer, self.minsize)
        eopatch[self.output_feature] = gdf
        return eopatch


class MergeBBoxesTask(EOTask):
    """This task will merge MRRs based on sorting column and "IoU"."""

    def __init__(
        self,
        input_feature: FeatureType,
        output_feature: FeatureType,
        iou_method: Callable[[Polygon, gpd.GeoSeries], pd.Series],
        iou_thr: float = 0.4,
        sorting_col: str = "area",
    ):
        """
        Args:
            input_feature (FeatureType): input vector feature (with prepared MRRs)
            output_feature (FeatureType): output vector feature (with remaining/merged MRR geometries)
            iou_method (Callable): method used to calculate "IoU"
            iou_thr (float): iou threshold (everything above iou_thr is deemed to be the same MRR and hence merged)
            sorting_col (str): column used to define "best" candidate for MRRs that need to be merged
        """
        self.input_feature = self.parse_feature(
            input_feature, allowed_feature_types=[FeatureType.VECTOR, FeatureType.VECTOR_TIMELESS]
        )

        self.output_feature = self.parse_feature(
            output_feature, allowed_feature_types=[FeatureType.VECTOR, FeatureType.VECTOR_TIMELESS]
        )

        self.iou_method = iou_method
        self.iou_thr = iou_thr
        self.sorting_col = sorting_col

    def execute(self, eopatch: EOPatch):
        bboxes = merge_bboxes(
            eopatch[self.input_feature], iou_method=self.iou_method, iou_thr=self.iou_thr, sorting_col=self.sorting_col
        )

        eopatch[self.output_feature] = bboxes
        return eopatch


class QPVectorImportTask(VectorImportTask):
    """Very similar to VectorImportTask, but allows for final transformation of data (e.g. to eopatch bbox crs)"""

    def __init__(self, feature, path, reproject=True, clip=False, config=None, **kwargs):
        super().__init__(feature=feature, path=path, reproject=reproject, clip=clip, config=config, **kwargs)

    def execute(self, eopatch=None, *, bbox=None, to_crs=None):
        """
        Args:
            eopatch (EOPatch): input EOPatch to execute task on (new EOPatch will be created else)
            bbox (BBox): A bounding box for which to load data. By default, if none is provided, it will take a
                bounding box of given EOPatch. If given EOPatch is not provided it will load the entire dataset.
            to_crs (pyproj.crs): crs to which feature should be reprojected
        Returns:
            EOPatch
        """
        eopatch = eopatch or EOPatch()
        bbox = bbox or eopatch.bbox

        data = self._load_vector_data(bbox)

        if to_crs:
            data = data.to_crs(to_crs)

        eopatch[self.feature] = data

        return eopatch


class FilterReferenceBuildingsTask(MapFeatureTask):
    """Takes only building polygons that are ok to use"""

    def map_method(self, feature):
        if not feature.empty:
            feature = feature[feature["is confirmed"]]
            feature = feature[feature["is building"]]
        return feature


class CalculateAreaProperty(MapFeatureTask):
    """Task to calculate area (column)"""

    def map_method(self, feature):
        if not feature.empty:
            feature["area"] = feature.area
        return feature
