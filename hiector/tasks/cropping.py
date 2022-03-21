"""
Utilities for cropping data
"""
import functools
from collections import defaultdict

import geopandas as gpd
import numpy as np
import shapely.ops
from shapely.affinity import translate
from shapely.wkt import loads as load_wkt

from eolearn.core import EOTask
from sentinelhub import parse_time, pixel_to_utm

from ..utils.preprocessing import round_point_coords


class CroppingTask(EOTask):
    def __init__(
        self,
        raster_feature,
        data_mask_feature,
        vector_feature,
        grid_feature,
        intersection_feature,
        data_stack_feature,
        no_data_value: float,
        size: int,
        overlap: float,
        resolution: float,
        cloud_mask_feature=None,
        valid_reference_mask_feature=None,
        valid_threshold: float = 0.0,
        iou_threshold: float = 0.5,
        take_closest_time_frame=None,
    ):
        """
        :param raster_feature: A data feature to crop
        :param data_mask_feature: A mask feature showing areas with no valid data
        :param vector_feature: A vector feature to crop
        :param grid_feature: A vector feature where cropped grid is saved at.
        :param intersection_feature: A vector feature where a spatial join between grid and reference polygons is
            stored.
        :param data_stack_feature: A data feature where output stack of data is stored.
        :param no_data_value: Value that will be set to all pixels that are masked.
        :param size: A size of of images to crop out of input image.
        :param overlap: Overlap between sub-images extracted.
        :param resolution: Resolution on which task is running.
        :param cloud_mask_feature: An input cloud mask feature. If not provided it will be ignored.
        :param valid_reference_mask_feature: An input mask feature defining where reference labels are available. If
            not provided it will be ignored.
        :param valid_threshold: Discard sub-images that have a fraction of valid data lower than threshold.
        :param iou_threshold: A minimal percentage of area of a building reference polygon that has to intersect
            with a grid polygon so that it will still be used for training for that grid polygon.
        :param take_closest_time_frame: Take a time frame that is closest to the given timestamp
        """
        self.raster_feature = raster_feature
        self.data_mask_feature = data_mask_feature
        self.cloud_mask_feature = cloud_mask_feature
        self.valid_reference_mask_feature = valid_reference_mask_feature
        self.vector_feature = vector_feature
        self.grid_feature = grid_feature
        self.intersection_feature = intersection_feature
        self.data_stack_feature = data_stack_feature
        self.no_data_value = no_data_value
        self.size = size
        self.overlap = overlap
        self.resolution = resolution
        self.valid_threshold = valid_threshold
        self.iou_threshold = iou_threshold
        self.take_closest_time_frame = (
            parse_time(take_closest_time_frame, force_datetime=True, ignoretz=True) if take_closest_time_frame else None
        )

    def _choose_time_index(self, timestamps):
        """Chooses an index of a timestamp that is closest to specified timestamp"""
        if not self.take_closest_time_frame:
            return 0

        if not timestamps:
            raise ValueError(
                "EOPatch has no timestamps, hence we cannot choose a timestamp closest to "
                f"{self.take_closest_time_frame}"
            )

        timestamps = [timestamp.replace(tzinfo=None) for timestamp in timestamps]
        time_differences = [abs((timestamp - self.take_closest_time_frame).total_seconds()) for timestamp in timestamps]
        return np.argmin(time_differences)

    def _apply_no_data_value(self, data, data_mask, cloud_mask, valid_reference_mask):
        """Sets no_data_value to all invalid data pixels"""
        if cloud_mask is not None:
            data_mask = data_mask & ~cloud_mask

        if valid_reference_mask is not None:
            data_mask = data_mask & valid_reference_mask

        data_mask = data_mask.squeeze(axis=-1)
        data[~data_mask.astype(bool), :] = self.no_data_value
        return data

    def _crop_data(self, data, data_mask, cloud_mask, valid_reference_mask):
        height, width, bands = data.shape
        stride = int(self.size * (1 - self.overlap))

        cropped_data = []
        stats = defaultdict(list)
        for x in range(0, width, stride):
            for y in range(0, height, stride):
                x2, y2 = min(x + self.size, width), min(y + self.size, height)
                x1, y1 = max(0, x2 - self.size), max(0, y2 - self.size)

                if x1 == x2 or y1 == y2:
                    continue

                data_slice = data[y1:y2, x1:x2, ...]
                data_mask_slice = data_mask[y1:y2, x1:x2, ...]
                valid_mask_slice = data_mask_slice
                if cloud_mask is not None:
                    cloud_mask_slice = cloud_mask[y1:y2, x1:x2, ...]
                    valid_mask_slice = valid_mask_slice & ~cloud_mask_slice

                if valid_reference_mask is not None:
                    valid_reference_mask_slice = valid_reference_mask[y1:y2, x1:x2, ...]
                    valid_mask_slice = valid_mask_slice & valid_reference_mask_slice

                valid_fraction = np.mean(valid_mask_slice)
                if valid_fraction < self.valid_threshold:
                    continue

                cropped_data.append(data_slice)

                stats["IS_DATA_RATIO"].append(np.mean(data_mask_slice))
                stats["VALID_DATA_RATIO"].append(np.mean(valid_mask_slice))
                if cloud_mask is not None:
                    stats["CLOUD_COVERAGE"].append(1 - np.mean(cloud_mask_slice))
                if valid_reference_mask is not None:
                    stats["HAS_REF_RATIO"].append(np.mean(valid_reference_mask_slice))

                polygon = shapely.geometry.box(x1, y1, x2, y2)
                stats["pixel_geometry"].append(polygon)

        cropped_data = (
            np.stack(cropped_data, axis=0)
            if cropped_data
            else np.zeros((0, self.size, self.size, bands), dtype=data.dtype)
        )
        return cropped_data, stats

    def _threshold_by_iou(self, row):
        """Intersects grid bbox with reference bbox and thresholds by intersection over union"""

        if isinstance(row.pixel_bbox, str):
            pixel_bbox = load_wkt(row.pixel_bbox)
        else:
            pixel_bbox = row.pixel_bbox

        pixel_geometry = row.pixel_geometry
        if pixel_bbox.area == 0:
            return False
        intersection_geo = pixel_geometry.intersection(pixel_bbox)
        iou = intersection_geo.area / pixel_bbox.area
        return iou > self.iou_threshold

    def execute(self, eopatch, *, eopatch_name):
        time_index = self._choose_time_index(eopatch.timestamp)
        data = eopatch[self.raster_feature][time_index]
        data_mask = eopatch[self.data_mask_feature][time_index]
        cloud_mask = eopatch[self.cloud_mask_feature][time_index].astype(bool) if self.cloud_mask_feature else None
        reference_mask = (
            eopatch[self.valid_reference_mask_feature].astype(bool) if self.valid_reference_mask_feature else None
        )

        reference_gdf = (
            eopatch[self.vector_feature]
            if self.vector_feature in eopatch
            else gpd.GeoDataFrame(geometry=[], crs=eopatch.bbox.crs.pyproj_crs())
        )

        data = self._apply_no_data_value(data, data_mask, cloud_mask, reference_mask)

        transform = eopatch.bbox.get_transform_vector(self.resolution, self.resolution)

        def pixel_to_utm_transformer(column, row):
            return pixel_to_utm(row, column, transform=transform)

        cropped_data, stats = self._crop_data(data, data_mask, cloud_mask, reference_mask)
        utm_polygons = [shapely.ops.transform(pixel_to_utm_transformer, polygon) for polygon in stats["pixel_geometry"]]

        crop_grid_gdf = gpd.GeoDataFrame(
            stats, geometry=utm_polygons, crs=eopatch.bbox.crs.pyproj_crs()
        ).drop_duplicates(subset="geometry", keep="first")
        # We drop the duplicate geometries, since the _crop_data function can return duplicates due to the clipping.

        cropped_data = cropped_data[crop_grid_gdf.index.values, ...]
        assert len(cropped_data) == len(
            crop_grid_gdf
        ), "Number of sampled images doesn't match number of bounding boxes"

        crop_grid_gdf["EOPATCH_NAME"] = eopatch_name
        crop_grid_gdf["NAME"] = crop_grid_gdf.pixel_geometry.apply(
            lambda geo: f"{eopatch_name}-{int(geo.bounds[0])}-{int(geo.bounds[1])}-{self.size}"
        )
        crop_grid_gdf["XB"] = crop_grid_gdf.pixel_geometry.apply(lambda geo: int(geo.bounds[0]))
        crop_grid_gdf["YB"] = crop_grid_gdf.pixel_geometry.apply(lambda geo: int(geo.bounds[1]))
        if reference_gdf.crs != crop_grid_gdf.crs:
            reference_gdf.to_crs(crop_grid_gdf.crs, inplace=True)
        joined_gdf = gpd.sjoin(crop_grid_gdf, reference_gdf)
        is_large_enough_iou = joined_gdf.apply(self._threshold_by_iou, axis=1)
        joined_gdf = joined_gdf[is_large_enough_iou]

        if not joined_gdf.empty:
            joined_gdf["pixel_bbox"] = joined_gdf[["pixel_bbox", "XB", "YB"]].apply(
                lambda row: translate(load_wkt(row.pixel_bbox), xoff=-row.XB, yoff=-row.YB), axis=1
            )
            # TODO: harmonize this with rounding in TransformToPixelsCoordTask
            rounder = functools.partial(round_point_coords, decimals=1)
            joined_gdf["pixel_bbox"] = joined_gdf["pixel_bbox"].apply(
                lambda bbox_polygon: shapely.ops.transform(rounder, bbox_polygon).wkt
            )

        counts = {name: count for name, count in joined_gdf.groupby("NAME").size().iteritems()}

        crop_grid_gdf["N_BBOXES"] = crop_grid_gdf.NAME.apply(lambda name: counts.get(name, 0))
        crop_grid_gdf["pixel_geometry"] = crop_grid_gdf.pixel_geometry.apply(lambda x: x.wkt)

        eopatch[self.data_stack_feature] = cropped_data
        eopatch[self.grid_feature] = crop_grid_gdf
        eopatch[self.intersection_feature] = joined_gdf
        return eopatch
