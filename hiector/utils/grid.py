from functools import partial
from typing import Optional, Tuple, Union, Callable

import geopandas as gpd
import pandas as pd
import pyproj
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import transform

from eolearn.core import (
    EOPatch,
    EOWorkflow,
    ExtractBandsTask,
    FeatureType,
    LoadTask,
    OverwritePermission,
    RemoveFeatureTask,
    SaveTask,
    get_filesystem,
    linearly_connect_tasks,
)
from eolearn.core.utils.fs import join_path
from sentinelhub import CRS, BBox
from sentinelhub.areas import UtmZoneSplitter, BBoxSplitter

from .geometry import merge_bboxes
from .multiprocess import multiprocess
from ..tasks.cropping import CroppingTask
from ..tasks.preprocessing import (
    CreatePolygonBBoxesTask,
    DropDuplicatePolygonsTask,
    FilterEmptyGeometriesTask,
    RemoveInvalidGeometryTask,
    ReprojectReferenceTask,
    TransformToPixelsCoordTask,
)
from ..tasks.training_data import ExportGeometriesAndLabelsTask, ExportRasterDataTask

Poly = Union[Polygon, MultiPolygon]


def split_zone(geometry: Poly, geometry_crs: CRS, grid_size: int, buffer: float = 0.0) -> gpd.GeoDataFrame:
    """Split zone into a UTM aligned grid.

    Args:
        geometry (Poly): Area to be split.
        geometry_crs (CRS): CRS of the input area.
        grid_size (int): Size of the grid.

    Returns:
        [gpd.geoDataFrame]: GeoDataFrame where each row represents a grid cell. Doesn't have CRS defined.
    """
    splitter = UtmZoneSplitter([geometry], geometry_crs, grid_size)
    bboxes = gpd.GeoDataFrame(
        [{"geometry": bbox.geometry, "bbox_crs": str(bbox.crs)} for bbox in splitter.get_bbox_list(buffer=buffer)]
    )
    return bboxes


def bboxes_to_wgs84(df: gpd.GeoDataFrame, crs_column: str = "bbox_crs") -> gpd.GeoDataFrame:
    """Converts bboxes returned from grid splitter to wGS84 and concatenates into single dataframe

    Args:
        df (gpd.GeoDataFrame): Input dataframe. Requires  column with CRS info.
        crs_column (str): column from which CRS of each geometry (row) is retrieved

    Returns:
        gpd.GeoDataFrame: Returns dataframe with bboxes from all UTMs in WGS84.
    """

    dfs = [
        df[df[crs_column] == crs].copy().set_crs(str(crs), inplace=True).to_crs("epsg:4326")
        for crs in df[crs_column].unique()
    ]
    return pd.concat(dfs)


def reproject_poly(poly: Poly, in_crs: str, out_crs: str) -> Poly:
    """Transforms polygon geometry from input crs to output crs.

    Args:
        poly (Poly): Polygon that will be transformed.
        in_crs (str): CRS of the polygon before transformation.
        out_crs (str): CRS of the polygon after transformation.
    Returns:
        Poly: Original polygon transformed into  a new CRS.
    """
    in_crs = pyproj.CRS(in_crs)
    out_crs = pyproj.CRS(out_crs)

    project = pyproj.Transformer.from_crs(in_crs, out_crs, always_xy=True).transform
    return transform(project, poly)


def construct_eopatch_name(geometry: Poly, out_crs: Optional[str] = None) -> str:
    """Constructs eopatch name from coordinates of the geometry.

    Args:
        geometry (Poly): Bounding box geometry (in WGS84)
        out_crs (str): If specified, the coordinates will be taken from geometry transformed to this crs.
    Returns:
        str: Name of the eopatch constructed from geometry coordinates.
    """
    if out_crs is not None:
        geometry = reproject_poly(geometry, "EPSG:4326", out_crs=out_crs)
    return f"eopatch-{int(geometry.bounds[0]):d}-{int(geometry.bounds[1]):d}"


def get_extent(eopatch: EOPatch) -> Tuple[float, float, float, float]:
    """Calculate the extent (bounds) of the patch.

    Args:
        eopatch: EOPatch for which the extent is calculated.
    Returns:
        Tuple[float, float, float, float]: The list of EOPatch bounds (min_x, max_x, min_y, max_y)
    """
    return eopatch.bbox.min_x, eopatch.bbox.max_x, eopatch.bbox.min_y, eopatch.bbox.max_y


def get_features(config):
    return {
        "bands": (FeatureType.DATA, config["bands_feature"]),
        "data_mask": (FeatureType.MASK, config["data_mask_feature"]),
        "reference": (FeatureType.VECTOR_TIMELESS, config["reference_feature"]),
        "cloud_mask": (FeatureType.MASK, config["cloud_mask_feature"]) if config.get("cloud_mask_feature") else None,
        "valid_reference_mask": (FeatureType.MASK_TIMELESS, config["valid_reference_mask_feature"])
        if config.get("valid_reference_mask_feature")
        else None,
        "grid": [
            (FeatureType.VECTOR_TIMELESS, f"{config['cropped_grid_feature']}_{size}") for size in config["scale_sizes"]
        ],
        "intersection": [(FeatureType.VECTOR_TIMELESS, f"BBOXES_IN_GRID_{size}") for size in config["scale_sizes"]],
        "data_stack": [(FeatureType.META_INFO, f"DATA_STACK_{size}") for size in config["scale_sizes"]]
        # Because the shape of arrays will be (n, h, w, b) and n is not the number of timestamps. These features
        # don't have to be saved, hence we can just put them under META_INFO
    }


def preprocess_workflow(config):

    features = get_features(config)

    reproject = ReprojectReferenceTask(reference_feature=features["reference"])
    filter_bands = ExtractBandsTask(features["bands"], features["bands"], bands=config["bands"])
    drop_duplicates = DropDuplicatePolygonsTask(features["reference"], features["reference"], column="geometry")
    create_bboxes = CreatePolygonBBoxesTask(features["reference"], features["reference"])
    filter_empty_geometries = FilterEmptyGeometriesTask(
        features["reference"], features["reference"], column=config["bbox_type"]
    )
    remove_invalid = RemoveInvalidGeometryTask(features["reference"], features["reference"])
    transform_coordinates = TransformToPixelsCoordTask(
        features["reference"],
        features["reference"],
        bbox_column=config["bbox_type"],
        resolution=config["resolution"],
        round_decimals=1,
    )
    cropping_tasks = []
    for size, grid_feature, intersection_feature, data_stack_feature in zip(
        config["scale_sizes"], features["grid"], features["intersection"], features["data_stack"]
    ):
        cropping = CroppingTask(
            raster_feature=features["bands"],
            data_mask_feature=features["data_mask"],
            cloud_mask_feature=features["cloud_mask"],
            valid_reference_mask_feature=features["valid_reference_mask"],
            take_closest_time_frame=config.get("take_closest_time_frame"),
            no_data_value=config["no_data_value"],
            vector_feature=features["reference"],
            grid_feature=grid_feature,
            intersection_feature=intersection_feature,
            data_stack_feature=data_stack_feature,
            size=size,
            overlap=config["overlap"],
            resolution=config["resolution"],
            valid_threshold=config["valid_thr"],
        )
        cropping_tasks.append(cropping)
    remove_bands_feature = RemoveFeatureTask([features["bands"], features["data_mask"]])

    workflow_nodes = linearly_connect_tasks(
        reproject,
        filter_bands,
        drop_duplicates,
        remove_invalid,
        create_bboxes,
        filter_empty_geometries,
        transform_coordinates,
        *cropping_tasks,
        remove_bands_feature,
    )

    return EOWorkflow(workflow_nodes=workflow_nodes)


def training_data_workflow(config, sh_config=None, filesystem=None):
    """Creates an EOWorkflow for creating training data"""
    features = get_features(config)
    features_to_load = [
        features["bands"],
        features["reference"],
        features["data_mask"],
        FeatureType.BBOX,
        FeatureType.TIMESTAMP,
    ]
    if features["cloud_mask"]:
        features_to_load.append(features["cloud_mask"])
    if features["valid_reference_mask"]:
        features_to_load.append(features["valid_reference_mask"])
    load = LoadTask(path=config["data_dir"], features=features_to_load, config=sh_config, filesystem=filesystem)
    save_vector = SaveTask(
        path=config["tmp_dir"],
        features=[features["reference"], *features["grid"]],
        overwrite_permission=OverwritePermission.OVERWRITE_FEATURES,
        config=sh_config,
    )

    preprocess_tasks = [node.task for node in preprocess_workflow(config).get_nodes()]

    output_filesystem = get_filesystem(config["out_dir"], config=sh_config)
    for folder in ["images", "labels"]:
        output_filesystem.makedirs(folder, recreate=True)

    export_tasks = []
    for grid_feature, intersection_feature, data_stack_feature in zip(
        features["grid"], features["intersection"], features["data_stack"]
    ):
        export_data = ExportRasterDataTask(
            raster_feature=data_stack_feature,
            grid_feature=grid_feature,
            path=join_path(config["out_dir"], "images"),
            config=sh_config,
        )
        export_labels = ExportGeometriesAndLabelsTask(
            reference_feature=intersection_feature, path=join_path(config["out_dir"], "labels"), config=sh_config
        )
        export_tasks.extend([export_data, export_labels])

    workflow_nodes = linearly_connect_tasks(
        load,
        *preprocess_tasks,
        save_vector,
        *export_tasks,
    )

    return EOWorkflow(workflow_nodes=workflow_nodes)


def merge_multiprocess(
    gdf: gpd.GeoDataFrame,
    iou_method: Callable[[Polygon, gpd.GeoSeries], pd.Series],
    split_size: Tuple[int, int] = (100, 100),
    iou_thr: float = 0.4,
    sorting_col: str = "area",
    max_workers: int = 5,
) -> gpd.GeoDataFrame:

    spatial_idx = gdf.sindex
    bounds = BBox(spatial_idx.bounds, crs=CRS(gdf.crs.to_epsg()))

    splitter = BBoxSplitter([bounds.geometry], bounds.crs, split_size)

    split_bboxes = []
    done_ilocs = set()
    for bb in splitter.bbox_list:
        ilocs = set(spatial_idx.query(bb.geometry))
        a = gdf.iloc[list(ilocs.difference(done_ilocs))].copy()
        done_ilocs.update(ilocs)
        if not a.empty:
            split_bboxes.append(a.reset_index(drop=True))
    del gdf

    merger = partial(merge_bboxes, iou_method=iou_method, iou_thr=iou_thr, sorting_col=sorting_col)

    results = multiprocess(merger, split_bboxes, total=len(split_bboxes), max_workers=max_workers)

    return pd.concat(results)
