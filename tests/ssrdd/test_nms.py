import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from hiector.ssrdd.utils.box.bbox_np import xywha2xy4
from hiector.ssrdd.utils.box.rbbox_np import rbbox_batched_nms


def to_polygon(bb):
    bbox = xywha2xy4(bb).ravel()
    return Polygon(list(zip(bbox[::2], bbox[1::2])))


@pytest.fixture(name="npys_in", scope="module")
def get_input_np_files(input_folder):
    """A pytest fixture to retrieve the input numpy arrays"""
    return (
        np.load(os.path.join(input_folder, "test-nms-bboxes.npy")),
        np.load(os.path.join(input_folder, "test-nms-scores.npy")),
        np.load(os.path.join(input_folder, "test-nms-labels.npy")),
    )


@pytest.fixture(name="gdf_out", scope="module")
def get_gdf_out(input_folder):
    """A pytest fixture to retrieve the output GeoDataFrame"""
    return gpd.read_file(os.path.join(input_folder, "merged-bboxes.gpkg"))


def test_nms(npys_in, gdf_out, output_folder):
    bboxes, scores, labels = npys_in

    geo_df = gpd.GeoDataFrame(data={"labels": labels, "scores": scores}, geometry=[to_polygon(b) for b in bboxes])

    nms = rbbox_batched_nms(bboxes, scores, labels, iou_thresh=0.5)
    gdf_nms = geo_df.iloc[nms].copy()
    gdf_nms = gdf_nms.sort_values("scores", ascending=False)

    gdf_nms.to_file(os.path.join(output_folder, "nms-output.gpkg"), driver="GPKG")

    pd.testing.assert_frame_equal(
        gdf_nms.reset_index(drop=True),
        gdf_out[["labels", "scores", "geometry"]],
        check_dtype=False,
        check_index_type=False,
    )
