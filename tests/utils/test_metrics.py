import os

import geopandas as gpd
import pandas as pd
import pytest

from sentinelhub import CRS, BBox

from hiector.utils.metrics import dota_results_to_gdf, get_ap, get_error_matrix, iou


@pytest.fixture(name="gdf_gt", scope="module")
def gdf_gt(input_folder):
    """A pytest fixture to retrieve GeoDataFrame for ground truth"""
    return gpd.read_file(os.path.join(input_folder, "gdf-gt.gpkg"))


@pytest.fixture(name="gdf_pr", scope="module")
def gdf_pr(input_folder):
    """A pytest fixture to retrieve GeoDataFrame for ground truth"""
    return gpd.read_file(os.path.join(input_folder, "gdf-pr.gpkg"))


@pytest.fixture(name="eop_data", scope="module")
def eop_data(input_folder):
    """A pytest fixture to retrieve dictionary with EOPatch BBox information"""
    df = pd.read_csv(os.path.join(input_folder, "eop_data.csv"))
    eop_data = {}
    for item in df.iteritems():
        eop_data[item[0]] = {
            "bbox": BBox([float(coord) for coord in item[1].values[0].split(",")], crs=CRS(int(item[1].values[1]))),
            "transform": tuple(float(cc) for cc in item[1].values[2].split("(")[-1].split(")")[0].split(",")),
        }
    return eop_data


def test_iou(gdf_gt, gdf_pr):

    iou_ = iou(gdf_gt, gdf_pr)

    assert len(iou_) == 35
    assert len(iou_[iou_.iou == 1]) == 1
    assert len(iou_[iou_.iou == 0]) == 7
    assert len(iou_[iou_.iou >= 0.5]) == 4

    iou_dupl = iou(gdf_gt, pd.concat([gdf_pr, gdf_pr[:1]]), drop_duplicates=False)
    iou_nodupl = iou(gdf_gt, pd.concat([gdf_pr, gdf_pr[:1]]), drop_duplicates=True)

    assert len(iou_dupl) == 36
    assert len(iou_nodupl) == 35


def test_error_matrix(gdf_gt, gdf_pr):

    for proba in [0.0, 0.5]:
        errors = get_error_matrix(gdf_gt, gdf_pr, proba_thr=proba, iou_thr=0.5)

        assert all([errors[df] is None for df in ["TP_df", "FP_df", "FN_df"]])
        assert errors["TP"] == 4
        assert errors["FN"] == 96
        assert errors["FP"] + errors["TP"] == len(gdf_pr[gdf_pr.pseudo_probability >= proba])

    errors = get_error_matrix(gdf_gt, gdf_pr, proba_thr=0.0, iou_thr=0.0, return_dfs=True)
    assert errors["TP"] == 35
    assert all([isinstance(errors[df], gpd.GeoDataFrame) for df in ["TP_df", "FP_df", "FN_df"]])

    # check geometries in TP and FP match the predicted ones
    pd.testing.assert_frame_equal(errors["TP_df"][["geometry"]][1:2], gdf_pr[["geometry"]][1:2])
    pd.testing.assert_frame_equal(errors["FP_df"][["geometry"]][:1], gdf_pr[["geometry"]][5:6])


def test_ap(gdf_gt, gdf_pr):

    proba = 0.0
    ap, mrec, mprec = get_ap(gdf_gt, gdf_pr, proba_thr=proba, iou_thr=0.0)

    assert ap == 0.3108179542562844
    assert (len(mrec) == len(mprec)) and (len(mrec) == len(gdf_pr[gdf_pr.pseudo_probability >= proba]) + 2)


def test_dota_results_to_gdf(input_folder, eop_data):
    detections_file = os.path.join(input_folder, "Task1_building.txt")
    results = dota_results_to_gdf(detections_file, eop_data, underscore=False)

    assert set(results.keys()) == {"epsg:32638", "epsg:32639"}, "Wrong key dictionaries generated"
    for crs in ["32638", "32639"]:
        assert results[f"epsg:{crs}"].crs == CRS(crs).pyproj_crs(), "Incorrect CRS"
