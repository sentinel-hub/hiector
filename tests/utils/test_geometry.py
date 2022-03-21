import geopandas as gpd
import pytest
from shapely.geometry import Polygon, box

from hiector.utils.geometry import merge_bboxes
from hiector.utils.metrics import intersection_over_smallest, iou_single


@pytest.fixture(name="basic_bboxes_example", scope="module")
def basic_bboxes_example():
    """A pytest fixture to retrieve GeoDataFrame for ground truth"""
    polys = [box(0, 0, 100, 100), box(0, 0, 50, 50), box(75, 75, 110, 110), box(5, 5, 95, 95), box(50, 102, 70, 105)]
    example_basic = gpd.GeoDataFrame(geometry=polys, crs="epsg:32639")
    example_basic["area"] = example_basic.area
    return example_basic


def test_merge_bboxes(basic_bboxes_example):
    merged_iou_01 = merge_bboxes(basic_bboxes_example, iou_method=intersection_over_smallest, iou_thr=0.1)

    assert len(merged_iou_01) == 2
    assert merged_iou_01.iloc[0].geometry == basic_bboxes_example.iloc[0].geometry
    assert merged_iou_01.iloc[1].geometry == basic_bboxes_example.iloc[4].geometry

    merged_iou_09 = merge_bboxes(basic_bboxes_example, iou_method=intersection_over_smallest, iou_thr=0.9)

    assert len(merged_iou_09) == 3
    assert merged_iou_09.iloc[0].geometry == basic_bboxes_example.iloc[0].geometry
    assert merged_iou_09.iloc[1].geometry == basic_bboxes_example.iloc[2].geometry
    assert merged_iou_09.iloc[2].geometry == basic_bboxes_example.iloc[4].geometry
