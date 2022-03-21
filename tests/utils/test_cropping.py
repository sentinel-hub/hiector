import os

import pytest

from eolearn.core import EOPatch, FeatureType

from hiector.tasks.cropping import CroppingTask


@pytest.fixture(name="test_eop", scope="module")
def eop(input_folder):
    """A pytest fixture to retrieve GeoDataFrame for ground truth"""
    return EOPatch.load(os.path.join(input_folder, "eop-preprocessed"))


def test_cropping_task(test_eop):
    crop = CroppingTask(
        raster_feature=(FeatureType.DATA, "bands"),
        data_mask_feature=(FeatureType.MASK, "dataMask"),
        cloud_mask_feature=None,
        valid_reference_mask_feature=None,
        no_data_value=0,
        vector_feature=(FeatureType.VECTOR_TIMELESS, "BUILDINGS"),
        grid_feature=(FeatureType.VECTOR_TIMELESS, "PATCHLETS_64"),
        intersection_feature=(FeatureType.VECTOR_TIMELESS, "BBOXES_IN_GRID_64"),
        data_stack_feature=(FeatureType.META_INFO, "DATA_STACK_64"),
        size=64,
        overlap=0.25,
        resolution=1.5,
        valid_threshold=0.6,
    )
    cropping_eop = crop.execute(test_eop, eopatch_name="preprocessed-eop")
    assert "BBOXES_IN_GRID_64" in cropping_eop.vector_timeless
    assert "PATCHLETS_64" in cropping_eop.vector_timeless
    assert "DATA_STACK_64" in cropping_eop.meta_info
    assert cropping_eop.meta_info["DATA_STACK_64"].shape == (121, 64, 64, 4)
    assert len(cropping_eop.vector_timeless["PATCHLETS_64"]) == 121
