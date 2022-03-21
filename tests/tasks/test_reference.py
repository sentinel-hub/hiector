import os
from copy import deepcopy

import numpy as np
import pytest

from eolearn.core import EOPatch, FeatureType

from hiector.tasks.reference import MergeBBoxesTask, PrepareMRRTask
from hiector.utils.metrics import intersection_over_smallest


@pytest.fixture(name="test_eop", scope="module")
def test_eop(input_folder):
    eop = EOPatch.load(os.path.join(input_folder, "test-eop"))
    return eop


@pytest.fixture(name="test_eop_prepared", scope="module")
def test_eop_prepared(input_folder):
    eop = EOPatch.load(os.path.join(input_folder, "test-eop-prepared"))
    return eop


def test_prepare_mmr(test_eop):

    task = PrepareMRRTask(
        input_feature=(FeatureType.VECTOR_TIMELESS, "BUILDINGS"),
        output_feature=(FeatureType.VECTOR_TIMELESS, "BUILDINGS"),
    )
    prepared = task.execute(deepcopy(test_eop))

    vec_in = test_eop.vector_timeless["BUILDINGS"]
    vec_out = prepared.vector_timeless["BUILDINGS"]

    assert len(vec_in) >= len(vec_out)
    for _id, geom in vec_in[["_ID_", "geometry"]].values:
        assert np.array([x.buffer(1e-1).contains(geom) for x in vec_out.geometry.values], dtype=bool).sum() > 0, _id


def test_merge_bbox(test_eop_prepared):
    merge_task = MergeBBoxesTask(
        input_feature=(FeatureType.VECTOR_TIMELESS, "BUILDINGS_MRR"),
        output_feature=(FeatureType.VECTOR_TIMELESS, "BUILDINGS_MRR"),
        iou_method=intersection_over_smallest,
        iou_thr=0.4,
        sorting_col="area",
    )

    eop_merged = merge_task.execute(deepcopy(test_eop_prepared))

    vec_in = test_eop_prepared.vector_timeless["BUILDINGS_MRR"]
    vec_out = eop_merged.vector_timeless["BUILDINGS_MRR"]

    assert set(vec_out._ID_.values) == {7417907, 8063303, 8375354, 8376000}
