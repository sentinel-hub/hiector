import argparse
import json
import logging
import sys
from itertools import product

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm.auto import tqdm as tqdm

from sentinelhub import CRS

from hiector.utils.aws_utils import LocalFile, get_filesystem
from hiector.utils.metrics import get_ap, get_bbox
from hiector.utils.multiprocess import multiprocess

stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [stdout_handler]
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", handlers=handlers
)
LOGGER = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description="Compute mAP on predicted OBBs.\n")

parser.add_argument("--config", type=str, help="Path to config file with execution parameters", required=True)

args = parser.parse_args()


RANGE = np.arange(0.0, 0.7, 0.05)
REF_CRS = "epsg:4326"


def get_ap_unpack(args):
    return get_ap(*args)


def compute_ap(config):

    filesystem = get_filesystem(bucket_name=config["s3_bucket_name"], profile_name=config["s3_profile_name"])

    LOGGER.info(f"Read predictions file: {config['predictions_filename']}")
    predictions = []
    with LocalFile(config["predictions_filename"], mode="r", filesystem=filesystem) as f:
        crss = fiona.listlayers(f.path)
        LOGGER.info(f"Reading and concatenating predictions: {crss}")
        for crs in crss:
            predictions.append(gpd.read_file(f.path, layer=crs).to_crs(crs=CRS(REF_CRS).pyproj_crs()))

    predictions = gpd.GeoDataFrame(pd.concat(predictions, axis=0, ignore_index=True))
    predictions.set_crs(crs=CRS(REF_CRS).pyproj_crs(), inplace=True)

    eopatches = predictions.eopatch.unique()
    to_keep = config["eopatch_names"]
    if to_keep is not None:
        eopatches = [eopatch for eopatch in eopatches if eopatch in to_keep]

    predictions = predictions[predictions["eopatch"].isin(eopatches)]

    LOGGER.info(f"Retrieving bounding box information")
    eop_data = {
        eopatch: get_bbox(eopatch, config["eopatches_dir"], filesystem, config["resolution"])
        for eopatch in tqdm(eopatches)
    }

    LOGGER.info(f"Read reference file: {config['reference_filename']}")
    with filesystem.openbin(config["reference_filename"], "rb") as handle_file:
        gt = gpd.read_file(handle_file)

    gt = gt[gt.is_valid]
    gt = gt[~gt.is_empty]
    gt.geometry = gt.geometry.buffer(0)
    gt.to_crs(CRS(REF_CRS).pyproj_crs(), inplace=True)
    assert gt.crs == predictions.crs, "CRSs don't match"

    if config["ml_aois_filename"] is not None:
        LOGGER.info(f"Read ML AOIs file: {config['ml_aois_filename']}")
        with filesystem.openbin(config["ml_aois_filename"], "rb") as handle_file:
            ml_aois = gpd.read_file(handle_file)
        ml_aois.to_crs(CRS(REF_CRS).pyproj_crs(), inplace=True)
        predictions = gpd.sjoin(predictions, ml_aois)
        gt = gpd.sjoin(gt, ml_aois)

    eopatches_gdf = gpd.GeoDataFrame(data={"eopatch": eopatches}, geometry=None, crs=CRS(REF_CRS).pyproj_crs())
    eopatches_gdf["geometry"] = eopatches_gdf.eopatch.apply(
        lambda e: eop_data[e]["bbox"].transform_bounds(CRS(REF_CRS)).geometry
    )

    gt = gt[gt.intersects(eopatches_gdf.geometry.unary_union)].copy()
    gt_obb = gt.copy()
    gt_obb.geometry = gt.geometry.apply(lambda g: g.minimum_rotated_rectangle)

    LOGGER.info(f"Computing APs for different IoUs and probas..")
    iou_thr, proba_thr = config["iou_thr"], config["proba_thr"]
    thresholds = list(product(RANGE, RANGE))
    if iou_thr is not None and proba_thr is not None:
        thresholds = [(proba_thr, iou_thr)]

    arguments = [(gt_obb, predictions, thrs[0], thrs[1]) for thrs in thresholds]
    results = multiprocess(get_ap_unpack, arguments, len(arguments), max_workers=config["max_workers"])
    aps = [dict(AP=result[0], PROBA=thrs[0], IOU=thrs[1]) for result, thrs in zip(results, thresholds)]

    LOGGER.info(f"Saving AP dataframe to: {config['aps_filename']}")
    with filesystem.open(config["aps_filename"], "w") as handle_file:
        pd.DataFrame(aps).to_csv(handle_file, index=False)


if __name__ == "__main__":
    # read config parameter file
    LOGGER.info(f"Reading configuration from {args.config}")
    with open(args.config, "r") as jfile:
        cfg_dict = json.load(jfile)

    cfg = cfg_dict["compute-ap"]

    compute_ap(cfg)
