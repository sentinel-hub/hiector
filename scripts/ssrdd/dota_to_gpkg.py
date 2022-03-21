import argparse
import json
import os
from collections import defaultdict
from functools import partial

import fs as fsys
import geopandas as gpd
import pandas as pd

from hiector.utils.aws_utils import LocalFile, get_filesystem
from hiector.utils.grid import merge_multiprocess
from hiector.utils.metrics import iou_single, convert_dota
from hiector.utils.multiprocess import multiprocess

parser = argparse.ArgumentParser(description="Converts predictions from DOTA into a combined GPKG.")

parser.add_argument("--config", type=str, help="Path to config file with execution parameters", required=True)
args = parser.parse_args()


def save_dota_to_gpkg(
    eop_dota: str, dota_dir: str, eops_dir: str, aws_profile: str, aws_bucket: str, resolution: int, out_folder: str
) -> None:
    filesystem = get_filesystem(aws_bucket, aws_profile)
    try:
        df = convert_dota(dota_dir, eops_dir, eop_dota, filesystem, resolution)
    except fsys.errors.ResourceNotFound as e:
        print(f"Could not open DOTA file: {eop_dota}. Skipping.")
        return
    except AttributeError as e:
        print(f"Could not open EOPatch for: {eop_dota}. Skipping.")
        return
    eopname = eop_dota.split("_")[2][:-4]
    for utm in df:
        filepath = os.path.join(out_folder, f"{eopname}_{utm}.gpkg")
        with LocalFile(filepath, mode="w", filesystem=filesystem) as outf:
            df[utm].to_file(outf.path, driver="GPKG")


def load_gpkg(gpkg):
    basename = os.path.splitext(os.path.basename(gpkg))[0]
    _, epsg = basename.split("_")
    try:
        with LocalFile(gpkg, filesystem=fs) as f:
            file = gpd.read_file(f.path)
            fs.remove(gpkg)
            return file, epsg
    except Exception as e:
        print(f"Something went wrong for {gpkg} with error {e}")
        return None, None


if __name__ == "__main__":
    # read config parameter file
    with open(args.config, "r") as jfile:
        cfg_dict = json.load(jfile)

    cfg = cfg_dict["execute"]
    fs = get_filesystem(cfg["s3_bucket_name"], cfg["s3_profile_name"])
    modality = cfg["datasources"]["evaluate"]["modality"]
    resolution = cfg["datasources"]["evaluate"]["resolution"]
    eopatches_dir = cfg["datasources"]["evaluate"]["eopatches_dir"]
    data_dir = cfg["datasources"]["evaluate"]["data_dir"]
    evaluate_gdf = cfg["grid_file"]
    dota_dir = cfg["aws_dota_dir"]
    gpkg_dir = cfg["aws_gpkg_dir"]
    num_workers = cfg["num_workers"]
    with LocalFile(evaluate_gdf, filesystem=fs) as f:
        gdf = gpd.read_file(f.path)
        dota_eopatches = gdf.eopatch.unique()
        dota_filenames = [f"Task1_building_{x}.txt" for x in dota_eopatches]
        gpkg_filenames = [
            f"{eop}_{crs.lower()}.gpkg" for eop, crs in gdf[["eopatch", "bbox_crs"]].drop_duplicates().values
        ]

    save_gpkg_func = partial(
        save_dota_to_gpkg,
        dota_dir=dota_dir,
        eops_dir=eopatches_dir,
        aws_profile=cfg["s3_profile_name"],
        aws_bucket=cfg["s3_bucket_name"],
        resolution=resolution,
        out_folder=gpkg_dir,
    )

    _ = multiprocess(save_gpkg_func, dota_filenames, max_workers=num_workers)

    predictions_per_utm = defaultdict(list)

    gpkgs = [os.path.join(gpkg_dir, gpkg) for gpkg in gpkg_filenames]
    results = multiprocess(load_gpkg, gpkgs, max_workers=num_workers)
    for gpkg, epsg in results:
        if gpkg is not None:
            predictions_per_utm[epsg].append(gpkg)

    with LocalFile(os.path.join(gpkg_dir, f"predictions_merged_{modality}.gpkg"), mode="w", filesystem=fs) as f:
        for utm in predictions_per_utm:
            utm_predictions = pd.concat(predictions_per_utm[utm])
            merged = merge_multiprocess(
                utm_predictions, iou_single, sorting_col="pseudo_probability", max_workers=num_workers
            )
            merged.to_file(f.path, layer=utm, driver="GPKG")
