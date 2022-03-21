import argparse
import json
import logging
import sys

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import torch

from eolearn.core.utils.fs import get_aws_credentials, join_path
from sentinelhub import SHConfig

from hiector.utils.aws_utils import LocalFile
from hiector.utils.training_data import filter_dataframe, train_test_val_split

stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [stdout_handler]
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", handlers=handlers
)
LOGGER = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description="Execute training and evaluation using the SSRDD model.\n")

parser.add_argument("--config", type=str, help="Path to config file with execution parameters", required=True)

args = parser.parse_args()


def prepare_data(config):
    data_dir = config["data_dir"]
    sh_config = SHConfig()

    if config["aws_profile"]:
        sh_config = get_aws_credentials(aws_profile=config["aws_profile"], config=sh_config)

    input_gpkg_path = join_path(data_dir, config["input_dataframe_filename"])
    dfs = []
    with LocalFile(input_gpkg_path, mode="r", config=sh_config) as local_file:
        for layername in fiona.listlayers(local_file.path):
            scale = int(layername.split("_")[1])
            # Assumes layers to be named as PATCHLETS_<SCALE>_<CRS_CODE>
            if scale in config["scale_sizes"]:
                LOGGER.info(f"Reading layer: {layername}")
                df = gpd.read_file(local_file.path, layer=layername)
                df["CRS"] = str(df.crs)
                df["LAYER_NAME"] = layername
                df["SCALE"] = scale
                # Convert to WGS84, because we want stuff from different CRSes to be stored together
                df = df.to_crs("epsg:4326")
                dfs.append(df)

    LOGGER.info("Concatenating layers together.")
    dataframe = pd.concat(dfs)
    LOGGER.info("Filtering dataframe.")
    filtered_df = filter_dataframe(
        dataframe,
        query=config.get("query"),
        frac=config.get("frac"),
        exclude_eops=config.get("exclude_eops"),
        seed=config.get("seed"),
    )

    LOGGER.info("Performing train/test/val/split.")
    split_df = train_test_val_split(
        filtered_df,
        fraction_train=config["fraction_train"],
        fraction_test=config["fraction_test"],
        fraction_val=config["fraction_val"],
    )
    output_gpkg_path = join_path(data_dir, config["output_dataframe_filename"])

    LOGGER.info(f"Saving prepared dataframe to: {output_gpkg_path}")
    with LocalFile(output_gpkg_path, mode="w", config=sh_config) as local_file:
        split_df.to_file(local_file.path, driver="GPKG")


if __name__ == "__main__":
    # read config parameter file
    LOGGER.info(f"Reading configuration from {args.config}")
    with open(args.config, "r") as jfile:
        cfg_dict = json.load(jfile)

    cfg = cfg_dict["select-data"]

    torch.manual_seed(0)
    np.random.seed(cfg["seed"])
    prepare_data(cfg)
