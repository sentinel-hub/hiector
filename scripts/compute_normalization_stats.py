import argparse
import json
import logging
import os
import sys
from functools import partial

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from hiector.utils.aws_utils import LocalFile, get_filesystem

stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [stdout_handler]
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", handlers=handlers
)
LOGGER = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description="Compute normalization factors")
parser.add_argument("--config", type=str, help="Path to config file with execution parameters", required=True)
args = parser.parse_args()

statistic_mapping = {
    "mean": np.mean,
    "median": np.median,
    "min": np.min,
    "max": np.max,
    "perc1": partial(np.percentile, q=1),
    "perc5": partial(np.percentile, q=5),
    "perc95": partial(np.percentile, q=95),
    "perc99": partial(np.percentile, q=99),
}


def compute_norm_stats(config):
    filesystem = get_filesystem(config["bucket_name"], config["aws_profile"])
    LOGGER.info("Opening the file descriptor the the samples file.")
    with LocalFile(config["samples_file"], mode="r", filesystem=filesystem) as f:
        layers = fiona.listlayers(f.path)
        layers_to_read = [x for x in layers if int(x.split("_")[1]) in config["scales"]]
        LOGGER.info(f"Reading and concatenating layers: {layers_to_read}")
        gdf = pd.concat([gpd.read_file(f.path, layer=layer) for layer in layers_to_read])

    if "query" in config:
        gdf = gdf.query(config["query"])

    gdf = gdf.sample(frac=config["fraction"], replace=False)
    sampled = []
    LOGGER.info("Sampling images...")
    for image_name in tqdm(gdf.NAME.values):
        imgpath = os.path.join(config["data_dir"], "images", f"{image_name}.npy")
        imgs = np.load(filesystem.openbin(imgpath, "rb"))
        sampled.append(imgs[np.newaxis, ...])
    sampled = np.concatenate(sampled)
    rows = []
    LOGGER.info("Calculating statistics...")
    for (
        statistic_name,
        statistic_f,
    ) in statistic_mapping.items():
        rows.append(
            {
                "modality": config["modality"],
                "statistic": statistic_name,
                "B": statistic_f(sampled[..., 0]),
                "G": statistic_f(sampled[..., 1]),
                "R": statistic_f(sampled[..., 2]),
                "N": statistic_f(sampled[..., 3]),
            }
        )
    rows = pd.DataFrame(rows)
    output_file = config["output_file"]
    with filesystem.open(output_file, "w") as f:
        LOGGER.info(f"Saving to file {output_file}..")
        rows.to_csv(f)


if __name__ == "__main__":
    LOGGER.info(f"Reading configuration from {args.config}")
    with open(args.config, "r") as jfile:
        cfg_dict = json.load(jfile)
    cfg = cfg_dict["compute_norm_stats"]
    compute_norm_stats(cfg)
