import argparse
import json
import os

import geopandas as gpd
import ray

from eolearn.core import EOExecutor, EONode, EOWorkflow
from eolearn.core.extra.ray import RayExecutor

from hiector.tasks.execute import PredictEOPatch
from hiector.utils.aws_utils import LocalFile, get_filesystem

parser = argparse.ArgumentParser(description="Execute evaluation using the SSRDD model and Ray")

parser.add_argument("--config", type=str, help="Path to config file with execution parameters", required=True)
parser.add_argument("--on-the-fly", help="Should the grid be calculated on the fly.", action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    ray.init(address="auto")
    # read config parameter file
    with open(args.config, "r") as jfile:
        cfg_dict = json.load(jfile)

    cfg = cfg_dict["execute"]

    cfg["checkpoint"] = None
    cfg["old_version"] = False
    fs = get_filesystem(cfg["s3_bucket_name"], cfg["s3_profile_name"])

    if args.on_the_fly:
        df_filename = cfg["grid_file"]
        eopatch_col = "eopatch"
    else:
        data_dir = cfg["datasources"]["evaluate"]["data_dir"]
        metadata_filename = cfg["datasources"]["evaluate"]["metadata_filename"]
        df_filename = os.path.join(data_dir, metadata_filename)
        eopatch_col = "EOPATCH_NAME"

    with LocalFile(df_filename, filesystem=fs) as file:
        df = gpd.read_file(file.path)

    predict_eop_task = PredictEOPatch(cfg, args.on_the_fly)
    predict_eop_node = EONode(task=predict_eop_task)
    workflow = EOWorkflow(workflow_nodes=[predict_eop_node])
    exec_args = [{predict_eop_node: dict(eopatch_name=eopname)} for eopname in df[eopatch_col].unique()]

    executor = RayExecutor(workflow, exec_args)
    executor.run()
    executor.make_report()
