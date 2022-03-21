"""
Prepare training data by processing EOPatches
"""
import argparse
import json
import logging
import sys

import fs
import geopandas as gpd
import ray
from tqdm.auto import tqdm

from eolearn.core import EOExecutor, EOPatch, FeatureType, LoadTask, SaveTask, get_filesystem
from eolearn.core.extra.ray import RayExecutor
from eolearn.core.utils.fs import get_aws_credentials, join_path
from sentinelhub import SHConfig

from hiector.tasks.cropping import CroppingTask
from hiector.utils.aws_utils import LocalFile
from hiector.utils.grid import training_data_workflow
from hiector.utils.vector import export_geopackage

stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [stdout_handler]
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", handlers=handlers
)
LOGGER = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Process EOPatches and prepare data for training/testing.\n")
parser.add_argument("--config", type=str, help="Path to config file with execution parameters", required=True)
args = parser.parse_args()


def get_execution_arguments(workflow, eopatch_names):
    """Prepares execution parameters for an EOWorkflow"""
    exec_args = []
    nodes = workflow.get_nodes()
    for eopatch_name in eopatch_names:
        single_exec_dict = {}
        for node in nodes:
            if isinstance(node.task, (SaveTask, LoadTask)):
                single_exec_dict[node] = dict(eopatch_folder=eopatch_name)
            if isinstance(node.task, CroppingTask):
                single_exec_dict[node] = dict(eopatch_name=eopatch_name)
        exec_args.append(single_exec_dict)

    return exec_args


def run_execution(workflow, exec_args, eopatch_names, config):

    """Runs EOWorkflow execution"""
    if config["use_ray"]:
        executor_cls = RayExecutor
        run_args = dict()
    else:
        executor_cls = EOExecutor
        run_args = dict(workers=config["workers"])
    executor = executor_cls(
        workflow,
        exec_args,
        save_logs=False,  # TODO: logs are also being sent to stout
        logs_folder=config["logs_dir"],
        execution_names=eopatch_names,
    )
    executor.run(**run_args)
    executor.make_report()

    successful = executor.get_successful_executions()
    failed = executor.get_failed_executions()
    LOGGER.info(
        "EOExecution finished with %d / %d success rate",
        len(successful),
        len(successful) + len(failed),
    )
    return successful, failed


def export_grids(config, eopatch_names, sh_config):
    """Exports Geopackages with grids of EOPatches and grids of training patchlets"""
    filename_ref = f"buildings-{config['bbox_type']}.gpkg"
    filename_grid = "-".join(
        map(str, ["grid", config["bbox_type"], *config["scale_sizes"], config["overlap"], config["valid_thr"]])
    )
    ref_geopackage_path = join_path(config["out_dir"], filename_ref)
    grid_geopackage_path = join_path(config["out_dir"], f"{filename_grid}.gpkg")

    input_filesystem = get_filesystem(config["tmp_dir"], config=sh_config)

    grid_features = [
        (FeatureType.VECTOR_TIMELESS, f"{config['cropped_grid_feature']}_{size}") for size in config["scale_sizes"]
    ]
    reference_feature = (FeatureType.VECTOR_TIMELESS, config["reference_feature"])
    features = grid_features + [reference_feature]

    columns = ["NAME", "EOPATCH_NAME", "N_BBOXES", "IS_DATA_RATIO", "VALID_DATA_RATIO"]
    if config.get("cloud_mask_feature"):
        columns.append("CLOUD_COVERAGE")
    if config.get("valid_reference_mask_feature"):
        columns.append("HAS_REF_RATIO")
    with LocalFile(ref_geopackage_path, mode="w", config=sh_config) as ref_file, LocalFile(
        grid_geopackage_path, mode="w", config=sh_config
    ) as grid_file:
        for eopatch_name in tqdm(eopatch_names, desc=f"Creating {ref_geopackage_path}, {grid_geopackage_path}"):
            eopatch = EOPatch.load(eopatch_name, filesystem=input_filesystem, features=features)
            export_geopackage(
                eopatch=eopatch,
                geopackage_path=ref_file.path,
                feature=reference_feature,
                geometry_column=config["bbox_type"],
                columns=["area"],
            )
            for grid_feature in grid_features:
                export_geopackage(
                    eopatch=eopatch, geopackage_path=grid_file.path, feature=grid_feature, columns=columns
                )


def main():
    LOGGER.info(f"Reading configuration from {args.config}")
    with open(args.config, "r") as jfile:
        full_config = json.load(jfile)

    config = full_config["prepare_eopatch"]

    if config["use_ray"]:
        ray.init(address="auto")

    sh_config = SHConfig()
    if config["aws_profile"]:
        sh_config = get_aws_credentials(aws_profile=config["aws_profile"], config=sh_config)

    workflow = training_data_workflow(config, sh_config)

    dirname, basename = fs.path.dirname(config["grid_file"]), fs.path.basename(config["grid_file"])
    filesystem = get_filesystem(dirname, config=sh_config)

    with LocalFile(basename, mode="r", filesystem=filesystem) as gridfile:
        eopatch_names = list(gpd.read_file(gridfile.path).eopatch.values)
    exec_args = get_execution_arguments(workflow, eopatch_names)

    finished, failed = run_execution(workflow, exec_args, eopatch_names, config)
    if failed:
        LOGGER.info("Some executions failed. The produced Geopackages might not have all EOPatches!")
    eopatch_names = [eopatch_names[index] for index in finished]

    export_grids(config, eopatch_names, sh_config)

    # Clean up data in temp dir
    LOGGER.info(f"Cleaning up temporary directory")
    tmp_filesystem = get_filesystem(config["tmp_dir"], config=sh_config)
    tmp_filesystem.removetree("/")


if __name__ == "__main__":
    main()
