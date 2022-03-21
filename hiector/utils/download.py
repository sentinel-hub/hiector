from eolearn.core import EOWorkflow, FeatureType, OverwritePermission, SaveTask, linearly_connect_tasks
from eolearn.geometry import VectorToRasterTask
from eolearn.io import SentinelHubEvalscriptTask, VectorImportTask
from sentinelhub import DataCollection, SHConfig


def get_download_workflow(
    collection: DataCollection,
    resolution: float,
    evalscript: str,
    sh_config: SHConfig,
    ref_path: str,
    ml_aois_path: str,
    save_path: str,
    mosaicking_order: str = "mostRecent",
    ref_feature_name: str = "reference_v0_0_1",
    ml_aois_feature_name: str = "ml_aois",
    has_ref_feature_name: str = "has_ref",
    raster_value: int = 1,
) -> EOWorkflow:
    """Construct a workflow for downloading data from SH, adding vectors and saving.

    Args:
        collection (DataCollection): SH Collection from which data will be downloaded.
        resolution (float): The resolution at which the data will be downloaded.
        evalscript (str): Evalscript used for download.
        sh_config (SHConfig): SHconfig object.
        ref_path (str): Location of the DataFrame with reference vector data.
        ml_aois_path (str): Location of the DataFrame containing ML AOI areas.
        save_path (str): Where the resulting EOPatch will be saved.
        mosaicking_order (str): Type of maosaicking applied by the SH service. Default to "mostRecent".
        ref_feature_name (str): Name of feature holding the vector reference buildings. Default to "reference_v_0_0_1".
        ml_aois_feature_name (str): Name of feature holding the vector ML AOIs. Default to "ml_aois".
        has_ref_feature_name (str): Name of feature holding the reference mask. Default to "has_ref".
        raster_value (int): Value used to rasterize the vector timeless reference feature. Default to 1.

    Returns:
        LinearWorkflow: eo-learn LinearWorkflow for downloading and processing data.
    """
    input_task = SentinelHubEvalscriptTask(
        features=[(FeatureType.DATA, "bands"), (FeatureType.MASK, "mask")],
        evalscript=evalscript,
        data_collection=collection,
        resolution=resolution,
        config=sh_config,
        mosaicking_order=mosaicking_order,
        max_threads=3,
    )
    add_ref = VectorImportTask(
        feature=(FeatureType.VECTOR_TIMELESS, ref_feature_name), path=ref_path, config=sh_config, reproject=True
    )
    add_ml_aois = VectorImportTask(
        feature=(FeatureType.VECTOR_TIMELESS, ml_aois_feature_name), path=ml_aois_path, config=sh_config, reproject=True
    )
    rasterise = VectorToRasterTask(
        vector_input=(FeatureType.VECTOR_TIMELESS, ml_aois_feature_name),
        raster_feature=(FeatureType.MASK_TIMELESS, has_ref_feature_name),
        values=raster_value,
        raster_resolution=resolution,
    )
    save = SaveTask(path=save_path, overwrite_permission=OverwritePermission.OVERWRITE_PATCH, config=sh_config)
    return EOWorkflow(workflow_nodes=linearly_connect_tasks(input_task, add_ref, add_ml_aois, rasterise, save))
