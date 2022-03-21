from eolearn.core import FeatureType
from eolearn.io import SentinelHubInputTask
from sentinelhub import DataCollection, SHConfig


def s2_data_input_task(
    maxcc: float = 0.2, mosaicking_order: str = "leastCC", config: SHConfig = None
) -> SentinelHubInputTask:
    """Construct a SentinelHubInput task to download single mosaicked image from time interval

    Args:
        maxcc (float): max cloud coverage to be used
        mosaicking_order (str): mosaicking order (e.g. mostRecent)
        config (SHConfig): SHConfig object

    Returns:
        SentinelHubInputTask: eo-learn task to download (all bands + dataMask + CLM) Sentinel-2 data
    """
    return SentinelHubInputTask(
        bands_feature=(FeatureType.DATA, "BANDS"),
        bands=["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"],
        data_collection=DataCollection.SENTINEL2_L1C,
        single_scene=True,
        resolution=10,
        additional_data=[(FeatureType.MASK, "dataMask"), (FeatureType.MASK, "CLM")],
        maxcc=maxcc,
        config=config,
        mosaicking_order=mosaicking_order,
    )
