import argparse
import logging
import sys
from typing import Optional

import geopandas as gpd

from hiector.utils.geometry import close_holes, merge_predictions

stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [stdout_handler]
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", handlers=handlers
)
LOGGER = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Convert DOTA-style detections to geo-spatial dataframe.")

parser.add_argument("--predictions_file", type=str, help="Path to vector predictions file. ", required=True)
parser.add_argument(
    "--pseudoproba_thr",
    type=float,
    help="The pseudoprobability threshold. Predictions with pseudoprobability below this value will be discarded.",
    required=True,
)
parser.add_argument(
    "--interior_thr",
    type=float,
    help="The threshold on the size of the interior hole. Holes with are below this size will be closed.",
    required=True,
)
parser.add_argument(
    "--simplify_tolerance", type=float, help="The tolerance for the simplify operation.", required=False
)
parser.add_argument("--outfile", type=str, help="The path to the output GeoPackage file.", required=True)
args = parser.parse_args()


def main(
    predictions_file: str,
    pseudoproba_thr: float,
    interior_thr: float,
    outfile: str,
    simplify_tolerance: Optional[float],
):
    predictions = gpd.read_file(predictions_file)
    predictions = predictions[predictions.pseudo_probability > pseudoproba_thr]
    predictions_merged = merge_predictions(predictions)
    predictions_merged.geometry = predictions_merged.geometry.apply(lambda x: close_holes(x, interior_thr))

    if simplify_tolerance:
        predictions_merged.geometry = predictions_merged.geometry.simplify(tolerance=simplify_tolerance)

    predictions_merged.to_file(outfile, driver="GPKG")


if __name__ == "__main__":
    main(**vars(args))
