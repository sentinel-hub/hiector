"""
Vector utilities
"""
import os
from typing import List, Optional

import fiona
from shapely.wkt import loads


def export_geopackage(
    eopatch,
    geopackage_path,
    feature,
    geometry_column: str = "geometry",
    columns: Optional[List[str]] = None,
):
    """A utility function for exporting

    :param eopatch: EOPatch to save
    :param geopackage_path: Output path where Geopackage will be written.
    :param feature: A vector feature from EOPatches that will be exported to Geopackage
    :param geometry_column: Name of a column that will be taken as a geometry column.
    :param columns: Columns from dataframe that will be written into Geopackage besides geometry column. By default
        all columns will be taken.

    Note: in the future it could be implemented as an eo-learn task, the main problem is that writing has to be
    consecutive.
    """
    existing_layers = fiona.listlayers(geopackage_path) if os.path.exists(geopackage_path) else []

    gdf = eopatch[feature]
    layer_name = f"{feature[1]}_{gdf.crs.to_epsg()}"
    mode = "a" if layer_name in existing_layers else "w"

    if not len(gdf.index):
        return

    # Only one geometry column can be saved to a Geopackage
    if isinstance(gdf[geometry_column].iloc[0], str):
        gdf[geometry_column] = gdf[geometry_column].apply(loads)

    gdf = gdf.set_geometry(geometry_column)
    if columns is not None:
        gdf = gdf.filter(columns + [geometry_column], axis=1)

    gdf.to_file(geopackage_path, mode=mode, layer=layer_name, driver="GPKG", encoding="utf-8")
