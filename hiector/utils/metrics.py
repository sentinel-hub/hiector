"""Utilities for dealing with metrics and dota to geodataframe format conversion"""
import os
from typing import Dict

import fs
from fs_s3fs import S3FS
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from eolearn.core import EOPatch
from sentinelhub import CRS
from sentinelhub.geo_utils import pixel_to_utm

from .aws_utils import LocalFile


def convert_dota(
    dota_dir: str, eop_dir: str, filename: str, filesystem: S3FS, resolution: int
) -> Dict[str, gpd.GeoDataFrame]:
    eopname = filename.split("_")[2][:-4]
    eopdata = {eopname: get_bbox(eopname, eop_dir, filesystem, resolution)}
    with LocalFile(fs.path.join(dota_dir, filename), mode="r", filesystem=filesystem) as local_file:
        return dota_results_to_gdf(local_file.path, eopdata, underscore=False)


def dota_results_to_gdf(detections_file: str, eop_data: dict, underscore: bool = False) -> Dict[str, gpd.GeoDataFrame]:
    """Convert DOTA-style predictions to GeoDataFrame

    :param detections_file: detection file output by the model.
            Format is assumed to be `<eop-name> <proba> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4> `.
    :param eop_data: dict holding info about the eopatch, such as name and bbox
    :param underscore: whether the eopatch name uses '_' or '-'
    """
    with open(detections_file, "r") as res:
        lines = res.read().splitlines()
        lines = [line.split(" ") for line in lines]

    df = pd.DataFrame(
        [
            {"eopatch": txt[0], "pseudo_probability": float(txt[1]), "coords": [float(d) for d in txt[2:]]}
            for txt in lines
        ]
    )
    df["CRS_EPSG"] = df.eopatch.apply(lambda x: eop_data.get(x)["bbox"].crs.epsg)

    preds = {}
    for epsg in df.CRS_EPSG.unique():
        gdf = gpd.GeoDataFrame(df[df["CRS_EPSG"] == epsg])
        gdf["geometry"] = gdf.apply(
            lambda row: to_crs(row.coords, eop_data[row.eopatch if underscore else row.eopatch.replace("_", "-")]),
            axis=1,
        )
        gdf = gdf.set_crs(crs=CRS(str(epsg)).pyproj_crs(), allow_override=True)
        gdf = gdf.drop(columns="coords")
        preds[f"epsg:{epsg}"] = gdf

    return preds


def get_bbox(eopatch: str, aws_eopatches_path: str, filesystem: fs.base.FS, resolution: float):
    """Get bounding box and transform associated to input eopatch

    :param eopatch: name of eopatch
    :param aws_eopatches_path: AWS path to eopatches
    :param filesystem: S3FS filesystem for access to AWS s3 bucket
    :param resolution: spatial resolution of pixel
    """
    if not filesystem.exists(f"{aws_eopatches_path}/{eopatch}"):
        filesystem.makedir(f"{aws_eopatches_path}/{eopatch}")

    bbox = EOPatch.load(f"{aws_eopatches_path}/{eopatch}", lazy_loading=True, filesystem=filesystem).bbox
    transform = bbox.get_transform_vector(resolution, resolution)
    return {"bbox": bbox, "transform": transform}


def to_crs(coords: list, eop_bbox: dict) -> Polygon:
    """Function to convert image coordinates (i.e. row/col) to UTM spatial coordinates

    :param coords: list of point coordinates [x1, y1, x2, y2,..] to be converted
    :param eop_bbox: dictionary holding bounding box coordinates and spatial transform
    """
    coord_pairs = list(zip(coords[::2], coords[1::2]))
    transform = eop_bbox["transform"]
    coord_pairs_trans = [pixel_to_utm(c[1], c[0], transform) for c in coord_pairs]
    return Polygon(coord_pairs_trans)


def iou_single(geom: Polygon, target_geoms: gpd.GeoSeries) -> pd.Series:
    """Calculates IOU between a single geometry and target geometries in GeoSeries.

    Args:
        geom (Polygon): Geometry.
        target_geoms (gpd.GeoSeries): Geometries over which to calculate IOU with "geom"

    Returns:
        pd.Series: Series containing IOUs scores for each geom, target_geom pair.
    """
    _intersection_areas = target_geoms.intersection(geom).area
    _union_areas = target_geoms.union(geom).area
    return _intersection_areas / _union_areas


def intersection_over_smallest(geom: Polygon, target_geoms: gpd.GeoSeries) -> gpd.GeoSeries:
    """Calculates (A ∩ B) / B between geom (A) and each target geom (B).

    Args:
        geom (Polygon): Geometry.
        target_geoms (gpd.GeoSeries): Geometries over which to calculate intersection over smallest.

    Returns:
        gpd.GeoSeries: Series containing intersection over smallest for each geom, target_geom pair.
    """
    _intersection_areas = target_geoms.intersection(geom).area
    return _intersection_areas / target_geoms.area


def iou(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, drop_duplicates: bool = True) -> gpd.GeoDataFrame:
    """Compute Intersection over Union between the two input dataframes

    This function returns a GeoDataFrame containing all the intersections between the two inputs with calculation of
    their IoU

    :param gdf1: first input geo data frame
    :param gdf2: second input geo data frame
    :param drop_duplicates: whether to drop duplicates or not. Default is `True`
    """
    df1 = gdf1.copy()
    df2 = gdf2.copy()

    if drop_duplicates:
        df1 = df1.drop_duplicates()
        df2 = df2.drop_duplicates()

    df1["geom"] = df1.geometry
    df2["geom"] = df2.geometry

    df1["iouidx"] = range(len(df1))
    df2["iouidx"] = range(len(df2))

    joined = gpd.sjoin(df1, df2, how="inner", lsuffix="df1", rsuffix="df2")
    if len(joined) == 0:
        return gpd.GeoDataFrame(data=[], geometry=[], crs=df1.crs)

    joined["geom_intersection"] = joined.apply(lambda r: r.geom_df1.intersection(r.geom_df2), axis=1)
    joined["geom_union"] = joined.apply(lambda r: r.geom_df1.union(r.geom_df2), axis=1)

    joined["area_intersection"] = joined.geom_intersection.apply(lambda g: g.area)
    joined["area_union"] = joined.geom_union.apply(lambda g: g.area)

    joined["iou"] = joined.area_intersection / joined.area_union

    return joined


def get_error_matrix(
    gdf_gt: gpd.GeoDataFrame,
    gdf_pr: gpd.GeoDataFrame,
    proba_thr: float = 0.0,
    iou_thr: float = 0.5,
    proba_col: str = "pseudo_probability",
    return_dfs: bool = False,
) -> dict:
    """Calculate the error matrix between reference and predicted detections.

    Number and geometries for TP/FP/FN values are returned. These values are computed as follows:

     * TPs are _predicted_ geometries that overlap with reference geometries for more than `iou_thr`
     * FNs are _reference_ geometries that don't overlap with any predicted geometry
     * FPs are _predicted_ geometries that are not TPs

    :param gdf_gt: geodataframe with reference geometries
    :param gdf_pr: geodataframe with predicted geometries
    :param proba_thr: keep predicted geometries with probability equal or greater than this value. Default is `0.0`
    :param iou_thr: consider true detections intersections with IoU equal or greater than this value. Default is `0.5`
    :param proba_col: name of columns in `gdf_pr` holding the pseudo-probability values. Default is `pseudo_probability`
    :param return_dfs: whether to return also dataframes or only counts. Default is `False`
    """
    assert proba_col in gdf_pr.columns

    gdf_gt_ = gdf_gt.copy()
    gdf_gt_ = gdf_gt_.drop_duplicates()

    gdf_pr_ = gdf_pr[gdf_pr[proba_col] >= proba_thr].copy()
    gdf_pr_ = gdf_pr_.drop_duplicates()

    iou_df_ = iou(gdf_gt_, gdf_pr_, drop_duplicates=False)

    tp = iou_df_[iou_df_.iou >= iou_thr]
    tp = tp.sort_values("iou", ascending=False).groupby("iouidx_df1", as_index=False).first()
    # now tp has the geometries of the GT, so we can compute the FN

    gdf_gt_["idx"] = range(len(gdf_gt_))
    fn = gdf_gt_[~gdf_gt_.idx.isin(tp.iouidx_df1)]
    number_fn = len(fn)

    # we compute the FP as the indices that are in TP but not in entire prediction
    gdf_pr_["idx"] = range(len(gdf_pr_))
    fp = gdf_pr_[~gdf_pr_.idx.isin(tp.iouidx_df2)]
    number_fp = len(fp)

    # the TPs can be expressed both in terms of reference or prediction geometries
    # in this case we consider the predicted geometries, so that TP + FP is equal to the number of predicted polys
    tp["geometry"] = tp["geom_df2"]
    number_tp = len(tp[["geometry"]].drop_duplicates())

    assert number_tp + number_fp == len(gdf_pr_)

    return {
        "pseudo_thr": proba_thr,
        "iou_thr": iou_thr,
        "TP": number_tp,
        "FN": number_fn,
        "FP": number_fp,
        "GT": len(gdf_gt_),
        "TP_df": tp if return_dfs else None,
        "FN_df": fn if return_dfs else None,
        "FP_df": fp if return_dfs else None,
    }


# taken from https://github.com/Cartucho/mAP/blob/master/main.py
def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1

    ap = 0.0
    for i in i_list:
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap, mrec, mpre


# the table used to compute TPs/FPs and precision/recall curves an be constructed from
# * the sjoin used to compute the IoU
# * the addition of false positives taken as the difference between the sjoin above and all predictions
def get_ap(
    gdf_gt: gpd.GeoDataFrame,
    gdf_pr: gpd.GeoDataFrame,
    proba_thr: float = 0.0,
    iou_thr: float = 0.5,
    proba_col: str = "pseudo_probability",
):
    """Compute average precision as area under the curve of the recall/precision curve.

    Average precision is computed as follows:
     * compute IoU between reference and predicted detections
     * compute true positives as detections that overlap with reference for more than `iou_thr`. If
    multiple geometries are found, keep the one with largest probability value
     * compute false positives as predictions not overlapping with any reference geometry
     * concatenate TPs and FPs and order in descending order by probability
     * compute cumulative precision/recall for each predicted geometry
     * compute AP as AUC or precision/recall curve


    :param gdf_gt: geodataframe with reference geometries
    :param gdf_pr: geodataframe with predicted geometries
    :param proba_thr: keep predicted geometries with probability equal or greater than this value. Default is `0.0`
    :param iou_thr: consider true detections intersections with IoU equal or greater than this value. Default is `0.5`
    :param proba_col: name of columns in `gdf_pr` holding the pseudo-probability values. Default is `pseudo_probability`
    """
    assert proba_col in gdf_pr.columns

    gdf_gt_ = gdf_gt.copy()
    gdf_gt_ = gdf_gt_.drop_duplicates()

    gdf_pr_ = gdf_pr[gdf_pr[proba_col] >= proba_thr].copy()
    gdf_pr_ = gdf_pr_.drop_duplicates()

    iou_df_ = iou(gdf_gt_, gdf_pr_, drop_duplicates=False)
    if iou_df_.empty:
        return 0, 0, 0
    iou_df_ = iou_df_.drop(columns=["index_df2", "geom_intersection", "geom_union", "area_intersection", "area_union"])
    iou_df_ = iou_df_.reset_index(drop=True)

    iou_df_["correct"] = iou_df_["iou"] >= iou_thr

    # For each GT poly there has to be only one prediction counted as TP
    # group by GT IDs
    iou_gdf_ = iou_df_.groupby("iouidx_df1")
    # find GT IDs with more than one overlap with predictions
    gtids = iou_gdf_[["correct"]].apply(sum).query("correct > 1")
    # keep only the first prediction
    for gtid in gtids.index.values:
        dfids = iou_df_[(iou_df_["iouidx_df1"] == gtid) & (iou_df_["correct"] == True)].index
        for dfid in dfids[1:]:
            iou_df_.at[dfid, "correct"] = False

    # append FPs, which are predictions that don't overlap at all with the GT, i.e. their index is not in the iou
    gdf_pr_["idx"] = range(len(gdf_pr_))
    fps = gdf_pr_[~gdf_pr_.idx.isin(iou_df_.iouidx_df2.unique())]
    fps = fps.assign(correct=False)

    ap_df = pd.concat([iou_df_[[proba_col, "correct"]], fps[[proba_col, "correct"]]], axis=0)

    ap_df = ap_df.sort_values(by=proba_col, axis=0, ascending=False)
    ap_df = ap_df.reset_index(drop=True)

    assert ap_df.correct.sum() < len(gdf_gt_)

    ap_df["cumsum"] = ap_df.correct.cumsum()
    ap_df["precision"] = ap_df["cumsum"].values / (ap_df.index.values + 1)
    ap_df["recall"] = ap_df["cumsum"] / len(gdf_gt_)

    ap, mrec, mpre = voc_ap(list(ap_df.recall.values), list(ap_df.precision.values))

    return ap, mrec, mpre
