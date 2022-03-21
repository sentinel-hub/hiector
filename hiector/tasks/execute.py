import os
from collections import defaultdict
from typing import Union

import boto3
import fiona
import fs
import fs_s3fs
import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import tqdm
from fiona.session import AWSSession
from fs.osfs import OSFS
from torch.utils.data import DataLoader

from eolearn.core import EOPatch, EOTask
from eolearn.core.utils.fs import join_path
from sentinelhub import CRS

from ..ssrdd.data.aug import ops
from ..ssrdd.data.aug.compose import Compose
from ..ssrdd.data.dataset.dataset import ObjectDetectionDataset, OnTheFlyEOPatchDataset

# The following import is required because backbone gets selected dynamically with a config parameter.
from ..ssrdd.model.backbone import darknet, resnet
from ..ssrdd.model.rdd import RDD
from ..ssrdd.utils.box.bbox_np import xy42xywha, xywha2xy4
from ..ssrdd.utils.box.rbbox_np import rbbox_batched_nms
from ..utils.aws_utils import get_filesystem
from ..utils.metrics import get_bbox


class PredictEOPatch(EOTask):
    def __init__(self, config, on_the_fly: bool):
        self.config = config
        self.data_config = self.config["datasources"]["evaluate"]
        self.on_the_fly = on_the_fly

    @staticmethod
    def _get_filesystem(root: str, s3_bucket_name: str, s3_profile_name: str) -> Union[OSFS, fs_s3fs.S3FS]:
        """Get either local or S3 filesystem

        Args:
            root (str): Path to data directory holding the metadata file and images/labels folder
            s3_bucket_name (str): Name of bucket or None
            s3_profile_name (str): Name of S3 profile to use or None

        Returns:
            Union[OSFS, fs_s3fs.S3FS]: Filesystem
        """
        filesystem = OSFS("/")
        if s3_bucket_name is not None and s3_profile_name is not None:
            filesystem = get_filesystem(s3_bucket_name, s3_profile_name)
        return filesystem

    def _get_filtered_dataframe(self, bbox):
        filepath = fs.path.join(self.data_config["data_dir"], self.data_config["metadata_filename"])
        aws_session = AWSSession(boto3.session.Session(profile_name=self.config["s3_profile_name"]))

        with fiona.Env(aws_session):
            awspath = join_path(f's3://{self.config["s3_bucket_name"]}', filepath)
            with fiona.open(awspath) as features:
                feature_iter = (
                    features.filter(bbox=bbox.transform_bounds(CRS("4326")).geometry.bounds) if bbox else features
                )
                return gpd.GeoDataFrame.from_features(
                    feature_iter, columns=list(features.schema["properties"]) + ["geometry"], crs=bbox.crs.pyproj_crs()
                )

    def _create_otf_dataset(
        self,
        eopatch_name,
        gridding_config,
        normalization_factors,
        aug: Compose,
        class_names: list,
    ) -> ObjectDetectionDataset:
        filesystem = self._get_filesystem(
            self.data_config["data_dir"], self.config["s3_bucket_name"], self.config["s3_profile_name"]
        )

        eopatch_path = os.path.join(self.config["datasources"]["evaluate"]["eopatches_dir"], eopatch_name)
        eopatch = EOPatch.load(eopatch_path, filesystem=filesystem, lazy_loading=True)

        dataset = OnTheFlyEOPatchDataset(
            eopatch=eopatch,
            eopatch_name=eopatch_name,
            gridding_config=gridding_config,
            normalization_factors=normalization_factors,
            aug=aug,
            class_names=class_names,
            filesystem=filesystem,
        )
        return dataset

    def _create_fromfiles_dataset(
        self,
        dataframe: pd.DataFrame,
        normalization_factors,
        aug: Compose,
        class_names: list,
    ) -> ObjectDetectionDataset:
        filesystem = self._get_filesystem(
            self.data_config["data_dir"], self.config["s3_bucket_name"], self.config["s3_profile_name"]
        )
        dataset = ObjectDetectionDataset(
            root=self.data_config["data_dir"],
            dataframe=dataframe,
            normalization_factors=normalization_factors,
            aug=aug,
            class_names=class_names,
            filesystem=filesystem,
        )
        return dataset

    @torch.no_grad()
    def execute(self, eopatch_name: str):
        """Function to evaluate a trained SSRDD model

        Args:
            config (dict): Configuration file specifying the parameters required to evaluate the SSRDD model
        """
        filesystem = self._get_filesystem(
            self.data_config["data_dir"], self.config["s3_bucket_name"], self.config["s3_profile_name"]
        )

        checkpoint = self.config["checkpoint"]
        if checkpoint is None:
            dir_weight = os.path.join(self.config["aws_model_dir"], "weight")
            indexes = [int(os.path.splitext(path)[0]) for path in filesystem.listdir(dir_weight)]
            current_step = max(indexes)
            checkpoint = os.path.join(dir_weight, "%d.pth" % current_step)

        backbone = eval(self.config["backbone"])

        image_size = self.config["image_size"]
        batch_size = self.config["batch_size"]
        num_workers = self.config["num_workers"]
        class_names = self.config["class_names"]

        aug = Compose([ops.PadSquare(), ops.Resize(image_size)])

        assert isinstance(self.config["datasources"]["evaluate"], dict)

        with filesystem.openbin(
            os.path.join(self.data_config["data_dir"], self.data_config["normalization"]["filename"])
        ) as file_handle:
            normalization_factors = pd.read_csv(file_handle)
            normalization_factors = normalization_factors[
                normalization_factors.modality == self.data_config["normalization"]["modality"]
            ]

        if self.on_the_fly:
            dataset = self._create_otf_dataset(
                eopatch_name, self.config["gridding_config"], normalization_factors, aug, class_names
            )
        else:
            query_test = f'({self.data_config["query_test"]}) & (EOPATCH_NAME == "{eopatch_name}")'
            bbox = get_bbox(eopatch_name, self.data_config["eopatches_dir"], filesystem, self.data_config["resolution"])
            dataframe = self._get_filtered_dataframe(bbox["bbox"])
            dataframe = dataframe.query(query_test)
            dataset = self._create_fromfiles_dataset(dataframe, normalization_factors, aug, class_names)

        loader = DataLoader(
            dataset, batch_size, num_workers=num_workers, pin_memory=True, collate_fn=ObjectDetectionDataset.collate
        )

        num_classes = len(class_names)

        strides = self.config["prior_box"]["strides"]
        sizes = self.config["prior_box"]["sizes"]
        aspects = self.config["prior_box"]["aspects"]
        scales = self.config["prior_box"]["scales"]
        prior_box = {
            "strides": strides,
            "sizes": sizes * len(strides) if len(sizes) == 1 else sizes,
            "aspects": [aspects] * len(strides),
            "scales": [scales] * len(strides),
            "old_version": self.config["old_version"],
        }

        conf_thresh = self.config["conf_thresh"]
        conf_thresh_2 = self.config["conf_thresh_2"]
        nms_thresh = self.config["nms_thresh"]

        model_cfg = {
            "prior_box": prior_box,
            "num_classes": num_classes,
            "extra": self.config["extra"],
            "conf_thresh": conf_thresh,
            "conf_thresh_2": conf_thresh_2,
            "nms_thresh": nms_thresh,
        }

        model = RDD(
            backbone(
                fetch_feature=True,
                head_stride_1=self.config["head_stride_1"],
                head_stride_2=self.config["head_stride_2"],
            ),
            model_cfg,
        )
        model.build_pipe(shape=[2, 4, image_size, image_size])
        model.restore(filesystem.openbin(checkpoint))
        model.eval()

        ret_raw = defaultdict(list)
        for images, _, infos in tqdm.tqdm(loader):
            dets = model(images)
            for (det, info) in zip(dets, infos):
                if det:
                    bboxes, scores, labels = det
                    bboxes = bboxes.cpu().numpy()
                    scores = scores.cpu().numpy()
                    labels = labels.cpu().numpy()
                    if len(bboxes) > 0:
                        scores = [scores] if bboxes.ndim == 1 else scores
                        labels = [labels] if bboxes.ndim == 1 else labels
                        bboxes = bboxes.reshape(1, 5) if bboxes.ndim == 1 else bboxes
                        try:
                            eop, xeop, yeop, x, y, hw = os.path.splitext(os.path.basename(info["img_path"]))[0].split(
                                "-"
                            )
                            fname = "-".join([eop, xeop, yeop])
                        except ValueError:
                            split = os.path.splitext(os.path.basename(info["img_path"]))[0].split("-")
                            x, y, hw = split[-3:]
                            fname = "-".join(split[:-3])
                        x, y, w, h = int(x), int(y), int(hw), int(hw)
                        long_edge = max(w, h)
                        pad_x, pad_y = (long_edge - w) // 2, (long_edge - h) // 2
                        bboxes = np.stack([xywha2xy4(bbox) for bbox in bboxes])
                        bboxes *= long_edge / image_size
                        bboxes -= [pad_x, pad_y]
                        bboxes += [x, y]
                        bboxes = np.stack([xy42xywha(bbox) for bbox in bboxes])
                        ret_raw[fname].append([bboxes, scores, labels])

        ret_save = defaultdict(list)
        for fname, dets in ret_raw.items():
            bboxes, scores, labels = zip(*dets)
            bboxes = np.concatenate(list(bboxes))
            scores = np.concatenate(list(scores))
            labels = np.concatenate(list(labels))
            keeps = rbbox_batched_nms(bboxes, scores, labels, nms_thresh)

            for bbox, score, label in zip(bboxes[keeps], scores[keeps], labels[keeps]):
                bbox = xywha2xy4(bbox).ravel()
                line = "%s %.12f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f" % (fname, score, *bbox)
                ret_save[dataset.label2name[label]].append(line)

        if "aws_dota_dir" in self.config:
            fs = self._get_filesystem(
                self.data_config["data_dir"], self.config["s3_bucket_name"], self.config["s3_profile_name"]
            )
            dota_dir = self.config["aws_dota_dir"]
        else:
            fs = OSFS(self.config["results_dir"])
            dota_dir = "."

        fs.makedirs(dota_dir, recreate=True)
        for name, dets in ret_save.items():
            with fs.open(os.path.join(dota_dir, "Task%d_%s_%s.txt" % (1, name, eopatch_name)), "wt") as f:
                f.write("\n".join(dets))
