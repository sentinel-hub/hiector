# -*- coding: utf-8 -*-
# File   : dataset.py
# Author : Kai Ao
# Email  : capino627@163.com
# Date   : 2020/12/12 10:44
#
# This file is part of Rotation-Decoupled Detector.
# https://github.com/Capino512/pytorch-rotation-decoupled-detector
# Distributed under MIT License.

import json
from collections import defaultdict
from copy import deepcopy
from typing import Callable, Dict, List, Tuple, Union

import fs
import numpy as np
import pandas as pd
import torch
from fs.osfs import OSFS
from fs_s3fs import S3FS
from shapely.wkt import loads as load_wkt
from torch.utils.data import Dataset

from eolearn.core import EOPatch

from hiector.ssrdd.utils.box.bbox_np import xy42xywha
from hiector.tasks.cropping import CroppingTask
from hiector.utils.grid import preprocess_workflow


class OnTheFlyEOPatchDataset(Dataset):
    NORM_BANDS = ["R", "G", "B", "N"]

    """Dataset of image/labels pairs. Created from a numpy arrays and json files stored on AWS S3 bucket."""

    def __init__(
        self,
        eopatch: EOPatch,
        eopatch_name: str,
        gridding_config: Dict,
        aug: Callable,
        class_names: List[str],
        normalization_factors: pd.DataFrame,
        filesystem: Union[S3FS, OSFS],
    ):
        """Create a dataset from pandas DataFrame."""
        self.eopatch = eopatch
        self.eopatch_name = eopatch_name
        self.gridding_config = gridding_config
        self.aug = aug
        self.filesystem = filesystem
        self.dataset = self.load_dataset()
        self.label2name = dict((label, name) for label, name in enumerate(class_names))
        self.name2label = dict((name, label) for label, name in enumerate(class_names))
        self.normalization_factors = normalization_factors

    def load_dataset(self):
        workflow = preprocess_workflow(self.gridding_config)

        nodes = workflow.get_nodes()
        for node in nodes:
            if isinstance(node.task, CroppingTask):
                self.eopatch = node.task.execute(self.eopatch, eopatch_name=self.eopatch_name)
            else:
                self.eopatch = node.task.execute(self.eopatch)

        per_subgrid = defaultdict(dict)

        grid_feature = self.gridding_config["cropped_grid_feature"]

        for scale in self.gridding_config["scale_sizes"]:
            for image, name in zip(
                self.eopatch.meta_info[f"DATA_STACK_{scale}"],
                self.eopatch.vector_timeless[f"{grid_feature}_{scale}"].NAME.values,
            ):
                per_subgrid[name]["image"] = image
            for name, df in self.eopatch.vector_timeless[f"BBOXES_IN_GRID_{scale}"][["pixel_bbox", "NAME"]].groupby(
                "NAME"
            ):
                payload = [
                    {
                        "label": "building",
                        "geometry": list(load_wkt(polygon).exterior.coords)[:-1]
                        if isinstance(polygon, str)
                        else list(polygon.exterior.coords)[:-1],
                    }
                    for polygon in df.pixel_bbox.values
                ]
                per_subgrid[name]["label"] = payload

        dataset = [(subgrid["image"], subgrid.get("label"), name) for name, subgrid in per_subgrid.items()]
        return dataset

    def normalize(self, image):
        perc5 = self.normalization_factors[self.normalization_factors.statistic == "perc5"][self.NORM_BANDS].values
        perc95 = self.normalization_factors[self.normalization_factors.statistic == "perc95"][self.NORM_BANDS].values
        return (image - perc5) / (perc95 - perc5)

    def transform_objs(self, anno: str) -> dict:
        """Load reference OBBs from json file

        Args:
            path (str): Path to json file with definition of reference OBBs

        Returns:
            dict: Dictionary storing the reference OBBs
        """
        if anno is None:
            return None

        bboxes = [obj["geometry"] for obj in anno]
        labels = [self.name2label[obj["label"]] if self.name2label else obj["label"] for obj in anno]
        objs = {"bboxes": np.array(bboxes, dtype=np.float32), "labels": np.array(labels)}
        return objs

    @staticmethod
    def convert_objs(objs: dict) -> dict:
        """Convert format of coordinates from list x,y vertex positions to centroid, height, width and angle

        Args:
            objs (dict): Dictionary with reference coordinates and target label

        Returns:
            dict: Dictionary with converted coordinates
        """
        target = dict()
        if objs:
            # Limit the angle between -45° and 45° by set flag=2
            target["bboxes"] = torch.from_numpy(np.stack([xy42xywha(bbox, flag=2) for bbox in objs["bboxes"]])).float()
            target["labels"] = torch.from_numpy(objs["labels"]).long()
        return target

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict, dict]:
        """Get dataset item

        Args:
            index (int): Index of item to retrieve, as defined in in put dataframe

        Returns:
            Tuple[torch.Tensor, dict, dict]: Features, labels and info about the item
        """
        features, annos, name = self.dataset[index]
        features = self.normalize(features)
        objs = self.transform_objs(annos)

        info = {"img_path": name, "anno_path": name, "shape": features.shape, "objs": objs}
        if self.aug is not None:
            features, objs = self.aug(features, deepcopy(objs))
        return features, objs, info

    @staticmethod
    def collate(batch: int) -> Tuple[torch.Tensor, list, list]:
        """A method to use to create batches of items

        Args:
            batch (int): Batch size

        Returns:
            Tuple[torch.Tensor, list, list]: Batch of items
        """
        images, targets, infos = [], [], []
        # Ensure data balance when parallelizing
        batch = sorted(batch, key=lambda x: len(x[1]["labels"]) if x[1] else 0)
        for features, labels, info in batch:
            images.append(torch.from_numpy(features).reshape(*features.shape[:2], -1).float())
            targets.append(ObjectDetectionDataset.convert_objs(labels))
            infos.append(info)
        return torch.stack(images).permute(0, 3, 1, 2), targets, infos

    def __len__(self):
        return len(self.dataset)


class ObjectDetectionDataset(Dataset):
    FEATURES_FOLDER = "images"
    FEATURES_EXT = ".npy"
    LABELS_FOLDER = "labels"
    LABELS_EXT = ".json"
    NAME_COL = "NAME"
    N_BBOXES_COL = "N_BBOXES"
    NORM_BANDS = ["R", "G", "B", "N"]

    """Dataset of image/labels pairs. Created from a numpy arrays and json files stored on AWS S3 bucket."""

    def __init__(
        self,
        root: str,
        dataframe: pd.DataFrame,
        aug: Callable,
        class_names: List[str],
        normalization_factors: pd.DataFrame,
        filesystem: Union[S3FS, OSFS],
    ):
        """Create a dataset from pandas DataFrame."""
        self.root = root
        self.dataframe = dataframe
        self.aug = aug
        self.filesystem = filesystem
        self.dataset = self.load_dataset()
        self.label2name = dict((label, name) for label, name in enumerate(class_names))
        self.name2label = dict((name, label) for label, name in enumerate(class_names))
        self.normalization_factors = normalization_factors

    def load_dataset(self):
        dataset = []
        for irow, row in self.dataframe.iterrows():
            name = row[self.NAME_COL]
            n_bboxes = row[self.N_BBOXES_COL]

            features_path = fs.path.join(self.root, self.FEATURES_FOLDER, f"{name}{self.FEATURES_EXT}")

            anno_path = fs.path.join(self.root, self.LABELS_FOLDER, f"{name}{self.LABELS_EXT}")
            anno_path = anno_path if n_bboxes > 0 else None

            dataset.append([features_path, anno_path])
        return dataset

    def normalize(self, image):
        perc5 = self.normalization_factors[self.normalization_factors.statistic == "perc5"][self.NORM_BANDS].values
        perc95 = self.normalization_factors[self.normalization_factors.statistic == "perc95"][self.NORM_BANDS].values
        return (image - perc5) / (perc95 - perc5)

    def load_objs(self, path: str) -> dict:
        """Load reference OBBs from json file

        Args:
            path (str): Path to json file with definition of reference OBBs

        Returns:
            dict: Dictionary storing the reference OBBs
        """
        if path is None or not self.filesystem.exists(path):
            return None

        with self.filesystem.openbin(path, "rb") as fp:
            objs = json.load(fp)

        bboxes = [obj["geometry"] for obj in objs]
        labels = [self.name2label[obj["label"]] if self.name2label else obj["label"] for obj in objs]
        objs = {"bboxes": np.array(bboxes, dtype=np.float32), "labels": np.array(labels)}
        return objs

    @staticmethod
    def convert_objs(objs: dict) -> dict:
        """Convert format of coordinates from list x,y vertex positions to centroid, height, width and angle

        Args:
            objs (dict): Dictionary with reference coordinates and target label

        Returns:
            dict: Dictionary with converted coordinates
        """
        target = dict()
        if objs:
            # Limit the angle between -45° and 45° by set flag=2
            target["bboxes"] = torch.from_numpy(np.stack([xy42xywha(bbox, flag=2) for bbox in objs["bboxes"]])).float()
            target["labels"] = torch.from_numpy(objs["labels"]).long()
        return target

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict, dict]:
        """Get dataset item

        Args:
            index (int): Index of item to retrieve, as defined in in put dataframe

        Returns:
            Tuple[torch.Tensor, dict, dict]: Features, labels and info about the item
        """
        features_path, anno_path = self.dataset[index]

        with self.filesystem.openbin(features_path, "rb") as fp:
            features = np.load(fp)

        features = self.normalize(features)
        objs = self.load_objs(anno_path)

        info = {"img_path": features_path, "anno_path": anno_path, "shape": features.shape, "objs": objs}
        if self.aug is not None:
            features, objs = self.aug(features, deepcopy(objs))
        return features, objs, info

    @staticmethod
    def collate(batch: int) -> Tuple[torch.Tensor, list, list]:
        """A method to use to create batches of items

        Args:
            batch (int): Batch size

        Returns:
            Tuple[torch.Tensor, list, list]: Batch of items
        """
        images, targets, infos = [], [], []
        # Ensure data balance when parallelizing
        batch = sorted(batch, key=lambda x: len(x[1]["labels"]) if x[1] else 0)
        for features, labels, info in batch:
            images.append(torch.from_numpy(features).reshape(*features.shape[:2], -1).float())
            targets.append(ObjectDetectionDataset.convert_objs(labels))
            infos.append(info)
        return torch.stack(images).permute(0, 3, 1, 2), targets, infos

    def __len__(self):
        return len(self.dataframe)
