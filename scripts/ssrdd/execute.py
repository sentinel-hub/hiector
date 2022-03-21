import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Union

import fs
import fs_s3fs
import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import tqdm
import wandb
from fs.copy import copy_dir
from fs.osfs import OSFS
from torch import optim
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from hiector.ssrdd.data.aug import ops
from hiector.ssrdd.data.aug.compose import Compose
from hiector.ssrdd.data.dataset import ObjectDetectionDataset

# The following import is required because backbone gets selected dynamically with a config parameter.
from hiector.ssrdd.model.backbone import darknet, resnet
from hiector.ssrdd.model.rdd import RDD
from hiector.ssrdd.utils.adjust_lr import adjust_lr_multi_step
from hiector.ssrdd.utils.box.bbox_np import xy42xywha, xywha2xy4
from hiector.ssrdd.utils.box.rbbox_np import rbbox_batched_nms
from hiector.ssrdd.utils.parallel import CustomDetDataParallel, convert_model
from hiector.utils.aws_utils import get_filesystem
from hiector.utils.metrics import dota_results_to_gdf, get_bbox

stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [stdout_handler]
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", handlers=handlers
)
LOGGER = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description="Execute training and evaluation using the SSRDD model.\n")

parser.add_argument("--config", type=str, help="Path to config file with execution parameters", required=True)
parser.add_argument(
    "--action", type=str, help="Specify action to execute. Only 'train' or 'evaluate' are supported", required=True
)
parser.add_argument(
    "--wandb_key",
    type=str,
    help=(
        "Optional wandb key to be used for tracking the training losses. This key is used to run `wandb login"
        " WANDB_KEY`"
    ),
    required=False,
    default=None,
)
args = parser.parse_args()


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


def create_dataset(
    data_dir: str,
    s3_bucket_name: str,
    s3_profile_name: str,
    metadata_filename: str,
    normalization_file: str,
    normalization_modality: str,
    query: str,
    aug_train: Compose,
    class_names: list,
):
    filesystem = _get_filesystem(data_dir, s3_bucket_name, s3_profile_name)
    with filesystem.openbin(fs.path.join(data_dir, metadata_filename), "rb") as file_handle:
        dataframe = gpd.read_file(file_handle)

    with filesystem.openbin(os.path.join(data_dir, normalization_file)) as file_handle:
        normalization_factors = pd.read_csv(file_handle)
    normalization_factors = normalization_factors[normalization_factors.modality == normalization_modality]

    dataframe = dataframe.query(query)
    dataset = ObjectDetectionDataset(
        root=data_dir,
        dataframe=dataframe,
        aug=aug_train,
        class_names=class_names,
        normalization_factors=normalization_factors,
        filesystem=filesystem,
    )
    return dataset


def train(config: dict, wandb_key: str = None):
    """Function to train a SSRDD model

    Args:
        config (dict): Configuration file specifying the parameters required to evaluate the SSRDD model
        wandb_key (str, optional): Login wandb key. Defaults to None.
    """
    use_wandb = wandb_key is not None
    if use_wandb:
        os.system(f"wandb login {wandb_key}")

    backbone = eval(config["backbone"])
    save_interval = config["save_interval"]

    dir_weight = os.path.join(config["model_dir"], "weight")
    dir_log = os.path.join(config["model_dir"], "log")
    os.makedirs(dir_weight, exist_ok=True)
    writer = SummaryWriter(dir_log)

    indexes = [int(os.path.splitext(path)[0]) for path in os.listdir(dir_weight)]
    current_step = max(indexes) if indexes else 0

    image_size = config["image_size"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    class_names = config["class_names"]

    aug_train = Compose(
        [
            ops.RandomHFlip(),
            ops.RandomVFlip(),
            ops.RandomRotate90(),
            ops.ResizeJitter([0.8, 1.2]),
            ops.PadSquare(),
            ops.Resize(image_size),
        ]
    )
    aug_val = Compose(
        [
            ops.PadSquare(),
            ops.Resize(image_size),
        ]
    )

    lr = config["lr"]
    max_step = config["max_step"]
    save_interval = config["save_interval"]
    lr_cfg = [[max_step / 3, lr], [2 * max_step / 3, lr / 10], [max_step, lr / 50]]
    warm_up = [max_step / 8, lr / 50, lr]

    train_datasets = [
        create_dataset(
            datasource["data_dir"],
            config["s3_bucket_name"],
            config["s3_profile_name"],
            datasource["metadata_filename"],
            datasource["normalization"]["filename"],
            datasource["normalization"]["modality"],
            datasource["query_train"],
            aug_train,
            class_names,
        )
        for datasource in config["datasources"]["train"]
    ]

    ds_train = ConcatDataset(train_datasets)
    loader_train = DataLoader(
        ds_train,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=ObjectDetectionDataset.collate,
    )

    cval_datasets = [
        create_dataset(
            datasource["data_dir"],
            config["s3_bucket_name"],
            config["s3_profile_name"],
            datasource["metadata_filename"],
            datasource["normalization"]["filename"],
            datasource["normalization"]["modality"],
            datasource["query_cval"],
            aug_val,
            class_names,
        )
        for datasource in config["datasources"]["train"]
    ]

    ds_cval = ConcatDataset(cval_datasets)
    loader_cval = DataLoader(
        ds_cval,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=ObjectDetectionDataset.collate,
    )

    num_classes = len(class_names)

    strides = config["prior_box"]["strides"]
    sizes = config["prior_box"]["sizes"]
    aspects = config["prior_box"]["aspects"]
    scales = config["prior_box"]["scales"]
    prior_box = {
        "strides": strides,
        "sizes": sizes * len(strides) if len(sizes) == 1 else sizes,
        "aspects": [aspects] * len(strides),
        "scales": [scales] * len(strides),
    }

    model_cfg = {
        "prior_box": prior_box,
        "num_classes": num_classes,
        "extra": config["extra"],
        "image_size": image_size,
        "lr": lr,
        "batch_size": batch_size,
        "max_step": max_step,
        "lr_cfg": lr_cfg,
        "warm_up": warm_up,
        "modality": "-".join([datasource["modality"] for datasource in config["datasources"]["train"]]),
    }

    if use_wandb:
        wandb.init(project=f"qp3", entity="eoresearch", config=model_cfg)

    model = RDD(
        backbone(fetch_feature=True, head_stride_1=config["head_stride_1"], head_stride_2=config["head_stride_2"]),
        model_cfg,
    )
    model.build_pipe(shape=[2, 4, image_size, image_size])
    if current_step:
        model.restore(os.path.join(dir_weight, "%d.pth" % current_step))
    else:
        model.init()
    if len(device_ids) > 1:
        model = convert_model(model)
        model = CustomDetDataParallel(model, device_ids)
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    training = True
    while training and current_step < max_step:
        tqdm_train_loader = tqdm.tqdm(loader_train)
        for images, targets, infos in tqdm_train_loader:
            current_step += 1
            adjust_lr_multi_step(optimizer, current_step, lr_cfg, warm_up)

            images = images.cuda()
            losses = model(images, targets)
            loss = sum(losses.values())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            for key, val in list(losses.items()):
                losses[key] = val.item()
                writer.add_scalar(key, val, global_step=current_step)

            if use_wandb:
                wandb.log(dict(train_losses=dict(losses)), step=current_step)

            writer.flush()
            tqdm_train_loader.set_postfix(losses)
            tqdm_train_loader.set_description(f"<{current_step}/{max_step}>")

            if current_step % save_interval == 0:
                save_path = os.path.join(dir_weight, "%d.pth" % current_step)
                state_dict = model.state_dict() if len(device_ids) == 1 else model.module.state_dict()
                torch.save(state_dict, save_path)
                cache_file = os.path.join(dir_weight, "%d.pth" % (current_step - save_interval))
                if os.path.exists(cache_file):
                    os.remove(cache_file)

            if current_step >= max_step:
                training = False
                writer.close()
                break

        tqdm_val_loader = None if loader_cval is None else tqdm.tqdm(loader_cval)
        if tqdm_val_loader:
            val_losses = dict(loss_cls=0.0, loss_loc=0.0)
            with torch.no_grad():
                for images, targets, infos in tqdm_val_loader:
                    images = images.cuda()
                    losses = model(images, targets)

                    for key, val in list(losses.items()):
                        val_losses[key] += val.item()

                    tqdm_val_loader.set_postfix(losses)

                for key, val in list(val_losses.items()):
                    val_losses[key] /= len(tqdm_val_loader)

                if use_wandb:
                    wandb.log(dict(val_losses=dict(val_losses)), step=current_step)
    filesystem = get_filesystem(config["s3_bucket_name"], config["s3_profile_name"])
    filesystem.makedirs(config["aws_model_dir"], recreate=True)
    copy_dir(config["model_dir"], ".", filesystem, config["aws_model_dir"])


@torch.no_grad()
def evaluate(config: dict):
    """Function to evaluate a trained SSRDD model

    Args:
        config (dict): Configuration file specifying the parameters required to evaluate the SSRDD model
    """
    checkpoint = config["checkpoint"]
    if checkpoint is None:
        dir_weight = os.path.join(config["model_dir"], "weight")
        indexes = [int(os.path.splitext(path)[0]) for path in os.listdir(dir_weight)]
        current_step = max(indexes)
        checkpoint = os.path.join(dir_weight, "%d.pth" % current_step)

    backbone = eval(config["backbone"])

    image_size = config["image_size"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    class_names = config["class_names"]

    aug = Compose([ops.PadSquare(), ops.Resize(image_size)])

    assert isinstance(config["datasources"]["evaluate"], dict)
    data_config = config["datasources"]["evaluate"]

    dataset = create_dataset(
        data_config["data_dir"],
        config["s3_bucket_name"],
        config["s3_profile_name"],
        data_config["metadata_filename"],
        data_config["normalization"]["filename"],
        data_config["normalization"]["modality"],
        data_config["query_test"],
        aug,
        class_names,
    )

    loader = DataLoader(
        dataset, batch_size, num_workers=num_workers, pin_memory=True, collate_fn=ObjectDetectionDataset.collate
    )

    num_classes = len(class_names)

    strides = config["prior_box"]["strides"]
    sizes = config["prior_box"]["sizes"]
    aspects = config["prior_box"]["aspects"]
    scales = config["prior_box"]["scales"]
    prior_box = {
        "strides": strides,
        "sizes": sizes * len(strides) if len(sizes) == 1 else sizes,
        "aspects": [aspects] * len(strides),
        "scales": [scales] * len(strides),
        "old_version": config["old_version"],
    }

    conf_thresh = config["conf_thresh"]
    conf_thresh_2 = config["conf_thresh_2"]
    nms_thresh = config["nms_thresh"]

    model_cfg = {
        "prior_box": prior_box,
        "num_classes": num_classes,
        "extra": config["extra"],
        "conf_thresh": conf_thresh,
        "conf_thresh_2": conf_thresh_2,
        "nms_thresh": nms_thresh,
    }

    model = RDD(
        backbone(fetch_feature=True, head_stride_1=config["head_stride_1"], head_stride_2=config["head_stride_2"]),
        model_cfg,
    )
    model.build_pipe(shape=[2, 4, image_size, image_size])
    model.restore(checkpoint)
    if len(device_ids) > 1:
        model = CustomDetDataParallel(model, device_ids)
    model.cuda()
    model.eval()

    ret_raw = defaultdict(list)
    for images, targets, infos in tqdm.tqdm(loader):
        images = images.cuda()
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
                        eop, xeop, yeop, x, y, hw = os.path.splitext(os.path.basename(info["img_path"]))[0].split("-")
                        fname = "-".join([eop, xeop, yeop])
                    except ValueError as e:
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

    LOGGER.info("merging results...")
    ret = []

    for fname, dets in ret_raw.items():
        bboxes, scores, labels = zip(*dets)
        bboxes = np.concatenate(list(bboxes))
        scores = np.concatenate(list(scores))
        labels = np.concatenate(list(labels))
        keeps = rbbox_batched_nms(bboxes, scores, labels, nms_thresh)
        ret.append([fname, [bboxes[keeps], scores[keeps], labels[keeps]]])

    LOGGER.info("converting to submission format...")
    ret_save = defaultdict(list)
    for fname, (bboxes, scores, labels) in ret:
        for bbox, score, label in zip(bboxes, scores, labels):
            bbox = xywha2xy4(bbox).ravel()
            line = "%s %.12f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f" % (fname, score, *bbox)
            ret_save[dataset.label2name[label]].append(line)

    LOGGER.info("saving DOTA...")
    os.makedirs(os.path.join(config["results_dir"]), exist_ok=True)
    for name, dets in ret_save.items():
        with open(os.path.join(config["results_dir"], "Task%d_%s.txt" % (1, name)), "wt") as f:
            f.write("\n".join(dets))

    LOGGER.info("saving GPKG...")
    filesystem = _get_filesystem(data_config["data_dir"], config["s3_bucket_name"], config["s3_profile_name"])
    eopatches = dataset.dataframe.EOPATCH_NAME.unique()

    eop_data = {
        eopatch: get_bbox(eopatch, data_config["eopatches_dir"], filesystem, data_config["resolution"])
        for eopatch in eopatches
    }

    for name, dets in ret_save.items():
        results = dota_results_to_gdf(
            os.path.join(config["results_dir"], "Task%d_%s.txt" % (1, name)), eop_data, underscore=False
        )
        for utm, gdf in results.items():
            gdf.to_file(
                os.path.join(config["results_dir"], f'{name}-{config["results_filename"]}'), layer=utm, driver="GPKG"
            )

    LOGGER.info("finished")


if __name__ == "__main__":
    assert args.action in ["train", "evaluate"], 'Supported actions are "train" or "evaluate"'

    # read config parameter file
    LOGGER.info(f"Reading configuration from {args.config}")
    with open(args.config, "r") as jfile:
        cfg_dict = json.load(jfile)

    cfg = cfg_dict["execute"]

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    # we assume we are on GPU for now, get this from args next
    device_ids = [0]
    torch.cuda.set_device(device_ids[0])

    if args.action == "train":
        train(config=cfg, wandb_key=args.wandb_key)
    else:
        cfg["checkpoint"] = None
        cfg["old_version"] = False
        evaluate(config=cfg)
