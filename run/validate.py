import os
import logging
import argparse
import pprint
from pathlib import Path

import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn

import _init_paths
from core.config import config, update_config, get_model_name
from utils.utils import load_checkpoint
import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", help="experiment configure file name", required=True, type=str)
    parser.add_argument("-t", "--tag", help="time tag of checkpoint", required=True, type=str)

    args, _ = parser.parse_known_args()
    update_config(args.cfg)

    return args


def main():
    args = parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logger.addHandler(console)

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    gpus = [int(i) for i in config.GPUS.split(',')]

    print("=> Loading data..")
    test_dataset = eval("dataset." + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    print("=> Constructing models..")
    model = eval("models." + config.MODEL + ".get_model")(config)
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).to(gpus[0])

    this_dir = Path(os.path.dirname(__file__))
    root_output_dir = (this_dir / ".." / config.OUTPUT_DIR).resolve()
    cfg_name = os.path.basename(args.cfg).split(".")[0]
    output_dir = root_output_dir / config.DATASET.TRAIN_DATASET / get_model_name(config) / cfg_name
    model, _ = load_checkpoint(model, None, output_dir, filename="model_best_{}.pth.tar".format(args.tag))

    print("=> Validating...")
    model.eval()

    preds = []
    confs = []
    with torch.no_grad():
        for i, batch_data in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
            kpts, pose_vis, joint_vis, pose_depths, joint_depths, meta = batch_data

            output_dict, _ = model(kpts=kpts, pose_vis=pose_vis, joint_vis=joint_vis, gt_pose_depths=pose_depths, gt_joint_depths=joint_depths, meta=meta)

            pred = output_dict["pred_depths"].detach().cpu().numpy()  # [B, Np, Nj]
            conf = output_dict["joint_depth_volume"].detach().cpu().numpy()  # [B, Np, Nj, Nrd]
            conf = np.max(conf, axis=-1)  # [B, Np, Nj]
            preds.append(pred)
            confs.append(conf)

    preds = np.concatenate(preds, axis=0)  # [N, Np, Nj]
    confs = np.concatenate(confs, axis=0)  # [N, Np, Nj]

    test_loader.dataset.evaluate(preds, confs)


if __name__ == "__main__":
    main()
