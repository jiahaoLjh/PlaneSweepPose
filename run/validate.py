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

from model.mvmppe import torch_back_project_pose,torch_unfold_camera_param, torch_back_project_point
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", help="experiment configure file name", required=True, type=str)
    parser.add_argument("-t", "--tag", help="time tag of checkpoint", required=True, type=str)

    args, _ = parser.parse_known_args()
    update_config(args.cfg)

    return args

kps_lines = ((0,1),(1,2),(2,3),(3,4),(4,5),(4,6),
            (5,7),(7,9),(9,11),
            (6,8),(8,10),(6,12),(11,12),
            (11,13),(13,15),
            (12,14),(14,16))
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
            #TODO --- add code for visualize
            pred_depth = output_dict["pred_pose_depths"].detach().cpu().numpy() 
            cam_target = meta[0]['camera']
            kpts_2d_target = kpts[0]
            kpts_3d = torch_back_project_pose(kpts_2d_target,pred_depth,cam_target)
            kpts_3d = np.squeeze(kpts_3d)
            
            fig = plt.figure(figsize(19,10))
            pose_ax = fig.add_subplot(1,1,1,projection='3d')
            pose_ax.set_title('Prediction')
            pose_ax.view_init(0,-90)
            pose_ax.set_xlim3d(-2000,2000)
            pose_ax.set_ylim3d(-1000,1000)
            pose_ax.set_zlim3d(-1000,1000)
            pose_ax.set_xlabel('X Label')
            pose_ax.set_xlabel('Y Label')
            pose_ax.set_xlabel('Z Label')
            for m in range(kpts_3d.shape[0]):
                for n in range(kpts_3d.shape[1]):
                    pose_ax.scatter(kpts_3d[m][n][0],kpts_3d[m][n][1],kpts_3d[m][n][2],s=10)
                    pose_ax.text(kpts_3d[m][n][0],kpts_3d[m][n][1],kpts_3d[m][n][2],n)
            
            x_3d = kpts_3d[m][:,0]
            y_3d = kpts_3d[m][:,1]
            z_3d = kpts_3d[m][:,2]
            for p in range(len(kps_lines)):
                i1 = kps_lines[p][0]
                i1 = kps_lines[p][1]
                x = np.array([x_3d[i1],x_3d[i2]])
                y = np.array([y_3d[i1],y_3d[i2]])
                z = np.array([z_3d[i1],z_3d[i2]])
                pose_ax.plot(x,y,z,linewidth=2)
            
            fig.tight_layout()
            plt.savefig('./{:0>4d}.jpg'.format(i))


if __name__ == "__main__":
    main()
