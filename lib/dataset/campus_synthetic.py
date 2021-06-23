import os
import json
import logging
import pickle
import random
import copy

import numpy as np
import torch

from utils.transforms import rotate_points, project_pose


logger = logging.getLogger(__name__)

coco_joints_def = {
    0: 'nose',
    1: 'Leye', 2: 'Reye',
    3: 'Lear', 4: 'Rear',
    5: 'Lsho', 6: 'Rsho',
    7: 'Lelb', 8: 'Relb',
    9: 'Lwri', 10: 'Rwri',
    11: 'Lhip', 12: 'Rhip',
    13: 'Lkne', 14: 'Rkne',
    15: 'Lank', 16: 'Rank',
}
coco_bones_def = [
    [0, 1], [0, 2], [1, 3], [2, 4],  # head
    [3, 5], [5, 7], [7, 9],  # left arm
    [4, 6], [6, 8], [8, 10],  # right arm
    [5, 11], [6, 12],  # trunk
    [11, 13], [13, 15],  # left leg
    [12, 14], [14, 16],  # right leg
]


class CampusSynthetic(torch.utils.data.Dataset):
    def __init__(self, cfg, image_set, is_train):
        super().__init__()

        self.num_joints = len(coco_joints_def)
        self.cam_list = [0, 1, 2]
        self.num_views = len(self.cam_list)

        self.is_train = is_train

        this_dir = os.path.dirname(__file__)
        self.dataset_root = os.path.join(this_dir, "../..", cfg.DATASET.ROOT)
        self.image_set = image_set
        self.dataset_name = "campus_synthetic"
        self.image_width = 360
        self.image_height = 288

        self.max_num_persons = cfg.MULTI_PERSON.MAX_NUM_PERSONS
        self.data_augmentation = cfg.DATASET.DATA_AUGMENTATION
        self.perturb_sigma = cfg.DATASET.PERTURB_SIGMA

        self.x_range = cfg.DATASET.SYNTHESIS_X_RANGE
        self.y_range = cfg.DATASET.SYNTHESIS_Y_RANGE

        pose_db_file = os.path.join(self.dataset_root, "..", "panoptic_training_pose.pkl")
        with open(pose_db_file, "rb") as f:
            pose_db = pickle.load(f)

        self.pose_db = []
        for pose in pose_db:
            if np.all(pose["vis"]):
                self.pose_db.append(pose)
        logger.info("{} poses loaded from {}".format(len(self.pose_db), pose_db_file))

        self.cameras = self._get_cam()

    def _get_cam(self):
        cam_file = os.path.join(self.dataset_root, "calibration_campus.json")
        with open(cam_file, "r") as f:
            cameras = json.load(f)

        for cam_id, cam in cameras.items():
            for k, v in cam.items():
                cameras[cam_id][k] = np.array(v)
            cameras[cam_id]["id"] = cam_id

        return cameras

    def __len__(self):
        return len(self.pose_db)

    def __getitem__(self, idx):
        nposes = np.random.choice(range(1, self.max_num_persons))

        bbox_list = []
        center_list = []

        select_poses = np.random.choice(self.pose_db, nposes)
        joints_3d = np.array([p['pose'] for p in select_poses])  # [Np, Nj, 3]
        joints_3d_vis = np.array([p['vis'] for p in select_poses])  # [Np, Nj, 3]
        joints_3d_vis = joints_3d_vis[:, :, 0]  # [Np, Nj]

        for n in range(nposes):
            points = joints_3d[n][:, :2].copy()
            center = (points[11, :2] + points[12, :2]) / 2  # middle point of left and right hip
            rot_rad = np.random.uniform(-180, 180)

            # === randomly place 3D pose in the scene
            new_center = self.get_new_center(center_list)

            # === rotate 3D pose along z-axis
            new_xy = rotate_points(points, center, rot_rad) - center + new_center

            loop_count = 0
            while not self.isvalid(new_center, self.calc_bbox(new_xy, joints_3d_vis[n]), bbox_list):
                loop_count += 1
                if loop_count >= 100:
                    break
                new_center = self.get_new_center(center_list)
                new_xy = rotate_points(points, center, rot_rad) - center + new_center

            if loop_count >= 100:
                nposes = n
                joints_3d = joints_3d[:n]
                joints_3d_vis = joints_3d_vis[:n]
                break
            else:
                center_list.append(new_center)
                bbox_list.append(self.calc_bbox(new_xy, joints_3d_vis[n]))
                joints_3d[n][:, :2] = new_xy

        kpts, pose_vis, joint_vis, pose_depths, joint_depths, meta = [], [], [], [], [], []
        for cam_id in np.random.permutation(list(self.cameras.keys())):
            k, pv, jv, pd, jd, m = self._get_single_view_item(joints_3d, joints_3d_vis, self.cameras[cam_id])
            kpts.append(k)  # [Np, Nj, 2]
            pose_vis.append(pv)  # [Np]
            joint_vis.append(jv)  # [Np, Nj]
            pose_depths.append(pd)  # [Np]
            joint_depths.append(jd)  # [Np, Nj]
            meta.append(m)

        return kpts, pose_vis, joint_vis, pose_depths[0], joint_depths[0], meta

    def _get_single_view_item(self, joints_3d, joints_3d_vis, cam):
        joints_3d = copy.deepcopy(joints_3d)
        joints_3d_vis = copy.deepcopy(joints_3d_vis)
        joints_3d_vis = joints_3d_vis.astype(np.float)
        nposes = len(joints_3d)

        kpts = np.zeros([self.max_num_persons, self.num_joints, 2])       # [Np, Nj, 2]
        pose_vis = np.zeros([self.max_num_persons])                       # [Np]
        joint_vis = np.zeros([self.max_num_persons, self.num_joints])     # [Np, Nj]
        pose_depths = np.zeros([self.max_num_persons])                    # [Np]
        joint_depths = np.zeros([self.max_num_persons, self.num_joints])  # [Np, Nj]

        valid_poses = 0
        for n in range(nposes):
            """
            1. project 3D pose to 2D
            2. check if 2D joints is outside canvas and update vis
            3. pose is visible <=> left and right hip is visible
            4. perturb 2D joints vis
            """
            vis = joints_3d_vis[n] > 0  # [Nj]

            pose2d, depths = project_pose(joints_3d[n], cam)  # [Nj, 2], [Nj]

            x_check = np.bitwise_and(pose2d[:, 0] >= 0,
                                     pose2d[:, 0] <= self.image_width - 1)
            y_check = np.bitwise_and(pose2d[:, 1] >= 0,
                                     pose2d[:, 1] <= self.image_height - 1)
            check = np.bitwise_and(x_check, y_check)
            vis[np.logical_not(check)] = 0  # visible both in 3D annotation and 2D image

            if vis[11] == 0.0 or vis[12] == 0.0:
                continue

            if self.data_augmentation:
                # === perturb visibility
                for j in range(self.num_joints):
                    offset = np.random.randn(2) * self.perturb_sigma  # [2]
                    pose2d[j] = pose2d[j] + offset

                    # === general visibility simulating occluded and inexact detections
                    dist = np.linalg.norm(offset)
                    scale = np.exp(-(dist ** 2 / 2.0) / (2 * self.perturb_sigma ** 2)) / 2.0 + 0.5

                    if j in [7, 8]:
                        # === elbow
                        scale = scale * np.random.uniform(0.5, 1.0) if random.random() < 0.1 else scale
                    elif j in [9, 10]:
                        # === wrist
                        scale = scale * np.random.uniform(0.5, 1.0) if random.random() < 0.2 else scale
                    else:
                        # === other joints
                        scale = scale * np.random.uniform(0.5, 1.0) if random.random() < 0.05 else scale

                    vis[j] = vis[j] * scale

            kpts[valid_poses] = pose2d
            pose_vis[valid_poses] = 1
            joint_vis[valid_poses] = vis
            pose_depths[valid_poses] = (depths[11] + depths[12]) / 2.0
            joint_depths[valid_poses] = depths - pose_depths[valid_poses]

            valid_poses += 1

        meta = {
            "image": "",
            "image_height": self.image_height,
            "image_width": self.image_width,
            "num_persons": valid_poses,
            "joints_2d": kpts,
            "joints_2d_vis": joint_vis,
            "pose_depths": pose_depths,
            "joint_depths": joint_depths,
            "pose_vis": pose_vis,
            "camera": cam,
        }
        kpts = torch.as_tensor(kpts, dtype=torch.float)
        pose_vis = torch.as_tensor(pose_vis, dtype=torch.float)
        joint_vis = torch.as_tensor(joint_vis, dtype=torch.float)
        pose_depths = torch.as_tensor(pose_depths, dtype=torch.float)
        joint_depths = torch.as_tensor(joint_depths, dtype=torch.float)

        return kpts, pose_vis, joint_vis, pose_depths, joint_depths, meta

    @staticmethod
    def calc_bbox(pose, pose_vis):
        index = pose_vis > 0
        bbox = [np.min(pose[index, 0]), np.min(pose[index, 1]),
                np.max(pose[index, 0]), np.max(pose[index, 1])]

        return np.array(bbox)

    def isvalid(self, new_center, bbox, bbox_list):
        new_center_us = new_center.reshape(1, -1)
        vis = 0
        for k, cam in self.cameras.items():
            loc_2d, _ = project_pose(np.hstack((new_center_us, [[1000.0]])), cam)
            if 10 < loc_2d[0, 0] < self.image_width - 10 and 10 < loc_2d[0, 1] < self.image_height - 10:
                vis += 1

        if len(bbox_list) == 0:
            # === at least visible from two cameras
            return vis >= 2

        bbox_list = np.array(bbox_list)
        x0 = np.maximum(bbox[0], bbox_list[:, 0])
        y0 = np.maximum(bbox[1], bbox_list[:, 1])
        x1 = np.minimum(bbox[2], bbox_list[:, 2])
        y1 = np.minimum(bbox[3], bbox_list[:, 3])

        intersection = np.maximum(0, (x1 - x0) * (y1 - y0))
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        area_list = (bbox_list[:, 2] - bbox_list[:, 0]) * (bbox_list[:, 3] - bbox_list[:, 1])
        iou_list = intersection / (area + area_list - intersection)

        return vis >= 2 and np.max(iou_list) < 0.01

    def get_new_center(self, center_list):
        if len(center_list) == 0 or random.random() < 0.7:
            x_min, x_max = self.x_range
            y_min, y_max = self.y_range
            new_center = np.array([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)])
        else:
            xy = center_list[np.random.choice(range(len(center_list)))]
            new_center = xy + np.random.normal(500, 50, 2) * np.random.choice([1, -1], 2)

        return new_center
