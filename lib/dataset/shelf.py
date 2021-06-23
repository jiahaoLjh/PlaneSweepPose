import os
import copy
import logging
import pickle
import json
from collections import OrderedDict

import numpy as np
import scipy.io as scio
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import torch

from utils.transforms import project_pose, back_project_pose


logger = logging.getLogger(__name__)

shelf_joints_def = {
    'Right-Ankle': 0,
    'Right-Knee': 1,
    'Right-Hip': 2,
    'Left-Hip': 3,
    'Left-Knee': 4,
    'Left-Ankle': 5,
    'Right-Wrist': 6,
    'Right-Elbow': 7,
    'Right-Shoulder': 8,
    'Left-Shoulder': 9,
    'Left-Elbow': 10,
    'Left-Wrist': 11,
    'Bottom-Head': 12,
    'Top-Head': 13
}
shelf_bones_def = [
    [13, 12],  # head
    [12, 9], [9, 10], [10, 11],  # left arm
    [12, 8], [8, 7], [7, 6],  # right arm
    [9, 3], [8, 2],  # trunk
    [3, 4], [4, 5],  # left leg
    [2, 1], [1, 0],  # right leg
]

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


class Shelf(torch.utils.data.Dataset):
    def __init__(self, cfg, image_set, is_train):
        super().__init__()

        self.num_joints = len(shelf_joints_def)
        self.num_joints_coco = len(coco_joints_def)
        self.cam_list = [0, 1, 2, 3, 4]
        self.num_views = len(self.cam_list)
        self.frame_range = list(range(300, 601))

        self.is_train = is_train

        this_dir = os.path.dirname(__file__)
        self.dataset_root = os.path.join(this_dir, "../..", cfg.DATASET.ROOT)
        self.image_set = image_set
        self.dataset_name = "shelf"
        self.image_width = 1032
        self.image_height = 776

        self.max_num_persons = cfg.MULTI_PERSON.MAX_NUM_PERSONS

        self.pred_pose2d = self._get_pred_pose2d()
        self.cameras = self._get_cam()
        self.db = self._get_db()

        self.use_pred_confidence = cfg.TEST.USE_PRED_CONFIDENCE
        self.nms_threshold = cfg.TEST.NMS_THRESHOLD

    def _get_pred_pose2d(self):
        fp = os.path.join(self.dataset_root, "pred_shelf_maskrcnn_hrnet_coco.pkl")
        with open(fp, "rb") as f:
            logging.info("=> load {}".format(fp))
            pred_2d = pickle.load(f)

        return pred_2d

    def _get_cam(self):
        cam_file = os.path.join(self.dataset_root, "calibration_shelf.json")
        with open(cam_file, "r") as f:
            cameras = json.load(f)

        for cam_id, cam in cameras.items():
            for k, v in cam.items():
                cameras[cam_id][k] = np.array(v)
            cameras[cam_id]["id"] = cam_id

        return cameras

    def _get_db(self):
        db = []

        datafile = os.path.join(self.dataset_root, "actorsGT.mat")
        data = scio.loadmat(datafile)
        actor_3d = np.array(np.array(data["actor3D"].tolist()).tolist()).squeeze()  # [Np, Nf]

        num_persons = 3

        all_depths = []
        for f in self.frame_range:
            for cam_id, cam in self.cameras.items():
                image_path = os.path.join("Camera{}".format(cam_id), "img_{:06d}.png".format(f))

                all_poses_3d = []
                all_poses_3d_vis = []
                all_poses_2d = []
                all_poses_2d_vis = []

                for pid in range(num_persons):
                    pose3d = actor_3d[pid][f] * 1000.0
                    if pose3d.size > 0:
                        all_poses_3d.append(pose3d)  # [Nj, 3]
                        all_poses_3d_vis.append(np.ones([self.num_joints]))  # [Nj]

                        pose2d, depths = project_pose(pose3d, cam)  # [Nj, 2], [Nj]
                        all_depths.extend(depths.tolist())

                        x_check = np.bitwise_and(pose2d[:, 0] >= 0,
                                                 pose2d[:, 0] <= self.image_width - 1)
                        y_check = np.bitwise_and(pose2d[:, 1] >= 0,
                                                 pose2d[:, 1] <= self.image_height - 1)
                        check = np.bitwise_and(x_check, y_check)

                        joints_2d_vis = np.ones([self.num_joints])
                        joints_2d_vis[np.logical_not(check)] = 0
                        all_poses_2d.append(pose2d)  # [Nj, 2]
                        all_poses_2d_vis.append(joints_2d_vis)  # [Nj]

                pred_index = "{}_{}".format(cam_id, f)
                preds = self.pred_pose2d[pred_index]
                preds = np.array([p["pred"] for p in preds])  # [Np, Nj_coco, 2+1]

                db.append({
                    "image_path": os.path.join(self.dataset_root, image_path),
                    "joints_3d": np.array(all_poses_3d),  # [Np, Nj, 3]
                    "joints_3d_vis": np.array(all_poses_3d_vis),  # [Np, Nj] all one
                    "joints_2d": np.array(all_poses_2d),  # [Np, Nj, 2]
                    "joints_2d_vis": np.array(all_poses_2d_vis),  # [Np, Nj]
                    "camera": cam,
                    "pred_pose2d": preds,  # [Np_hrnet, Nj_coco, 2+1]
                })

        return db

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        frame_id = idx // self.num_views

        # === obtain all camera views corresponding to the frame
        start = frame_id * self.num_views
        end = start + self.num_views
        views = list(range(start, end))
        # === move the current view to the first as target view
        del views[idx - start]
        views.insert(0, idx)

        kpts, pose_vis, joint_vis, pose_depths, joint_depths, meta = [], [], [], [], [], []
        for view_id in views:
            k, pv, jv, pd, jd, m = self._get_single_view_item(view_id)
            kpts.append(k)  # [Np, Nj, 2]
            pose_vis.append(pv)  # [Np]
            joint_vis.append(jv)  # [Np, Nj]
            pose_depths.append(pd)  # [Np]
            joint_depths.append(jd)  # [Np, Nj]
            meta.append(m)

        return kpts, pose_vis, joint_vis, pose_depths[0], joint_depths[0], meta

    def _get_single_view_item(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        kpts = np.zeros([self.max_num_persons, self.num_joints_coco, 2])       # [Np, Nj, 2]
        pose_vis = np.zeros([self.max_num_persons])                            # [Np]
        joint_vis = np.zeros([self.max_num_persons, self.num_joints_coco])     # [Np, Nj]
        pose_depths = np.zeros([self.max_num_persons])                         # [Np]
        joint_depths = np.zeros([self.max_num_persons, self.num_joints_coco])  # [Np, Nj]

        pred_pose2d = db_rec["pred_pose2d"]  # [Np_hrnet, Nj_coco, 2+1]
        nposes = pred_pose2d.shape[0]

        for n in range(nposes):
            kpts[n] = pred_pose2d[n, :, :2]
            pose_vis[n] = 1
            if self.use_pred_confidence:
                joint_vis[n] = pred_pose2d[n, :, 2]
            else:
                joint_vis[n] = 1.0

        meta = {
            "image": db_rec["image_path"],
            "image_height": self.image_height,
            "image_width": self.image_width,
            "num_persons": nposes,
            "joints_2d": kpts,
            "joints_2d_vis": joint_vis,
            "pose_depths": pose_depths,
            "joint_depths": joint_depths,
            "pose_vis": pose_vis,
            "camera": db_rec["camera"],
        }
        kpts = torch.as_tensor(kpts, dtype=torch.float)
        pose_vis = torch.as_tensor(pose_vis, dtype=torch.float)
        joint_vis = torch.as_tensor(joint_vis, dtype=torch.float)
        pose_depths = torch.as_tensor(pose_depths, dtype=torch.float)
        joint_depths = torch.as_tensor(joint_depths, dtype=torch.float)

        return kpts, pose_vis, joint_vis, pose_depths, joint_depths, meta

    def evaluate(self, preds, confs, recall_threshold=500):
        """
        Args
            preds: [N, Np, Nj]
        """
        datafile = os.path.join(self.dataset_root, "actorsGT.mat")
        data = scio.loadmat(datafile)
        actor_3d = np.array(np.array(data["actor3D"].tolist()).tolist()).squeeze()  # [Np, Nf]

        num_persons = 3

        alpha = 0.5
        limbs = [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11], [12, 13]]

        total_gt = 0
        match_gt = 0
        correct_parts = np.zeros([num_persons])
        total_parts = np.zeros([num_persons])
        bone_correct_parts = np.zeros([num_persons, 10])

        for frame_id, frame_no in enumerate(self.frame_range):
            pose3d_pool = []
            angle_pool = []
            for cam_id in range(self.num_views):
                view_id = frame_id * self.num_views + cam_id

                pred_pose2d = self.db[view_id]["pred_pose2d"]  # [Np, Nj, 2+1]
                pred_depth = preds[view_id].copy()  # [Np_max, Nj]
                pred_depth = pred_depth[:pred_pose2d.shape[0]]  # [Np, Nj]

                conf_depth = confs[view_id].copy()  # [Np_max, Nj]
                conf_depth = conf_depth[:pred_pose2d.shape[0]]  # [Np, Nj]

                conf_pose2d = pred_pose2d[:, :, 2]  # [Np, Nj]
                conf_pose2d = conf_pose2d * conf_depth

                # === back project [2D + depth estimation] to 3D pose
                pred_pose2d = pred_pose2d[:, :, :2].reshape(-1, 2)  # [Np * Nj, 2]
                pred_depth = pred_depth.reshape(-1)  # [Np * Nj]
                pred_coco = back_project_pose(pred_pose2d, pred_depth, self.db[view_id]["camera"])  # [Np * Nj, 3]
                pred_coco = pred_coco.reshape(-1, self.num_joints_coco, 3)  # [Np, Nj, 3]
                pred_coco = np.concatenate([pred_coco, conf_pose2d[:, :, np.newaxis]], axis=-1)  # [Np, Nj, 4]

                # === use the angle between the facing direction of each pose and the camera ray pointing towards the pose as the weight for fusion
                for pose in pred_coco:
                    pose3d_pool.append(pose)

                    lsh = pose[5, :3]
                    rsh = pose[6, :3]
                    lhip = pose[11, :3]
                    rhip = pose[12, :3]

                    msh = (lsh + rsh) / 2.0
                    mhip = (lhip + rhip) / 2.0

                    sh = rsh - lsh
                    spine = mhip - msh
                    person_dir = np.cross(sh, spine)

                    cam_loc = self.db[view_id]["camera"]["T"].flatten()
                    person_cam = msh - cam_loc

                    v1 = person_dir / np.linalg.norm(person_dir)
                    v2 = person_cam / np.linalg.norm(person_cam)

                    angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)) / np.pi * 180.0
                    if angle > 90:
                        angle = 180 - angle
                    angle_pool.append(angle)

            # === fuse multiple views
            pose3d_pool = np.stack(pose3d_pool, axis=0)  # [N, Nj, 4]
            angle_pool = np.array(angle_pool)  # [N]
            dist_matrix = np.expand_dims(pose3d_pool[:, :, :3], axis=1) - np.expand_dims(pose3d_pool[:, :, :3], axis=0)  # [N, N, Nj, 3]
            dist_matrix = np.sqrt(np.sum(dist_matrix ** 2, axis=-1))  # [N, N, Nj]
            dist_matrix = np.mean(dist_matrix, axis=-1)  # [N, N]

            dist_vector = squareform(dist_matrix)
            Z = linkage(dist_vector, 'single')
            labels = fcluster(Z, t=self.nms_threshold, criterion='distance')

            clusters = [[] for _ in range(labels.max())]
            for pid, label in enumerate(labels):
                clusters[label - 1].append(pid)

            final_pose3d_pool = []

            for cluster in clusters:
                if len(cluster) == 1:
                    final_pose3d_pool.append(pose3d_pool[cluster[0]])
                else:
                    all_pose3d = pose3d_pool[np.array(cluster)]  # [Nc, Nj, 4]
                    all_angle = angle_pool[np.array(cluster)]  # [Nc]

                    weights = 90 - all_angle
                    mean_pose3d = np.sum(all_pose3d[:, :, :3] * weights.reshape(-1, 1, 1), axis=0) / (np.sum(weights) + 1e-8)
                    final_pose3d_pool.append(mean_pose3d)

            pred = np.stack([self.coco2shelf3D(p[:, :3]) for p in final_pose3d_pool])  # [Np, Nj, 3]

            for person in range(num_persons):
                gt = actor_3d[person][frame_no] * 1000.0  # [Nj, 3]
                if gt.size == 0:
                    continue

                mpjpes = np.mean(np.sqrt(np.sum((gt[np.newaxis] - pred) ** 2, axis=-1)), axis=-1)  # [Np]
                min_n = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                if min_mpjpe < recall_threshold:
                    match_gt += 1
                total_gt += 1

                for j, k in enumerate(limbs):
                    total_parts[person] += 1
                    error_s = np.linalg.norm(pred[min_n, k[0]] - gt[k[0]])
                    error_e = np.linalg.norm(pred[min_n, k[1]] - gt[k[1]])
                    limb_length = np.linalg.norm(gt[k[0]] - gt[k[1]])
                    if (error_s + error_e) / 2.0 <= alpha * limb_length:
                        correct_parts[person] += 1
                        bone_correct_parts[person, j] += 1
                pred_hip = (pred[min_n, 2] + pred[min_n, 3]) / 2.0
                gt_hip = (gt[2] + gt[3]) / 2.0
                total_parts[person] += 1
                error_s = np.linalg.norm(pred_hip - gt_hip)
                error_e = np.linalg.norm(pred[min_n, 12] - gt[12])
                limb_length = np.linalg.norm(gt_hip - gt[12])
                if (error_s + error_e) / 2.0 <= alpha * limb_length:
                    correct_parts[person] += 1
                    bone_correct_parts[person, 9] += 1

        bone_group = OrderedDict(
            [('Head', [8]), ('Torso', [9]), ('Upper arms', [5, 6]),
             ('Lower arms', [4, 7]), ('Upper legs', [1, 2]), ('Lower legs', [0, 3])])

        actor_pcp = correct_parts / (total_parts + 1e-8)
        avg_pcp = np.mean(actor_pcp[:3])

        bone_person_pcp = OrderedDict()
        for k, v in bone_group.items():
            bone_person_pcp[k] = np.sum(bone_correct_parts[:, v], axis=-1) / (total_parts / 10 * len(v) + 1e-8)

        logger.info("==============================================\n"
                    "     | Actor 1 | Actor 2 | Actor 3 | Average |\n"
                    " PCP |  {pcp_1:.2f}  |  {pcp_2:.2f}  |  {pcp_3:.2f}  |  {pcp_avg:.2f}  |\t Recall@500m: {recall:.4f}".format(
                        pcp_1=actor_pcp[0] * 100, pcp_2=actor_pcp[1] * 100, pcp_3=actor_pcp[2] * 100, pcp_avg=avg_pcp * 100, recall=match_gt / (total_gt + 1e-8)))
        for k, v in bone_person_pcp.items():
            logger.info("{:10s}: {:.2f}".format(k, np.mean(v)))

        return avg_pcp

    @staticmethod
    def coco2shelf3D(coco_pose):
        """
        transform coco order 3d pose to shelf dataset order with interpolation
        :param coco_pose: np.array with shape 17x3
        :return: 3D pose in shelf order with shape 14x3
        """
        shelf_pose = np.zeros([14, 3])
        coco2shelf = np.array([16, 14, 12, 11, 13, 15, 10, 8, 6, 5, 7, 9])
        shelf_pose[:12] = coco_pose[coco2shelf]

        mid_sho = (coco_pose[5] + coco_pose[6]) / 2  # L and R shoulder
        head_center = (coco_pose[3] + coco_pose[4]) / 2  # middle of two ear

        head_bottom = (mid_sho + head_center) / 2  # nose and head center
        head_top = head_bottom + (head_center - head_bottom) * 2
        # shelf_pose[12] = head_bottom
        # shelf_pose[13] = head_top

        shelf_pose[12] = (coco_pose[5] + coco_pose[6]) / 2  # Use middle of shoulder to init
        shelf_pose[13] = coco_pose[0]  # Use nose to init

        shelf_pose[13] = shelf_pose[12] + (shelf_pose[13] - shelf_pose[12]) * np.array([0.75, 0.75, 1.5])
        shelf_pose[12] = shelf_pose[12] + (coco_pose[0] - shelf_pose[12]) * np.array([0.5, 0.5, 0.5])

        alpha = 0.75
        shelf_pose[13] = shelf_pose[13] * alpha + head_top * (1 - alpha)
        shelf_pose[12] = shelf_pose[12] * alpha + head_bottom * (1 - alpha)

        return shelf_pose
