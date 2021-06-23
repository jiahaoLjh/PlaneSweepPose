import os
import copy
import logging
import json
import glob
from collections import defaultdict

import tqdm
import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from utils.transforms import project_pose, back_project_pose


logger = logging.getLogger(__name__)

TRAIN_LIST = [
    '160422_ultimatum1',
    '160224_haggling1',
    '160226_haggling1',
    '161202_haggling1',
    '160906_ian1',
    '160906_ian2',
    '160906_ian3',
    '160906_band1',
    '160906_band2',
    # '160906_band3',
]
VAL_LIST = [
    '160906_pizza1',
    '160422_haggling1',
    '160906_ian5',
    '160906_band4',
]

panoptic_joints_def = {
    'neck': 0,
    'nose': 1,
    'mid-hip': 2,
    'l-shoulder': 3,
    'l-elbow': 4,
    'l-wrist': 5,
    'l-hip': 6,
    'l-knee': 7,
    'l-ankle': 8,
    'r-shoulder': 9,
    'r-elbow': 10,
    'r-wrist': 11,
    'r-hip': 12,
    'r-knee': 13,
    'r-ankle': 14,
}
panoptic_bones_def = [
    [0, 1], [0, 2],  # trunk
    [0, 3], [3, 4], [4, 5],  # left arm
    [0, 9], [9, 10], [10, 11],  # right arm
    [2, 6], [6, 7], [7, 8],  # left leg
    [2, 12], [12, 13], [13, 14],  # right leg
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
    15: 'Lank', 16: 'Rank'
}
coco_bones_def = [
    [0, 1], [0, 2], [1, 3], [2, 4],  # head
    [3, 5], [5, 7], [7, 9],  # left arm
    [4, 6], [6, 8], [8, 10],  # right arm
    [5, 11], [6, 12],  # trunk
    [11, 13], [13, 15],  # left leg
    [12, 14], [14, 16],  # right leg
]

# === common joints definition between panoptic and coco
panoptic_to_unified = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
coco_to_unified = [0, 5, 7, 9, 11, 13, 15, 6, 8, 10, 12, 14, 16]
unified_joints_def = {
    'nose': 0,
    'l-shoulder': 1,
    'l-elbow': 2,
    'l-wrist': 3,
    'l-hip': 4,
    'l-knee': 5,
    'l-ankle': 6,
    'r-shoulder': 7,
    'r-elbow': 8,
    'r-wrist': 9,
    'r-hip': 10,
    'r-knee': 11,
    'r-ankle': 12,
}
unified_bones_def = [
    [0, 1], [0, 7],  # head
    [1, 2], [2, 3],  # left arm
    [7, 8], [8, 9],  # right arm
    [1, 4], [7, 10],  # trunk
    [4, 5], [5, 6],  # left leg
    [10, 11], [11, 12],  # right leg
]


class Panoptic(torch.utils.data.Dataset):
    def __init__(self, cfg, image_set, is_train):
        super().__init__()

        self.num_joints = len(panoptic_joints_def)
        self.num_joints_coco = len(coco_joints_def)
        self.num_joints_unified = len(unified_joints_def)
        self.cam_list = [(0, 3), (0, 6), (0, 12), (0, 13), (0, 23)]
        self.num_views = len(self.cam_list)

        self.is_train = is_train

        this_dir = os.path.dirname(__file__)
        self.dataset_root = os.path.join(this_dir, "../..", cfg.DATASET.ROOT)
        self.image_set = image_set
        self.dataset_name = "panoptic"
        self.image_width = 1920
        self.image_height = 1080

        self.max_num_persons = cfg.MULTI_PERSON.MAX_NUM_PERSONS
        self.root_id = 2
        self.use_pred_confidence = cfg.TEST.USE_PRED_CONFIDENCE
        self.nms_threshold = cfg.TEST.NMS_THRESHOLD

        if self.image_set == "train":
            self.sequence_list = TRAIN_LIST
            self._interval = 3
        elif self.image_set == "validation":
            self.sequence_list = VAL_LIST
            self._interval = 12

        self.pred_pose2d = self._get_pred_pose2d(os.path.join(self.dataset_root, "keypoints_{}_results.json".format(self.image_set)))
        self.cameras = self._get_cam()
        self.db = self._get_db()

    def _get_pred_pose2d(self, fp):
        with open(fp, "r") as f:
            logging.info("=> load {}".format(fp))
            preds = json.load(f)

        if self.is_train:
            image_to_preds = defaultdict(dict)
            for pred in preds:
                # === GT bounding boxes are used to obtain 2D pose estimation for training data so that we have the identity of each detected 2D pose
                #     Identity is needed in order to provide supervision on depths
                image_to_preds[pred["image_name"]][pred["id"]] = np.array(pred["pred"]).reshape([-1, 3])
        else:
            image_to_preds = defaultdict(list)
            for pred in preds:
                image_to_preds[pred["image_name"]].append(np.array(pred["pred"]).reshape([-1, 3]))
        logging.info("=> {} estimated 2D poses from {} images loaded".format(len(preds), len(image_to_preds)))

        return image_to_preds

    def _get_cam(self):
        cameras = dict()
        M = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [0.0, 1.0, 0.0]])

        for seq in self.sequence_list:
            cameras[seq] = dict()

            cam_file = os.path.join(self.dataset_root, seq, "calibration_{:s}.json".format(seq))
            with open(cam_file, "r") as f:
                calib = json.load(f)

            for cam in calib["cameras"]:
                if (cam['panel'], cam['node']) in self.cam_list:
                    sel_cam = {}
                    sel_cam['K'] = np.array(cam['K'])
                    sel_cam['distCoef'] = np.array(cam['distCoef'])
                    sel_cam['R'] = np.array(cam['R']).dot(M)
                    sel_cam['t'] = np.array(cam['t']).reshape((3, 1))
                    cameras[seq][(cam['panel'], cam['node'])] = sel_cam
        return cameras

    def _get_db(self):
        db = []

        for seq in tqdm.tqdm(self.sequence_list):
            cameras = self.cameras[seq]

            curr_anno = os.path.join(self.dataset_root, seq, 'hdPose3d_stage1_coco19')
            anno_files = sorted(glob.iglob("{:s}/*.json".format(curr_anno)))

            for i, anno_file in enumerate(anno_files):
                if i % self._interval == 0:
                    with open(anno_file, "r") as f:
                        bodies = json.load(f)["bodies"]
                    if len(bodies) == 0:
                        continue

                    missing_image = False
                    for k, v in cameras.items():
                        suffix = os.path.basename(anno_file).replace("body3DScene", "")
                        prefix = "{:02d}_{:02d}".format(k[0], k[1])
                        image_path = os.path.join(seq, "hdImgs", prefix, prefix + suffix)
                        image_path = image_path.replace("json", "jpg")
                        if not os.path.exists(os.path.join(self.dataset_root, image_path)):
                            logger.info("Image not found: {}. Skipped.".format(image_path))
                            missing_image = True
                            break

                        our_cam = dict()
                        our_cam['R'] = v['R']
                        our_cam['T'] = -np.dot(v['R'].T, v['t']) * 10.0  # the order to handle rotation and translation is reversed
                        our_cam['fx'] = np.array(v['K'][0, 0])
                        our_cam['fy'] = np.array(v['K'][1, 1])
                        our_cam['cx'] = np.array(v['K'][0, 2])
                        our_cam['cy'] = np.array(v['K'][1, 2])
                        our_cam['k'] = v['distCoef'][[0, 1, 4]].reshape(3, 1)
                        our_cam['p'] = v['distCoef'][[2, 3]].reshape(2, 1)

                        all_poses_3d = []
                        all_poses_3d_vis = []
                        all_poses_2d_pred = []
                        all_poses_2d_depth = []
                        person_depths = []

                        for body in bodies:
                            body_id = body["id"]

                            pose3d = np.array(body["joints19"]).reshape([-1, 4])
                            pose3d = pose3d[:self.num_joints, :]  # [Nj, 4] <x, y, z, c>

                            joints_vis = pose3d[:, -1]  # [Nj]
                            joints_vis = np.maximum(joints_vis, 0.0)

                            if joints_vis[self.root_id] <= 0.1:
                                continue

                            # === Coordinate transformation
                            M = np.array([[1.0, 0.0, 0.0],
                                          [0.0, 0.0, -1.0],
                                          [0.0, 1.0, 0.0]])
                            pose3d = pose3d[:, :3].dot(M)

                            all_poses_3d.append(pose3d * 10.0)  # [Nj, 3]
                            all_poses_3d_vis.append(joints_vis)  # [Nj]

                            pose2d, depths = project_pose(pose3d * 10.0, our_cam)
                            x_check = np.bitwise_and(pose2d[:, 0] >= 0,
                                                     pose2d[:, 0] <= self.image_width - 1)
                            y_check = np.bitwise_and(pose2d[:, 1] >= 0,
                                                     pose2d[:, 1] <= self.image_height - 1)
                            check = np.bitwise_and(x_check, y_check)

                            joints_2d_vis = np.copy(joints_vis)
                            joints_2d_vis[np.logical_not(check)] = 0

                            # === obtain 2D pose preds for training data
                            if self.is_train and body_id in self.pred_pose2d[image_path]:
                                pred_pose2d = self.pred_pose2d[image_path][body_id]  # [Nj_coco, 2+1]
                                if np.any(pred_pose2d[:, -1] > 0.5):
                                    # === GT depths
                                    all_poses_2d_depth.append(depths[panoptic_to_unified])
                                    person_depths.append(depths[self.root_id])

                                    # === pred 2D
                                    unified_poses_2d_pred = pred_pose2d[coco_to_unified, :]  # [Nj, 2+1]
                                    unified_poses_2d_pred[:, :2] = unified_poses_2d_pred[:, :2] * 0.5 + pose2d[panoptic_to_unified] * 0.5

                                    all_poses_2d_pred.append(unified_poses_2d_pred)

                        # === obtain 2D pose preds for validation data
                        if not self.is_train:
                            for pred_pose2d in self.pred_pose2d[image_path]:  # list of [Nj_coco, 2+1]
                                all_poses_2d_pred.append(pred_pose2d[coco_to_unified, :])

                        if len(all_poses_3d) > 0:
                            db.append({
                                "image_path": os.path.join(self.dataset_root, image_path),
                                "joints_3d": np.array(all_poses_3d),  # [Np, Nj, 3]
                                "joints_3d_vis": np.array(all_poses_3d_vis),  # [Np, Nj]
                                "joints_2d_pred": np.array(all_poses_2d_pred),  # [Np_hrnet, Nj_unified, 2+1]
                                "joints_2d_depth": np.array(all_poses_2d_depth),  # [Np_hrnet, Nj_unified]
                                "person_depths": np.array(person_depths),  # [Np_hrnet]
                                "camera": our_cam,
                            })

                    if missing_image:
                        continue

        logger.info("=> {} data from {} views loaded".format(len(db), self.num_views))

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

        kpts = np.zeros([self.max_num_persons, self.num_joints_unified, 2])       # [Np, Nj, 2]
        pose_vis = np.zeros([self.max_num_persons])                               # [Np]
        joint_vis = np.zeros([self.max_num_persons, self.num_joints_unified])     # [Np, Nj]
        pose_depths = np.zeros([self.max_num_persons])                            # [Np]
        joint_depths = np.zeros([self.max_num_persons, self.num_joints_unified])  # [Np, Nj]

        joints_2d_pred = db_rec["joints_2d_pred"]  # [Np_hrnet, Nj_unified, 2+1]
        joints_2d_depth = db_rec["joints_2d_depth"]  # [Np_hrnet, Nj_unified]
        person_depths = db_rec["person_depths"]  # [Np_hrnet]
        nposes = joints_2d_pred.shape[0]

        for n in range(nposes):
            kpts[n] = joints_2d_pred[n, :, :2]
            pose_vis[n] = 1
            if self.use_pred_confidence:
                joint_vis[n] = joints_2d_pred[n, :, 2]
            else:
                joint_vis[n] = 1.0

            if self.is_train:
                pose_depths[n] = person_depths[n]
                joint_depths[n] = joints_2d_depth[n] - person_depths[n]

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

    def evaluate(self, preds, confs):
        gt_num = len(self.db) // self.num_views
        assert len(preds) == len(self.db)
        assert len(confs) == len(self.db)

        eval_list = []
        total_gt = 0
        total_pred = 0

        for i in range(gt_num):
            index = self.num_views * i
            db_rec = copy.deepcopy(self.db[index])
            joints_3d = db_rec["joints_3d"]
            joints_3d_vis = db_rec["joints_3d_vis"]

            assert len(joints_3d) > 0, i

            pose3d_pool = []
            for cam_id in range(self.num_views):
                view_id = index + cam_id

                pred_pose2d = self.db[view_id]["joints_2d_pred"]  # [Np_hrnet, Nj_unified, 2+1]
                if pred_pose2d.size == 0:
                    continue

                pred_depth = preds[view_id].copy()  # [Np_max, Nj_unified]
                pred_depth = pred_depth[:pred_pose2d.shape[0]]  # [Np_hrnet, Nj_unified]

                conf_depth = confs[view_id].copy()  # [Np_max, Nj_unified]
                conf_depth = conf_depth[:pred_pose2d.shape[0]]  # [Np_hrnet, Nj_unified]

                conf_pose2d = pred_pose2d[:, :, 2]  # [Np_hrnet, Nj_unified]
                conf_pose2d = conf_pose2d * conf_depth

                # === back project [2D + depth estimation] to 3D pose
                pred_pose2d = pred_pose2d[:, :, :2].reshape(-1, 2)  # [Np_hrnet * Nj_unified, 2]
                pred_depth = pred_depth.reshape(-1)  # [Np_hrnet * Nj_unified]
                pred_pose3d = back_project_pose(pred_pose2d, pred_depth, self.db[view_id]["camera"])  # [Np_hrnet * Nj_unified, 3]
                pred_pose3d = pred_pose3d.reshape(-1, len(unified_joints_def), 3)  # [Np_hrnet, Nj_unified, 3]
                pred_pose3d = np.concatenate([pred_pose3d, conf_pose2d[:, :, np.newaxis]], axis=-1)  # [Np_hrnet, Nj_unified, 4]

                for pose in pred_pose3d:
                    pose3d_pool.append(pose)

            # === fuse multiple views
            pose3d_pool = np.stack(pose3d_pool, axis=0)  # [N, Nj, 4]
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

                    avg_pose3d = np.sum(all_pose3d[:, :, :3] * all_pose3d[:, :, -1:], axis=0) / np.sum(all_pose3d[:, :, -1:], axis=0)  # [Nj, 3]
                    final_pose3d_pool.append(np.concatenate([avg_pose3d, all_pose3d[:, :, -1:].mean(axis=0)], axis=-1))  # [Nj, 4]

            pred = np.stack([p[:, :3] for p in final_pose3d_pool])  # [Np, Nj, 3]
            scores = np.stack([np.mean(p[:, 3]) for p in final_pose3d_pool])  # [Np]
            for pose, score in zip(pred, scores):
                # === the neck joint is inferred as the middle point between left and right shoulders
                pose_neck = (pose[1, :] + pose[7, :]) / 2.0
                pose_14 = np.concatenate([pose_neck.reshape([1, 3]), pose], axis=0)
                mpjpes = []
                for gt, gt_vis in zip(joints_3d, joints_3d_vis):
                    unified_gt = gt[[0] + panoptic_to_unified]
                    unified_gt_vis = gt_vis[[0] + panoptic_to_unified]

                    vis = unified_gt_vis > 0.1  # [Nj_unified]
                    mpjpe = np.mean(np.sqrt(np.sum((pose_14[vis] - unified_gt[vis]) ** 2, axis=-1)))
                    mpjpes.append(mpjpe)
                min_gt = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                # === scores used to evaluate ap: average 2D pose confidence
                eval_list.append({
                    "mpjpe": float(min_mpjpe),
                    "score": float(score),
                    "gt_id": int(total_gt + min_gt),
                })

            total_gt += len(joints_3d)
            total_pred += len(pred)

        mpjpe_threshold = [25, 50, 100, 150]

        aps = []
        recs = []
        for t in mpjpe_threshold:
            ap, rec = self._eval_list_to_ap(eval_list, total_gt, t)
            aps.append(ap)
            recs.append(rec)

        mpjpe = self._eval_list_to_mpjpe(eval_list)
        recall = self._eval_list_to_recall(eval_list, total_gt)

        msg = "ap@25: {aps_25:.4f}\tap@50: {aps_50:.4f}\tap@100: {aps_100:.4f}\tap@150: {aps_150:.4f}\trecall@500mm: {recall:.4f}\tmpjpe@500mm: {mpjpe:.3f}".format(aps_25=aps[0], aps_50=aps[1], aps_100=aps[2], aps_150=aps[3], recall=recall, mpjpe=mpjpe)
        logger.info("==============================================\n"
                    "{} total GT poses. {} total estimated poses.\n"
                    "{}".format(total_gt, total_pred, msg))

        return mpjpe

    @staticmethod
    def _eval_list_to_ap(eval_list, total_gt, threshold):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        total_num = len(eval_list)

        tp = np.zeros(total_num)
        fp = np.zeros(total_num)
        gt_det = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] <= threshold and item["gt_id"] not in gt_det:
                tp[i] = 1
                gt_det.append(item["gt_id"])
            else:
                fp[i] = 1
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / (total_gt + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        for n in range(total_num - 2, -1, -1):
            precision[n] = max(precision[n], precision[n + 1])

        precision = np.concatenate(([0], precision, [0]))
        recall = np.concatenate(([0], recall, [1]))
        index = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[index + 1] - recall[index]) * precision[index + 1])

        return ap, recall[-2]

    @staticmethod
    def _eval_list_to_mpjpe(eval_list, threshold=500):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        gt_det = []

        mpjpes = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                mpjpes.append(item["mpjpe"])
                gt_det.append(item["gt_id"])

        return np.mean(mpjpes) if len(mpjpes) > 0 else np.inf

    @staticmethod
    def _eval_list_to_recall(eval_list, total_gt, threshold=500):
        gt_ids = [e["gt_id"] for e in eval_list if e["mpjpe"] < threshold]

        return len(np.unique(gt_ids)) / total_gt
