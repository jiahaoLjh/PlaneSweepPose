import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.transforms import torch_back_project_pose, torch_project_pose
from models.softargmax import SoftArgMax
from models.cnns import PoseCNN, JointCNN


class MultiViewMultiPersonPoseNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.dataset = cfg.DATASET.TEST_DATASET

        self.pose_min_depth = cfg.MULTI_PERSON.POSE_MIN_DEPTH
        self.pose_max_depth = cfg.MULTI_PERSON.POSE_MAX_DEPTH
        self.pose_num_depth_layers = cfg.MULTI_PERSON.POSE_NUM_DEPTH_LAYERS
        self.joint_min_depth = cfg.MULTI_PERSON.JOINT_MIN_DEPTH
        self.joint_max_depth = cfg.MULTI_PERSON.JOINT_MAX_DEPTH
        self.joint_num_depth_layers = cfg.MULTI_PERSON.JOINT_NUM_DEPTH_LAYERS
        self.pose_sigma = cfg.MULTI_PERSON.POSE_SIGMA
        self.joint_sigma = cfg.MULTI_PERSON.JOINT_SIGMA

        self.plabels = np.arange(self.pose_num_depth_layers) / (self.pose_num_depth_layers - 1) * (self.pose_max_depth - self.pose_min_depth) + self.pose_min_depth  # [D]
        self.jlabels = np.arange(self.joint_num_depth_layers) / (self.joint_num_depth_layers - 1) * (self.joint_max_depth - self.joint_min_depth) + self.joint_min_depth  # [RD]
        self.register_buffer("pose_depth_labels", torch.as_tensor(self.plabels, dtype=torch.float))
        self.register_buffer("joint_relative_depth_labels", torch.as_tensor(self.jlabels, dtype=torch.float))

        self.pose_cnn = PoseCNN(num_joints=cfg.NETWORK.NUM_JOINTS, hidden_size=cfg.NETWORK.HIDDEN_SIZE, output_size=1)
        self.joint_cnn = JointCNN(num_joints=cfg.NETWORK.NUM_JOINTS, hidden_size=cfg.NETWORK.HIDDEN_SIZE, output_size=cfg.NETWORK.NUM_JOINTS)
        self.softargmax_kernel_size = cfg.NETWORK.SOFTARGMAX_KERNEL_SIZE
        self.softargmax_net = SoftArgMax()

    def feature_extraction(self, poses_3d, poses_2d_ref, vis_target, vis_ref, meta_target, meta_ref, sigma):
        """
        Args
            poses_3d: [B, Npt, Nj, Nd, 3]
            poses_2d_ref: [B, Npr, Nj, 2]
            vis_target: [B, Npt, Nj]
            vis_ref: [B, Npr, Nj]
        Steps:
            1. project poses_3d to reference view
            2. search for the nearest pose from poses_2d_ref
            3. compute score
            4. compute visibility in the reference view (bounding)
            5. return per joint score and bounding (used for fusing multiple views)
        Returns
            score: [B, Npt, Nj, Nd]
            bounding: [B, Npt, Nj, Nd]
        """
        batch_size, num_persons, num_joints, num_depth_levels, _ = poses_3d.size()
        device = poses_3d.device

        cam_ref = meta_ref["camera"]

        # === project 3d pose to reference view
        poses_2d_target = torch_project_pose(poses_3d.reshape(batch_size, num_persons, num_joints * num_depth_levels, 3), cam_ref)  # [B, Npt, Nj * Nd, 2]
        poses_2d_target = poses_2d_target.reshape(batch_size, num_persons, num_joints, num_depth_levels, 2)  # [B, Npt, Nj, Nd, 2]

        # === form pose distance matrix between target and reference view
        pt = poses_2d_target.reshape(batch_size, num_persons, 1, num_joints, num_depth_levels, 2)  # [B, Npt, 1, Nj, Nd, 2]
        pr = poses_2d_ref.reshape(batch_size, 1, num_persons, num_joints, 1, 2)  # [B, 1, Npr, Nj, 1, 2]
        poses_dist = torch.sum((pt - pr) ** 2, dim=-1)  # [B, Npt, Npf, Nj, Nd]
        # === distance is weighted by joint vis of reference poses
        poses_dist = torch.sum(poses_dist * vis_ref.reshape(batch_size, 1, num_persons, num_joints, 1), dim=-2) / (torch.sum(vis_ref.reshape(batch_size, 1, num_persons, num_joints, 1), dim=-2) + 1e-8)  # [B, Npt, Npf, Nd]

        # === set distance to padding poses to a large value
        for b in range(batch_size):
            poses_dist[b, :, meta_ref["num_persons"][b]:, :] = 1e5

        # === obtain the nearest pose
        min_dist, min_matching = poses_dist.min(dim=-2)  # [B, Npt, Nd], [B, Npt, Nd]
        matched_poses_2d_ref = torch.gather(poses_2d_ref.unsqueeze(3).repeat(1, 1, 1, num_depth_levels, 1), dim=1, index=min_matching.reshape(batch_size, num_persons, 1, num_depth_levels, 1).repeat(1, 1, num_joints, 1, 2))  # [B, Npt, Nj, Nd, 2]
        matched_vis_ref = torch.gather(vis_ref.reshape(batch_size, num_persons, num_joints, 1, 1).repeat(1, 1, 1, num_depth_levels, 1),
                                       dim=1,
                                       index=min_matching.reshape(batch_size, num_persons, 1, num_depth_levels, 1).repeat(1, 1, num_joints, 1, 1))  # [B, Npt, Nj, Nd, 1]

        # === compute score for each target pose based on the distance to its respective matched reference pose
        matching_dist = torch.sum((poses_2d_target - matched_poses_2d_ref) ** 2, dim=-1)  # [B, Npt, Nj, Nd]
        vr = matched_vis_ref.reshape(batch_size, num_persons, num_joints, num_depth_levels)  # [B, Npt, Nj, Nd]
        if "panoptic" in self.dataset:
            score = torch.exp(-torch.sqrt(matching_dist) / sigma)  # [B, Npt, Nj, Nd]
        else:
            score = torch.exp(-matching_dist / (sigma ** 2))  # [B, Npt, Nj, Nd]

        # === compute the visibility of each target joint in the reference view
        bounding = torch.zeros(batch_size, num_persons, num_joints, num_depth_levels)  # [B, Npt, Nj, Nd]
        bounding = bounding.to(device)
        for b in range(batch_size):
            image_width = meta_ref["image_width"][b]
            image_height = meta_ref["image_height"][b]
            bounding[b, :, :, :] = (poses_2d_target[b, :, :, :, 0] >= 0) & (poses_2d_target[b, :, :, :, 1] >= 0) & (poses_2d_target[b, :, :, :, 0] <= image_width - 1) & (poses_2d_target[b, :, :, :, 1] <= image_height - 1)

        # === incorporate reference joint visibility into the aggregation of scores
        bounding = bounding * vr

        return score, bounding

    def forward(self, kpts, pose_vis, joint_vis, gt_pose_depths, gt_joint_depths, meta):
        """
        kpts:            2D poses per view
                         list (view) of [B, Np, Nj, 2]
        pose_vis:        pose visibility per view
                         list (view) of [B, Np]
        joint_vis:       joint visibility per view
                         list (view) of [B, Np, Nj]
        gt_pose_depths:  pose depth in target view
                         [B, Np]
        gt_joint_depths: joint depth in target view
                         [B, Np, Nj]
        meta:            list (view) of dict
        """

        output = dict()

        num_views = len(kpts)
        batch_size, num_persons, num_joints, _ = kpts[0].size()

        cam_target = meta[0]["camera"]
        kpts_2d_target = kpts[0]  # [B, Np, Nj, 2]
        device = kpts_2d_target.device

        # === stage 1
        kpts_3d_all_depth = []
        for depth_id, depth_label in enumerate(self.pose_depth_labels):
            depth = depth_label.reshape(1).repeat(batch_size)  # [B]
            depth = depth.to(device)

            # === back project target poses to 3D
            kpts_3d = torch_back_project_pose(kpts_2d_target, depth, cam_target)  # [B, Np, Nj, 3]
            kpts_3d_all_depth.append(kpts_3d)

        kpts_3d_all_depth = torch.stack(kpts_3d_all_depth, dim=3)  # [B, Np, Nj, Nd, 3]

        # === extract scores from reference views
        scores = None
        boundings = None
        for rv in range(1, num_views):
            score, bounding = self.feature_extraction(kpts_3d_all_depth, kpts[rv], joint_vis[0], joint_vis[rv], meta[0], meta[rv], self.pose_sigma)  # [B, Np, Nj, Nd], [B, Np, Nj, Nd]
            if scores is None:
                scores = score * bounding
            else:
                scores += score * bounding
            if boundings is None:
                boundings = bounding
            else:
                boundings += bounding

        pose_score_volume = scores / (boundings + 1e-8)  # [B, Np, Nj, Nd]

        output["pose_score_volume"] = pose_score_volume  # [B, Np, Nj, Nd]

        pose_score_volume = pose_score_volume.reshape(batch_size * num_persons, num_joints, len(self.pose_depth_labels))  # [B * Np, Nj, Nd]
        pose_depth_volume = self.pose_cnn(pose_score_volume)  # [B * Np, 1, Nd]
        pose_depth_volume = F.softmax(pose_depth_volume, dim=-1)  # [B * Np, 1, ~Nd]

        output["pose_depth_volume"] = pose_depth_volume.reshape(batch_size, num_persons, len(self.pose_depth_labels))  # [B, Np, Nd]

        pred_pose_indices = self.softargmax_net(pose_depth_volume, torch.as_tensor(np.arange(len(self.pose_depth_labels)), dtype=torch.float, device=device), kernel_size=self.softargmax_kernel_size)  # [B * Np, 1]
        pred_pose_indices = pred_pose_indices.reshape(batch_size, num_persons)  # [B, Np]
        pred_pose_depths = pred_pose_indices / (self.pose_num_depth_layers - 1) * (self.pose_max_depth - self.pose_min_depth) + self.pose_min_depth  # [B, Np]

        output["pred_pose_indices"] = pred_pose_indices  # [B, Np]
        output["pred_pose_depths"] = pred_pose_depths  # [B, Np]

        # === stage 2
        kpts_3d_all_depth = []
        for depth_id, depth_label in enumerate(self.joint_relative_depth_labels):
            if self.training:
                depth = depth_label.reshape(1, 1).repeat(batch_size, num_persons) + gt_pose_depths  # [B, Np]
            else:
                depth = depth_label.reshape(1, 1).repeat(batch_size, num_persons) + pred_pose_depths  # [B, Np]

            # === back project target poses to 3D
            kpts_3d = torch_back_project_pose(kpts_2d_target, depth, cam_target)  # [B, Np, Nj, 3]
            kpts_3d_all_depth.append(kpts_3d)

        kpts_3d_all_depth = torch.stack(kpts_3d_all_depth, dim=3)  # [B, Np, Nj, Nd, 3]

        # === extract scores from reference views
        scores = None
        boundings = None
        for rv in range(1, num_views):
            score, bounding = self.feature_extraction(kpts_3d_all_depth, kpts[rv], joint_vis[0], joint_vis[rv], meta[0], meta[rv], self.joint_sigma)  # [B, Np, Nj, Nrd], [B, Np, Nj, Nrd]
            if scores is None:
                scores = score * bounding
            else:
                scores += score * bounding
            if boundings is None:
                boundings = bounding
            else:
                boundings += bounding

        joint_score_volume = scores / (boundings + 1e-8)  # [B, Np, Nj, Nrd]

        output["joint_score_volume"] = joint_score_volume  # [B, Np, Nj, Nrd]

        joint_score_volume = joint_score_volume.reshape(batch_size * num_persons, num_joints, len(self.joint_relative_depth_labels))  # [B * Np, Nj, Nrd]
        joint_depth_volume = self.joint_cnn(joint_score_volume)  # [B * Np, Nj, Nrd]
        joint_depth_volume = F.softmax(joint_depth_volume, dim=-1)  # [B * Np, Nj, ~Nrd]

        output["joint_depth_volume"] = joint_depth_volume.reshape(batch_size, num_persons, num_joints, len(self.joint_relative_depth_labels))  # [B, Np, Nj, Nrd]

        pred_joint_indices = self.softargmax_net(joint_depth_volume, torch.as_tensor(np.arange(len(self.joint_relative_depth_labels)), dtype=torch.float, device=device))  # [B * Np, Nj]
        pred_joint_indices = pred_joint_indices.reshape(batch_size, num_persons, num_joints)  # [B, Np, Nj]
        pred_joint_depths = pred_joint_indices / (self.joint_num_depth_layers - 1) * (self.joint_max_depth - self.joint_min_depth) + self.joint_min_depth  # [B, Np, Nj]

        output["pred_joint_indices"] = pred_joint_indices  # [B, Np, Nj]
        output["pred_joint_depths"] = pred_joint_depths  # [B, Np, Nj]

        # === losses
        if self.training:
            loss_pose = F.smooth_l1_loss(pred_pose_depths * pose_vis[0], gt_pose_depths * pose_vis[0], reduction="sum") / (torch.sum(pose_vis[0]) + 1e-8)
            loss_joint = F.smooth_l1_loss(pred_joint_depths * joint_vis[0], gt_joint_depths * joint_vis[0], reduction="sum") / (torch.sum(joint_vis[0]) + 1e-8)
            loss = {
                "pose": loss_pose,
                "joint": loss_joint,
                "total": loss_pose + loss_joint,
            }
        else:
            loss = None

        # === merge pose depth and joint depth
        merged_depths = pred_pose_depths.reshape(batch_size, num_persons, 1) + pred_joint_depths  # [B, Np, Nj]

        output["pred_depths"] = merged_depths  # [B, Np, Nj]

        return output, loss


def get_model(cfg):
    model = MultiViewMultiPersonPoseNet(cfg)
    return model
