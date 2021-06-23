import time
import logging

import numpy as np
import torch


logger = logging.getLogger(__name__)


def train_3d(config, model, optimizer, loader, epoch, output_dir):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_pose = AverageMeter()
    losses_joint = AverageMeter()

    model.train()

    end = time.time()

    for i, batch_data in enumerate(loader):
        if i >= config.TRAIN.STEP_PER_EPOCH:
            break

        kpts, pose_vis, joint_vis, pose_depths, joint_depths, meta = batch_data
        data_time.update(time.time() - end)

        # === forward pass
        output_dict, loss_dict = model(kpts=kpts, pose_vis=pose_vis, joint_vis=joint_vis, gt_pose_depths=pose_depths, gt_joint_depths=joint_depths, meta=meta)

        loss = loss_dict["total"]
        loss_pose = loss_dict["pose"]
        loss_joint = loss_dict["joint"]
        losses.update(loss.item())
        losses_pose.update(loss_pose.item())
        losses_joint.update(loss_joint.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % config.PRINT_FREQ == 0 or i == len(loader) - 1:
            msg = "Epoch: [{0}][{1}/{2}]\t" \
                  "Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t" \
                  "Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t" \
                  "Loss: {loss.val:.6f} ({loss.avg:.6f})\t" \
                  "Loss_pose: {loss_pose.val:.7f} ({loss_pose.avg:.7f})\t" \
                  "Loss_joint: {loss_joint.val:.7f} ({loss_joint.avg:.7f})".format(
                      epoch, i + 1, min(len(loader), config.TRAIN.STEP_PER_EPOCH),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses, loss_pose=losses_pose, loss_joint=losses_joint)
            logger.info(msg)


def validate_3d(config, model, loader, epoch, output_dir):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.eval()

    preds = []
    confs = []
    with torch.no_grad():
        end = time.time()

        for i, batch_data in enumerate(loader):
            kpts, pose_vis, joint_vis, pose_depths, joint_depths, meta = batch_data
            data_time.update(time.time() - end)

            # === forward pass
            output_dict, _ = model(kpts=kpts, pose_vis=pose_vis, joint_vis=joint_vis, gt_pose_depths=pose_depths, gt_joint_depths=joint_depths, meta=meta)

            pred = output_dict["pred_depths"].detach().cpu().numpy()  # [B, Np, Nj]
            conf = output_dict["joint_depth_volume"].detach().cpu().numpy()  # [B, Np, Nj, Nrd]
            conf = np.max(conf, axis=-1)  # [B, Np, Nj]
            preds.append(pred)
            confs.append(conf)

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % config.PRINT_FREQ == 0 or i == len(loader) - 1:
                msg = "Test: [{0}][{1}/{2}]\t" \
                      "Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t" \
                      "Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)".format(
                          epoch, i + 1, len(loader),
                          batch_time=batch_time,
                          data_time=data_time)
                logger.info(msg)

    preds = np.concatenate(preds, axis=0)  # [N, Np, Nj]
    confs = np.concatenate(confs, axis=0)  # [N, Np, Nj]

    result = loader.dataset.evaluate(preds, confs)

    return result


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
