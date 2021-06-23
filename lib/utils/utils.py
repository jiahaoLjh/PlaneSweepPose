import os
import time
import logging
from pathlib import Path

import torch

from core.config import get_model_name


def create_logger(cfg, cfg_name, phase="train"):
    this_dir = Path(os.path.dirname(__file__))
    root_output_dir = (this_dir / ".." / ".." / cfg.OUTPUT_DIR).resolve()

    # === logger folder
    if not root_output_dir.exists():
        print("=> creating {}".format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.TRAIN_DATASET
    model = get_model_name(cfg)
    cfg_name = os.path.basename(cfg_name).split(".")[0]

    final_output_dir = root_output_dir / dataset / model / cfg_name

    print("=> creating output folder {}".format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # === logger
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
    log_file = "{}_{}_{}.log".format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    print("=> creating logging file {}".format(final_log_file))
    head = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)

    return logger, str(final_output_dir), time_str


def load_checkpoint(model, optimizer, output_dir, filename="checkpoint.pth.tar"):
    f = os.path.join(output_dir, filename)
    if os.path.isfile(f):
        checkpoint = torch.load(f)
        if "state_dict" in checkpoint:
            model.module.load_state_dict(checkpoint["state_dict"])
        else:
            model.module.load_state_dict(checkpoint)
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        print("=> load checkpoint {}".format(f))

        return model, optimizer

    else:
        print("=> no checkpoint found at {}".format(f))
        return model, optimizer


def save_checkpoint(states, is_best, output_dir, time_str):
    torch.save(states, os.path.join(output_dir, "checkpoint_{}.pth.tar".format(time_str)))
    if is_best and "state_dict" in states:
        torch.save(states["state_dict"], os.path.join(output_dir, "model_best_{}.pth.tar".format(time_str)))
