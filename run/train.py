import argparse
import pprint

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import _init_paths
from core.config import config, update_config
from core.function import train_3d, validate_3d
from utils.utils import create_logger, save_checkpoint
import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", help="experiment configure file name", required=True, type=str)

    args, _ = parser.parse_known_args()
    update_config(args.cfg)

    return args


def get_optimizer(model):
    lr = config.TRAIN.LR
    optimizer = optim.Adam(model.module.parameters(), lr=lr)

    return model, optimizer


def main():
    args = parse_args()
    logger, final_output_dir, time_str = create_logger(config, args.cfg, "train")

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    gpus = [int(i) for i in config.GPUS.split(',')]

    print("=> Loading data..")
    train_dataset = eval("dataset." + config.DATASET.TRAIN_DATASET)(
        config, config.DATASET.TRAIN_SUBSET, True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True)

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

    model, optimizer = get_optimizer(model)

    start_epoch = config.TRAIN.BEGIN_EPOCH
    end_epoch = config.TRAIN.END_EPOCH

    best_result = None

    print("=> Training...")
    for epoch in range(start_epoch, end_epoch):
        print("Epoch {}".format(epoch))

        train_3d(config, model, optimizer, train_loader, epoch, final_output_dir)
        result = validate_3d(config, model, test_loader, epoch, final_output_dir)

        best_model = False
        if best_result is None:
            best_result = result
        else:
            if config.DATASET.TEST_DATASET == "panoptic":
                if result < best_result:
                    best_result = result
                    best_model = True
            else:
                if result > best_result:
                    best_result = result
                    best_model = True

        logger.info("=> saving checkpoint to {} (Best: {})".format(final_output_dir, best_model))
        save_checkpoint({
            "epoch": epoch,
            "state_dict": model.module.state_dict(),
            "result": best_result,
            "optimizer": optimizer.state_dict(),
        }, best_model, final_output_dir, time_str=time_str)


if __name__ == "__main__":
    main()
