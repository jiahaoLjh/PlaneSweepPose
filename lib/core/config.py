import yaml

import numpy as np
from easydict import EasyDict as edict


config = edict()

# === base configs
config.OUTPUT_DIR = "output"
config.MODEL = ""
config.GPUS = "0"
config.WORKERS = 8
config.PRINT_FREQ = 100

# === cudnn configs
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# === network configs
config.NETWORK = edict()
config.NETWORK.NUM_JOINTS = 17
config.NETWORK.HIDDEN_SIZE = 1024
config.NETWORK.SOFTARGMAX_KERNEL_SIZE = 0

# === datasets configs
config.DATASET = edict()
config.DATASET.ROOT = ""
config.DATASET.TRAIN_DATASET = ""
config.DATASET.TRAIN_SUBSET = ""
config.DATASET.TEST_DATASET = ""
config.DATASET.TEST_SUBSET = ""

config.DATASET.DATA_AUGMENTATION = True
config.DATASET.PERTURB_SIGMA = 1.0

config.DATASET.SYNTHESIS_X_RANGE = np.array([-1000, 1000])
config.DATASET.SYNTHESIS_Y_RANGE = np.array([-1000, 1000])

# === train configs
config.TRAIN = edict()
config.TRAIN.LR = 0.001

config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 100
config.TRAIN.STEP_PER_EPOCH = 3000

config.TRAIN.BATCH_SIZE = 8
config.TRAIN.SHUFFLE = True

# === validation configs
config.TEST = edict()
config.TEST.BATCH_SIZE = 8
config.TEST.USE_PRED_CONFIDENCE = True
config.TEST.NMS_THRESHOLD = 100

# === multi-person configs
config.MULTI_PERSON = edict()
config.MULTI_PERSON.MAX_NUM_PERSONS = 10

config.MULTI_PERSON.POSE_MIN_DEPTH = 1.0
config.MULTI_PERSON.POSE_MAX_DEPTH = 10.0
config.MULTI_PERSON.POSE_NUM_DEPTH_LAYERS = 64
config.MULTI_PERSON.POSE_SIGMA = 10

config.MULTI_PERSON.JOINT_MIN_DEPTH = -1000.0
config.MULTI_PERSON.JOINT_MAX_DEPTH = 1000.0
config.MULTI_PERSON.JOINT_NUM_DEPTH_LAYERS = 64
config.MULTI_PERSON.JOINT_SIGMA = 10


def _update_dict(k, v):
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))


def get_model_name(cfg):
    name = "{model}".format(
        model=cfg.MODEL,
    )

    return name
