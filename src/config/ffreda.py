SEED = 42
NUM_CLASSES = 19
DEVICE = "cuda"
LR = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 100
LOG_FREQUENCY = 10

BATCH_SIZE = 8

N_STYLE = 10

PARTITION = "A"
SPLIT = 1

N_ROUND = 50
CLIENT_PER_ROUND = 5  # clients picked each round
NUM_EPOCHS = 2

if PARTITION == "A":
    if SPLIT == 1:
        TOT_CLIENTS = 36
    else:
        TOT_CLIENTS = 46
else:
    if SPLIT == 1:
        TOT_CLIENTS = 25
    else:
        TOT_CLIENTS = 33

MAX_SAMPLE_PER_CLIENT = 20

GTA5_ROOT = "./data/GTA5/"
CTSC_ROOT = "./data/Cityscapes/"

CKPT_PATH = "./checkpoints/"
CKPT_DIR = "./checkpoints/"
