PARTITION = "A"
SPLIT = 1

BATCH_SIZE = 2
NUM_EPOCHS = 1
LR = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

LOG_FREQUENCY = 20
LOG_FREQUENCY_EPOCH = 3

MAX_SAMPLE_PER_CLIENT = 20

FDA = True
N_STYLE = 1
BETA_WINDOW_SIZE = 0

T_ROUND = 1
PSEUDO_LAB = True

N_ROUND = 2
CLIENT_PER_ROUND = 2  # clients picked each round

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

CHECKPOINTS = 5
LOAD_CKPT_PATH = "step4_A_S1_FDA_0.08.pth"
CKPT_PATH = "./checkpoints/"
CKPT_DIR = "./checkpoints/"
LOAD_CKPT = True

CTSC_ROOT = "./data/Cityscapes/"
GTA5_ROOT = "./data/GTA5/"

DEVICE = "cuda"
SEED = 42

NUM_CLASSES = 19
cl19 = True
