# STEP 2-3-4-5:
PARTITION = "A"
BATCH_SIZE = 2
NUM_EPOCHS = 1
LR = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

# STEP 3:
SPLIT = 1  # 1=uniform, 2=heterogeneous
N_ROUND = 2  # number of rounds
CLIENT_PER_ROUND = 2  # clients picked each round
MAX_SAMPLE_PER_CLIENT = 20

# STEP 4:
FDA = True
N_STYLE = 1  # number of styles to extract
BETA_WINDOW_SIZE = 0  # 0 = 1x1 1 = 3x3 2 = 5x5


# STEP 5:
LOAD_CKPT = True  # load a checkpoint
T_ROUND = 1  # model update frequency
PSEUDO_LAB = True
LOAD_CKPT_PATH = "step4_A_S1_FDA_0.08.pth"  # path to the checkpoint to load


# logging and checkpoints options
LOG_FREQUENCY = 20  # log every 20 batches
LOG_FREQUENCY_EPOCH = 3  # log every 3 epochs for the clients
CHECKPOINTS = 10  # after how many epochs save a checkpoint

# directories
CKPT_DIR = "./checkpoints/"


CTSC_ROOT = "./data/Cityscapes/"
GTA5_ROOT = "./data/GTA5/"

DEVICE = "cuda"
SEED = 42

NUM_CLASSES = 19
cl19 = True

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
