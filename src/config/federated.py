PARTITION = "B"  # 'A' or 'B'
SPLIT = 1  # 1 or 2 // 1 = Uniform : 2 = Heterogenous
MAX_SAMPLE_PER_CLIENT = 20

IMAGES_FINAL = "leftImg8bit"
TARGET_FINAL = "gtFine_labelIds"

N_ROUND = 50
CLIENT_PER_ROUND = 5  # clients picked each round
NUM_EPOCHS = 2


CHECKPOINTS = 5

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
