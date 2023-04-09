import random
import numpy as np
import torch

from config.pseudolabels import *


def main(args, logger):
    logger.info("pseudo-labels main")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
