import random
import os
import json
import math
import numpy as np
import torch.utils.data as torch_data

from torchvision.io import read_image
from torch import from_numpy

from config.baseline import *
from config.transform import *
from config.federated import *
from utils.wandb_setup import setup as wb_setup
from utils.utils import setup_transform
from dataset.cityscapes import Cityscapes


def main(args):
    print("centralized baseline main")
    random.seed(SEED)
    np.random.seed(SEED)
    print("setting up transforms ... ")
    transforms = setup_transform()

    # print("generating the datasets")
    # generate_splits()
