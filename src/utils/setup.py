import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data
import numpy as np
import matplotlib.pyplot as plt
import random
import torchvision
import logging
import warnings
import math
import json
import wandb

from torchvision import transforms
from torch.backends import cudnn
from torch import from_numpy
from PIL import Image
from torch.utils.data import Subset, DataLoader
from torchvision.io import read_image  # importare solo se si usa nella classe Dataset
from torchmetrics.classification import MulticlassJaccardIndex

# ignore/don't display warnings in Python
warnings.resetwarnings()
warnings.simplefilter("ignore")
# warnings.simplefilter('ignore', FutureWarning) FutureWarning, DeprecationWarning, SyntaxWarning, RuntimeWarning...
# from bisenetV2.bisenetv2 import BiSeNetV2


def setup():
    print("setup")
