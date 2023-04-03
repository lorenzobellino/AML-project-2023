import os
import wandb
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data

from torchvision.io import read_image
from torch.utils.data import DataLoader
from torch.backends import cudnn


from utils.utils import setup_transform
from config.baseline import *
from utils.wandb_setup import setup as wb_setup
from bisenetV2.bisenetv2 import BiSeNetV2
from utils.utils import *
from dataset.cityscapes import Cityscapes


def train(model, dataloader, logger):
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    parameters_to_optimize = model.parameters()
    optimizer = optim.SGD(
        parameters_to_optimize, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
    model = model.to(DEVICE)

    cudnn.benchmark  # Calling this optimizes runtime

    epochs = []

    wandb.watch(model, log="all")

    current_step = 0
    # Start iterating over the epochs
    for epoch in range(NUM_EPOCHS):
        logger.info("Starting epoch {}/{}".format(epoch + 1, NUM_EPOCHS))
        epochs.append(epoch + 1)

        # Iterate over the dataset
        for images, labels in dataloader:
            images = images.to(DEVICE, dtype=torch.float32)
            labels = labels.to(DEVICE, dtype=torch.long)
            model.train()
            optimizer.zero_grad()

            predictions = model(images)
            loss = criterion(predictions, labels.squeeze())

            # Log loss
            if current_step % LOG_FREQUENCY == 0:
                logger.info("Step {}, Loss {}".format(current_step, loss.item()))
                wandb.log({"train/loss": loss})
            # Compute gradients for each layer and update weights
            loss.backward()
            optimizer.step()

            current_step += 1


def create_train_dataloader(transforms):
    if PARTITION == "A":
        train_dataset = Cityscapes(
            root=ROOT_DIR + "data/Cityscapes/",
            transform=transforms,
            cl19=True,
            filename="train_A.txt",
        )
    elif PARTITION == "B":
        train_dataset = Cityscapes(
            root=ROOT_DIR + "data/Citytscapes/",
            transform=transforms,
            cl19=True,
            filename="train_b.txt",
        )
    else:
        raise NotImplementedError

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    return train_dataloader


def create_val_dataloader(transforms):
    if PARTITION == "A":
        val_dataset = Cityscapes(
            root=ROOT_DIR + "data/Cityscapes/",
            transform=transforms,
            cl19=True,
            filename="test_A.txt",
        )
    elif PARTITION == "B":
        val_dataset = Cityscapes(
            root=ROOT_DIR + "data/Cityscapes/",
            transform=transforms,
            cl19=True,
            filename="test_B.txt",
        )
    else:
        raise NotImplementedError

    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True
    )

    return val_dataloader


def validation(model, dataloader, logger):
    miou = compute_moiu(net=model, val_dataloader=dataloader)
    logger.info("Validation MIoU: {}".format(miou))
    wandb.log({"val/miou": miou})
    wandb.finish()
    logger.info("validation plot : ")
    validation_plot(net=model, val_dataloader=dataloader, n_image=20)
    torch.cuda.empty_cache()


def save_model(model, logger):
    name = f"step2_{PARTITION}_model.pth"
    if not os.path.exists(ROOT_DIR + "models/STEP2/"):
        logger.info("creating models directory")
        os.makedirs(ROOT_DIR + "models/STEP2/")
    torch.save(model.state_dict(), ROOT_DIR + "models/STEP2/" + name)


def main(args, logger):
    logger.info("centralized baseline main")
    random.seed(SEED)
    np.random.seed(SEED)
    logger.info("setting up transforms ... ")
    transforms = setup_transform()

    logger.info("choosing dataset and creating dataloader ... ")
    train_dataloader = create_train_dataloader(transforms)

    logger.info("setup for wandb")
    wb_setup(step=2)

    logger.info("start the training loop")
    model = BiSeNetV2(NUM_CLASSES, output_aux=False, pretrained=True)

    train(model, train_dataloader, logger)

    logger.info("saving model")
    save_model(model, logger)

    logger.info("creating validation dataloader")
    val_dataloader = create_val_dataloader(transforms)

    validation(model, val_dataloader, logger)
