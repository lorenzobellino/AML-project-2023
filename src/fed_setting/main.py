import random
import os
import json
import math
import torch
import wandb
import numpy as np


from torchvision.io import read_image
from torch import from_numpy
from torch.utils.data import DataLoader

from config.baseline import *
from config.transform import *
from config.federated import *
from utils.wandb_setup import setup as wb_setup
from utils.utils import setup_transform, compute_miou, validation_plot
from dataset.cityscapes import Cityscapes
from bisenetV2.bisenetv2 import BiSeNetV2
from client import Client
from server import Server


def setup_clients(n_client, model, transforms):
    clients = []
    if PARTITION == "A":
        if SPLIT == 1:
            filename = "uniformA.json"
        else:
            filename = "heterogeneuosA.json"
        for i in range(n_client):
            train_dataset = Cityscapes(
                root=ROOT_DIR,
                transform=transforms,
                cl19=cl19,
                filename=filename,
                id_client=i,
            )
            client = Client(client_id=i, dataset=train_dataset, model=model)
            clients.append(client)
    else:
        if SPLIT == 1:
            filename = "uniformB.json"
        else:
            filename = "heterogeneuosB.json"
        for i in range(n_client):
            train_dataset = Cityscapes(
                root=ROOT_DIR,
                transform=transforms,
                cl19=cl19,
                filename=filename,
                id_client=i,
            )
            client = Client(client_id=i, dataset=train_dataset, model=model)
            clients.append(client)

    return clients


def create_round_val_dataloader(transforms):
    if PARTITION == "A":
        if SPLIT == 1:
            filename = "uniformA.json"
        else:
            filename = "heterogeneuosA.json"
    else:
        if SPLIT == 1:
            filename = "uniformB.json"
        else:
            filename = "heterogeneuosB.json"
    val_dataset = Cityscapes(
        root=ROOT_DIR,
        transform=transforms,
        cl19=cl19,
        filename=filename,
        id_client=0,
        train=False,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True
    )
    return val_dataloader


def train(model, server, train_clients, val_dataloader, logger):
    for r in range(N_ROUND):
        logger.info(f"ROUND {r + 1}/{N_ROUND}: Training {CLIENT_PER_ROUND} Clients...")
        server.select_clients(r, train_clients, num_clients=CLIENT_PER_ROUND)
        server.train_round()
        server.update_model()
        miou = compute_miou(net=server.model, val_dataloader=val_dataloader)
        wandb.log({"server/miou": miou})
        logger.info(f"Validation MIoU: {miou}")
        if r % CHECKPOINTS == 0:
            logger.info(f"Saving the model")
            torch.save(
                model.state_dict(),
                ROOT_DIR
                + "models/STEP3/"
                + f"model_P{PARTITION}_S{SPLIT}_round{r:02}.pth",
            )


def create_val_dataloader(transforms):
    if PARTITION == "A":
        val_dataset = Cityscapes(
            root=ROOT_DIR, transform=transforms, cl19=cl19, filename="test_A.txt"
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            drop_last=True,
        )
    elif PARTITION == "B":
        val_dataset = Cityscapes(
            root=ROOT_DIR, transform=transforms, cl19=cl19, filename="test_B.txt"
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            drop_last=True,
        )
    return val_dataloader


def validation(model, val_dataloader, logger):
    miou = compute_miou(net=model, val_dataloader=val_dataloader)
    logger.info("Validation MIoU: {}".format(miou))
    wandb.log({"val/miou": miou})
    wandb.finish()
    logger.info("validation plot : ")
    validation_plot(net=model, val_dataloader=val_dataloader, n_image=20)
    torch.cuda.empty_cache()


def main(args, logger):
    logger.info("centralized baseline main")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    logger.info("setting up transforms ... ")
    transforms = setup_transform()

    if args.load is not None:
        logger.info("loading the model ... ")
        model = BiSeNetV2(NUM_CLASSES, output_aux=False, pretrained=False)
        model.load_state_dict(torch.load("models/" + args.load))
        model.eval()
    else:
        logger.info("loading the model ... ")
        model = BiSeNetV2(NUM_CLASSES, output_aux=False, pretrained=True)
    model = model.to(DEVICE)

    # logger.info("generating the datasets")
    # generate_splits()
    logger.info("setting up the clients ... ")
    train_clients = setup_clients(
        n_clients=TOT_CLIENTS, model=model, transforms=transforms
    )
    round_val_dataloader = create_round_val_dataloader(transforms)

    logger.info("setting up the server ... ")
    server = Server(model, lr=LR, momentum=MOMENTUM)

    logger.info("setting up wandb ... ")
    wb_setup(step=3)

    logger.info("start the training loop")
    train(model, server, train_clients, round_val_dataloader, logger)

    logger.info("training completed")
    logger.info("saving the model")
    torch.save(
        model.state_dict(),
        ROOT_DIR + "models/STEP3/" + f"model_P{PARTITION}_S{SPLIT}.pth",
    )

    logger.info("creating validation dataloader")
    val_dataloader = create_val_dataloader(transforms)

    logger.info("Validation step")
    validation(server.model, val_dataloader, logger)
