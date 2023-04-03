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
from utils.utils import setup_transform, compute_moiu, validation_plot
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


def train(model, server, train_clients, val_dataloader):
    for r in range(N_ROUND):
        print(f"ROUND {r + 1}/{N_ROUND}: Training {CLIENT_PER_ROUND} Clients...")
        server.select_clients(r, train_clients, num_clients=CLIENT_PER_ROUND)
        server.train_round()
        server.update_model()
        miou = compute_moiu(net=server.model, val_dataloader=val_dataloader)
        wandb.log({"server/miou": miou})
        print(f"Validation MIoU: {miou}")
        if r % CHECKPOINTS == 0:
            print(f"Saving the model")
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


def validation(model, val_dataloader):
    miou = compute_moiu(net=model, val_dataloader=val_dataloader)
    print("Validation MIoU: {}".format(miou))
    wandb.log({"val/miou": miou})
    wandb.finish()
    print("validation plot : ")
    validation_plot(net=model, val_dataloader=val_dataloader, n_image=20)
    torch.cuda.empty_cache()


def main(args):
    print("centralized baseline main")
    random.seed(SEED)
    np.random.seed(SEED)
    print("setting up transforms ... ")
    transforms = setup_transform()

    print("loading the model ... ")
    model = BiSeNetV2(NUM_CLASSES, output_aux=False, pretrained=True)
    model = model.to(DEVICE)
    # print("generating the datasets")
    # generate_splits()
    print("setting up the clients ... ")
    train_clients = setup_clients(
        n_clients=TOT_CLIENTS, model=model, transforms=transforms
    )
    round_val_dataloader = create_round_val_dataloader(transforms)

    print("setting up the server ... ")
    server = Server(model, lr=LR, momentum=MOMENTUM)

    print("start the training loop")
    train(model, server, train_clients, round_val_dataloader)

    print("training completed")
    print("saving the model")
    torch.save(
        model.state_dict(),
        ROOT_DIR + "models/STEP3/" + f"model_P{PARTITION}_S{SPLIT}.pth",
    )

    print("creating validation dataloader")
    val_dataloader = create_val_dataloader(transforms)

    print("Validation step")
    validation(server.model, val_dataloader)
