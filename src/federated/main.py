import os
import wandb
import torch
import random
import numpy as np
from torch.utils.data import DataLoader


from config.config_options import *
from utils.utils import setup_transform, setup_wandb, compute_miou, validation_plot

from datasets.GTA import GTA5
from datasets.cityscapes import Cityscapes
from networks.bisenetv2 import BiSeNetV2
from networks.deeplabv3 import deeplabv3_mobilenetv2
from .client import Client
from .server import Server


def create_miou_dataloader(transforms, logger):
    if SPLIT == 1:
        filename = f"uniform{PARTITION}.json"
    else:
        filename = f"heterogeneuos{PARTITION}.json"

    miou_dataset = Cityscapes(
        root=CTSC_ROOT,
        transform=transforms,
        cl19=cl19,
        filename=filename,
        id_client=random.randint(0, TOT_CLIENTS - 1),
    )
    miou_dataloader = DataLoader(
        miou_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True
    )

    return miou_dataloader


def setup_clients(n_client, model, transforms, logger):
    clients = []
    if SPLIT == 1:
        filename = f"uniform{PARTITION}.json"
    else:
        filename = f"heterogeneuos{PARTITION}.json"

    for i in range(n_client):
        train_dataset = Cityscapes(
            root=CTSC_ROOT,
            transform=transforms,
            cl19=cl19,
            filename=filename,
            id_client=i,
        )
        client = Client(
            client_id=i,
            dataset=train_dataset,
            model=model,
            pseudo_lab=PSEUDO_LAB,
            teacher_model=model,
        )
        clients.append(client)

    return clients


def training_loop(args, logger, server, train_clients, miou_dataloader):
    if args.step == 5:
        model_base_name = f"step{args.step}{'_FDA' if FDA else ''}_T{T_ROUND}"
    else:
        model_base_name = f"step{args.step}"
    for r in range(N_ROUND):
        logger.info(f"ROUND {r + 1}/{N_ROUND}: Training {CLIENT_PER_ROUND} Clients...")
        if args.step == 5 and r % T_ROUND == 0:
            logger.info("Teacher model updated")
            server.set_clients_teacher(train_clients)
        server.select_clients(r, train_clients, num_clients=CLIENT_PER_ROUND)
        server.train_round()
        server.update_model()

        miou = compute_miou(args=args, net=server.model, val_dataloader=miou_dataloader)
        wandb.log({"server/miou": miou})
        logger.info(f"Validation MIoU: {miou}")

        if r % CHECKPOINTS == 0:
            logger.info(f"Saving the model")
            torch.save(
                server.model.state_dict(),
                CKPT_DIR + model_base_name + f"_P{PARTITION}_S{SPLIT}_round{r:02}.pth",
            )


def main(args, logger):
    logger.debug("Starting the main function")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    logger.info("Setting up WandB")
    setup_wandb(args, logger)

    logger.info("Setting u the transforms")
    transforms = setup_transform()

    logger.info("Setting up the model")
    if args.network == "bisenet":
        model = BiSeNetV2(n_classes=NUM_CLASSES, output_aux=False, pretrained=False)
    elif args.network == "mobilenet":
        model = deeplabv3_mobilenetv2(n_classes=NUM_CLASSES)

    if LOAD_CKPT:
        logger.info("Loading a previous checkpoint")
        if LOAD_CKPT_PATH in os.listdir(CKPT_DIR):
            checkpoint = torch.load(os.path.join(CKPT_DIR, LOAD_CKPT_PATH))
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            logger.error("The checkpoint does not exist")

    model.eval()
    model.to(DEVICE)

    logger.info("Setting up the clients")
    clients = setup_clients(TOT_CLIENTS, model, transforms, logger)

    logger.info("Setting up the server")
    server = Server(model, lr=LR, momentum=MOMENTUM)

    logger.info("Creating the MIOU dataloader")
    miou_dataloader = create_miou_dataloader(transforms, logger)

    logger.info("Starting the training")
    training_loop(args, logger, server, clients, miou_dataloader)

    logger.info("Training completed")
    logger.info("Saving the model")
    torch.save(
        server.model.state_dict(),
        "./models/" + f"model{args.step}_final.pth",
    )

    logger.info("Creating the validation dataloader")
    val_dataset = Cityscapes(
        root=CTSC_ROOT,
        transform=transforms,
        cl19=cl19,
        filename=f"test_{PARTITION}.txt",
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True
    )

    logger.info("Validation step")
    logger.info("Computing the MIOU")
    miou = compute_miou(args=args, net=server.model, val_dataloader=val_dataloader)
    wandb.log({"server/miou": miou})
    wandb.finish()
    logger.info(f"Validation MIoU: {miou}")

    logger.info("Creating the validation plot")
    validation_plot(
        args=args, net=server.model, val_dataloader=val_dataloader, n_images=5
    )
