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

from config.config_options import *
from utils.utils import setup_transform, setup_wandb, compute_miou, validation_plot

from datasets.GTA import GTA5
from datasets.cityscapes import Cityscapes
from networks.bisenetv2 import BiSeNetV2
from .styleaugment import StyleAugment


def setup_dataloader(args, transforms, logger):
    if args.dataset == "CTSC":
        train_dataset = Cityscapes(
            root=CTSC_ROOT,
            transform=transforms,
            cl19=True,
            filename=f"train_{PARTITION}.txt",
        )
    elif args.dataset == "GTA":
        train_dataset = GTA5(
            root=GTA5_ROOT,
            transform=transforms,
        )

        if FDA:
            logger.info("FDA enabled: extracting the styles from cityscapes")
            if SPLIT == 1:
                filename = f"uniform{PARTITION}.json"
            else:
                filename = f"heterogeneuos{PARTITION}.json"
            logger.info("extracting the styles from cityscapes")
            SA = StyleAugment(
                n_images_per_style=MAX_SAMPLE_PER_CLIENT,
                L=0.01,
                size=(1024, 512),
                b=BETA_WINDOW_SIZE,
            )
            clients = random.sample([_ for _ in range(TOT_CLIENTS)], N_STYLE)
            for c in clients:
                print(f"client {c}")
                client_dataset = Cityscapes(
                    root=CTSC_ROOT,
                    transform=transforms,
                    cl19=cl19,
                    filename=filename,
                    id_client=c,
                )
                SA.add_style(client_dataset)

            train_dataset.set_style_tf_fn(SA.apply_style)
    else:
        logger.error("Specify a valid dataset (CTSC or GTA)")
        exit()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    return train_dataloader


def training_loop(args, logger, model, train_dataloader, miou_dataloader):
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    parameters_to_optimize = model.parameters()
    optimizer = optim.SGD(
        parameters_to_optimize, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
    model = model.to(DEVICE)

    cudnn.benchmark  # Calling this optimizes runtime

    wandb.watch(model, log="all")

    current_step = 0
    best_miou = 0

    for epoch in range(NUM_EPOCHS):
        logger.info(f"Starting epoch {epoch + 1}/{NUM_EPOCHS}")
        for images, labels in train_dataloader:
            images = images.to(DEVICE, dtype=torch.float32)
            labels = labels.to(DEVICE, dtype=torch.long)
            model.train()
            optimizer.zero_grad()

            predictions = model(images)
            loss = criterion(predictions, labels.squeeze())

            # Log loss
            if current_step % LOG_FREQUENCY == 0:
                print(f"Step {current_step}, Loss {loss.item()}")
                wandb.log({"train/loss": loss})
                # Compute gradients for each layer and update weights
            loss.backward()
            optimizer.step()

            current_step += 1
        if miou_dataloader is not None:
            miou = compute_miou(model, miou_dataloader)
            wandb.log({"train/miou": miou})
            if miou > best_miou:
                logger.info(f"Saving the model with miou {miou:.2f} (best so far)")
                best_miou = miou
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
                        "miou": miou,
                    },
                    CKPT_PATH
                    + f"step{args.step}_{PARTITION}_S{SPLIT}{'_FDA' if FDA else ''}_{miou:.2f}.pth",
                )
    return best_miou


def main(args, logger):
    logger.debug("Starting the main function")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    logger.info("Setting u the transforms")
    transforms = setup_transform()

    logger.info("Setting up the dataloader")
    train_dataloader = setup_dataloader(args, transforms, logger)

    logger.info("Setting up WandB")
    setup_wandb(args, logger)

    logger.info("Setting up the model")
    model = BiSeNetV2(n_classes=NUM_CLASSES, output_aux=False, pretrained=False)

    if LOAD_CKPT:
        logger.info("Loading a previous checkpoint")
        try:
            model.load_state_dict(torch.load(os.path.join(LOAD_CKPT_PATH)))
        except:
            logger.error("Checkpoint not found")

    if args.dataset == "GTA":
        logger.info("Creating the MIOU dataloader")
        miou_filename = f"uniform{PARTITION}.json"
        miou_dataset = Cityscapes(
            root=CTSC_ROOT,
            transform=transforms,
            cl19=cl19,
            filename=miou_filename,
            id_client=random.randint(0, TOT_CLIENTS - 1),
        )
        miou_dataloader = DataLoader(
            miou_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            drop_last=True,
        )
    else:
        miou_dataloader = None

    logger.info("Starting the training loop")
    miou = training_loop(args, logger, model, train_dataloader, miou_dataloader)

    logger.info("Training finished")

    logger.info("Saving the model state dict")
    torch.save(model.state_dict(), f"./models/model_step{args.step}.pth")

    logger.info("Creating the Validation dataloader")
    val_dataset = Cityscapes(
        root=CTSC_ROOT,
        transform=transforms,
        cl19=cl19,
        filename=f"test_{PARTITION}.txt",
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True
    )

    torch.cuda.empty_cache()
    model = BiSeNetV2(NUM_CLASSES, output_aux=False, pretrained=True)
    logger.info("Extracting the best model ...")
    if args.dataset == "CTSC":
        model.load_state_dict(torch.load(f"./models/model_step{args.step}.pth"))
    else:
        ckpt = torch.load(
            CKPT_PATH
            + f"step4_{PARTITION}_S{SPLIT}{'_FDA' if FDA else ''}_{miou:.2f}.pth"
        )
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    model.to(DEVICE)

    logger.info("Validation step")
    logger.info("computing miou ...")
    miou = compute_miou(net=model, val_dataloader=val_dataloader)
    logger.info("Validation MIoU: {}".format(miou))
    wandb.log({"val/miou": miou})
    wandb.finish()

    logger.info("Creating sample results ...")
    validation_plot(net=model, val_dataloader=val_dataloader, n_images=5)
    torch.cuda.empty_cache()
