import random
import wandb
import torch
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data
from torch.backends import cudnn

from torch.utils.data import DataLoader

from dataset.GTA import GTA5
from dataset.cityscapes import Cityscapes

# from config.baseline import *
from config.ffreda import *
from utils.wandb_setup import setup as wb_setup
from utils.utils import *
from bisenetV2.bisenetv2 import BiSeNetV2

# from .styleaugment import StyleAugment

from .styleaugment import StyleAugment


def train(model, dataloader, miou_dataloader, logger):
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

    best_miou = 0

    # Start iterating over the epochs
    for epoch in range(NUM_EPOCHS):
        logger.info(f"Starting epoch {epoch + 1}/{NUM_EPOCHS}")
        epochs.append(epoch + 1)
        for images, labels in dataloader:
            images = images.to(DEVICE, dtype=torch.float32)
            labels = labels.to(DEVICE, dtype=torch.long)
            model.train()
            optimizer.zero_grad()

            predictions = model(images)
            loss = criterion(predictions, labels.squeeze())

            # Log loss
            if current_step % LOG_FREQUENCY == 0:
                logger.info(f"Step {current_step}, Loss {loss.item()}")
                wandb.log({"train/loss": loss})
            # Compute gradients for each layer and update weights
            loss.backward()
            optimizer.step()

            current_step += 1
        miou = compute_miou(model, miou_dataloader)
        if miou > best_miou:
            best_miou = miou
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "miou": miou,
                },
                CKPT_PATH + f"step4_pretraining_{miou:.5}.pth",
            )


def validation(model, dataloader, logger):
    torch.cuda.empty_cache()
    miou = compute_miou(net=model, val_dataloader=dataloader)
    logger.info("Validation MIoU: {}".format(miou))
    wandb.log({"val/miou": miou})
    wandb.finish()
    logger.info("validation plot : ")
    validation_plot(net=model, val_dataloader=dataloader, n_image=20)
    torch.cuda.empty_cache()


def create_train_dataloader(transforms):
    train_dataset = GTA5(root=GTA5_ROOT, transform=transforms)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    return train_dataloader


def create_miou_dataloader(transforms):
    miou_dataset = Cityscapes(
        root=CTSC_ROOT,
        transform=transforms,
        cl19=True,
        filename="uniformA.json",
        id_client=0,
    )
    miou_dataloader = DataLoader(
        miou_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=True,
    )
    return miou_dataloader


def create_val_dataloader(transforms):
    val_dataset = Cityscapes(
        root=CTSC_ROOT,
        transform=transforms,
        cl19=True,
        filename="test_B.txt",
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE - 2,
        shuffle=False,
        num_workers=4,
        drop_last=True,
    )
    return val_dataloader


def train_FDA(dataloader, miou_dataloader, logger, args):
    logger.info("crearting the model")
    model = BiSeNetV2(NUM_CLASSES, output_aux=False, pretrained=True)
    criterion = nn.CrossEntropyLoss()
    parameters_to_optimize = model.parameters()
    optimizer = optim.SGD(
        parameters_to_optimize, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )

    ckpt_epoch = 0
    current_step = 0
    ckpt_miou = 0

    if args.load is not None:
        logger.info("loading checkpoint")
        checkpoint = torch.load(os.path.join(CKPT_DIR, args.load))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        ckpt_epoch = checkpoint["epoch"]
        # ckpt_loss = checkpoint["loss"]
        ckpt_miou = checkpoint["miou"]

    model = model.to(DEVICE)

    for epoch in range(ckpt_epoch, NUM_EPOCHS):
        logger.info(f"Starting epoch {epoch + 1}/{NUM_EPOCHS}")
        # epochs.append(epoch + 1)

        for images, labels in dataloader:
            images = images.to(DEVICE, dtype=torch.float32)
            labels = labels.to(DEVICE, dtype=torch.long)
            model.train()
            optimizer.zero_grad()

            predictions = model(images)
            loss = criterion(predictions, labels.squeeze())

            # Log loss
            if current_step % LOG_FREQUENCY == 0:
                logger.info(f"Step {current_step}, Loss {loss.item()}")

                wandb.log({"train/loss": loss})
            # Compute gradients for each layer and update weights
            loss.backward()
            optimizer.step()

            current_step += 1

        miou = compute_miou(model, miou_dataloader)
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "miou": miou,
            },
            CKPT_PATH + f"step4_FDA_{miou:.5}.pth",
        )

        if miou > ckpt_miou:
            ckpt_miou = miou
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "miou": miou,
                },
                CKPT_PATH + f"step4_FDA_bestckpt.pth",
            )


def main(args, logger):
    logger.info("ffreda setting main")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    if args.pretrain:
        logger.info("pre-training phase")
        logger.info("setting up transforms ... ")
        transforms = setup_transform()

        logger.info("choosing dataset and creating dataloader ... ")
        train_dataloader = create_train_dataloader(transforms)

        logger.info("creating miou validation dataloader ...")
        miou_dataloader = create_miou_dataloader(transforms)

        logger.info("setup for wandb")
        wb_setup(step=4, args=args)

        if args.load is not None:
            logger.info("loading model")
            model = BiSeNetV2(NUM_CLASSES, output_aux=False, pretrained=False)
            model.load_state_dict(torch.load("models/" + args.load))
        else:
            logger.info("creating model ... ")
            model = BiSeNetV2(NUM_CLASSES, output_aux=False, pretrained=True)

        logger.info("start the training loop")
        train(model, train_dataloader, miou_dataloader, logger)

        logger.info("creating validation dataloader")
        val_dataloader = create_val_dataloader(transforms)

        validation(model, val_dataloader, logger)

    elif args.FDA:
        logger.info("Training with FDA")
        logger.info("setting up transforms ... ")
        transforms = setup_transform()

        logger.info("Extracting styles from Cytiscapes ...")
        SA = StyleAugment(
            n_images_per_style=MAX_SAMPLE_PER_CLIENT, L=0.01, size=(1024, 512), b=1
        )
        clients = random.sample(range(TOT_CLIENTS), N_STYLE)

        for c in clients:
            client_dataset = Cityscapes(
                root=CTSC_ROOT,
                transform=transforms,
                cl19=True,
                filename="uniformA.json",
                id_client=c,
            )
            SA.add_style(client_dataset)

        logger.info("creating train dataloader ... ")
        train_dataset = GTA5(root=GTA5_ROOT, transform=transforms)
        logger.info("applying the styles to the dataset ...")
        train_dataset.set_style_tf_fn(SA.apply_style)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            drop_last=True,
        )

        logger.info("creating miou validation dataloader ...")
        miou_dataloader = create_miou_dataloader(transforms)

        logger.info("creating a checkpoint folder")
        if not os.path.exists(CKPT_DIR):
            os.makedirs(CKPT_DIR)

        logger.info("setup for wandb")
        wb_setup(step=4, args=args)

        logger.info("start the training loop")
        train_FDA(train_dataloader, miou_dataloader, logger, args)

        logger.info("creating validation dataloader")
        id_client = random.randint(range(TOT_CLIENTS))
        val_dataset = Cityscapes(
            root=CTSC_ROOT, transform=transforms, cl19=True, id_client=id_client
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            drop_last=True,
        )

        logger.info("loading the best checkpoint")
        try:
            model = BiSeNetV2(NUM_CLASSES, output_aux=False, pretrained=True)
            best_ckpt = torch.load(CKPT_PATH + "step4_FDA_bestckpt.pth")
            model.load_state_dict(best_ckpt["model_state_dict"])
            model.eval()
        except:
            logger.info("No checkpoint found")
            raise Exception("No checkpoint found")

        logger.info("start the validation")
        validation(model, val_dataloader, logger)
    else:
        logger.error(
            "No phase selected: select between -p and -FDA for pretraining phase or FDA"
        )
        raise Exception("Select a phase to cmpute")
