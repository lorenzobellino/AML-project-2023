import torch
import numpy as np
import os
import wandb
import matplotlib.pyplot as plt

from torchmetrics.classification import MulticlassJaccardIndex

import utils.transform as T

from config.config_transforms import *
from config.config_options import *

from config.config_options import DEVICE, NUM_CLASSES


colors = [
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [0, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
]

label_colours = dict(zip(range(NUM_CLASSES), colors))


def setup_transform():
    transformers = []
    if RANDOM_HORIZONTAL_FLIP is not None:
        transformers.append(T.RandomHorizontalFlip(RANDOM_HORIZONTAL_FLIP))
    if COLOR_JITTER is not None:
        transformers.append(T.ColorJitter(*COLOR_JITTER))
    if RANDOM_ROTATION is not None:
        transformers.append(T.RandomRotation(RANDOM_ROTATION))
    if RANDOM_CROP is not None:
        transformers.append(T.RandomCrop(RANDOM_CROP))
    if RANDOM_VERTICAL_FLIP is not None:
        transformers.append(T.RandomVerticalFlip(RANDOM_VERTICAL_FLIP))
    if CENTRAL_CROP is not None:
        transformers.append(T.CenterCrop(CENTRAL_CROP))
    if RANDOM_RESIZE_CROP is not None:
        transformers.append(T.RandomResizedCrop(RANDOM_RESIZE_CROP))
    if RESIZE is not None:
        transformers.append(T.Resize(RESIZE))

    transformers.append(T.ToTensor())

    transforms = T.Compose(transformers)

    return transforms


def decode_segmap(temp):
    # convert gray scale to color
    # print colored map
    temp = temp.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, NUM_CLASSES):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb


def compute_miou(net, val_dataloader):
    net = net.to(DEVICE)
    net.train(False)  # Set Network to evaluation mode
    jaccard = MulticlassJaccardIndex(num_classes=NUM_CLASSES, ignore_index=255).to(
        DEVICE
    )

    jacc = 0
    count = 0
    for images, labels in val_dataloader:
        images = images.to(DEVICE, dtype=torch.float32)
        labels = labels.to(DEVICE, dtype=torch.long)
        # Forward Pass
        outputs = net(images)
        # Get predictions
        _, preds = torch.max(outputs.data, 1)

        # Update Corrects
        jacc += jaccard(preds, labels.squeeze())
        count += 1

    # Calculate Accuracy
    metric = jacc.item() / count
    # net.train(True)
    return metric


def validation_plot(net, val_dataloader, n_images):
    net = net.to(DEVICE)
    net.train(False)
    rows = 1
    columns = 3
    n = 0
    if not os.path.exists("./results"):
        os.makedirs("./results")
    for imgs, targets in val_dataloader:
        # i = random.randint(BATCH_SIZE)
        if n >= n_images:
            break
        imgsfloat = imgs.to(DEVICE, dtype=torch.float32)
        outputs = net(imgsfloat)
        _, preds = torch.max(outputs.data, 1)
        # Added in order to use the decode_segmap function
        preds = preds.cpu()  # or equally preds = preds.to('cpu')

        for i in range(len(imgs)):
            if n >= n_images:
                break
            # pick the first image of each batch
            # print(imgs[i].shape, targets[i].shape)
            # print(
            #     "img:", imgs[i].squeeze().shape, " target:", targets[i].squeeze().shape
            # )
            # print("pred:", preds.shape)

            figure = plt.figure(figsize=(10, 10))
            figure.add_subplot(rows, columns, 1)
            # plt.imshow(imgs[0].permute((1, 2, 0)).squeeze())
            plt.imshow(imgs[i].permute((1, 2, 0)).squeeze())
            plt.axis("off")
            plt.title("Image")

            figure.add_subplot(rows, columns, 2)
            # plt.imshow(decode_segmap(targets[0].permute((1, 2, 0)).squeeze()))
            plt.imshow(decode_segmap(targets[i]))
            plt.axis("off")
            plt.title("Groundtruth")

            figure.add_subplot(rows, columns, 3)
            # plt.imshow(decode_segmap(preds[0].squeeze()))
            plt.imshow(decode_segmap(preds[i]))
            plt.axis("off")
            plt.title("Prediction")
            plt.savefig(f"./results/validation_{n}.png", transparent=True)
            n += 1
    return


def setup_wandb(args, logger):
    wandb.login()
    transformer_dictionary = {
        "random-horizontal-flip": RANDOM_HORIZONTAL_FLIP,
        "color-jitter": COLOR_JITTER,
        "random-rotation": RANDOM_ROTATION,
        "random-crop": RANDOM_CROP,
        "random-vertical-flip": RANDOM_VERTICAL_FLIP,
        "central-crop": CENTRAL_CROP,
        "random-resized-crop": RANDOM_RESIZE_CROP,
        "resize": RESIZE,
    }

    if args.step == 2:
        config = {
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "num_epochs": NUM_EPOCHS,
            "transformers": transformer_dictionary,
        }
        name = f"Step_2_{ PARTITION}_lr{ LR}_bs{ BATCH_SIZE}_e{ NUM_EPOCHS}"
        project = "STEP2"
    elif args.step == 3:
        config = {
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "momentum": MOMENTUM,
            "num_epochs": NUM_EPOCHS,
            "n_client": CLIENT_PER_ROUND,
            "round": N_ROUND,
            "tot_client": TOT_CLIENTS,
        }
        name = f"Step_3_{ PARTITION}_S{ SPLIT}_R{ N_ROUND}_c{ CLIENT_PER_ROUND}"
        project = "STEP3"
    elif args.step == 4:
        config = {
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "num_epochs": NUM_EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "momentum": MOMENTUM,
            # "step_size": STEP_SIZE,
            "transformers": transformer_dictionary,
            "FDA": FDA,
            "num_styles": N_STYLE,
            "beta_window_size": BETA_WINDOW_SIZE,
        }

        name = f"step4{'_FDA' if  FDA else ''}_{ PARTITION}_S{ SPLIT}_lr{ LR}_bs{ BATCH_SIZE}_e{ NUM_EPOCHS}"

        project = f"STEP{'4.4' if  FDA else '4.2' }"
    elif args.step == 5:
        config = {
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "num_epochs": NUM_EPOCHS,
            "n_client": CLIENT_PER_ROUND,
            "round": N_ROUND,
            "tot_client": TOT_CLIENTS,
            "transformers": transformer_dictionary,
        }

        name = f"Step_5{'_FDA' if  FDA else ''}_T{ T_ROUND}_{ PARTITION}_S{ SPLIT}_R{ N_ROUND}_c{ CLIENT_PER_ROUND}"
        project = "STEP5"

    else:
        raise NotImplementedError

    wandb.init(
        project=project,
        # entity="lor-bellino",
        config=config,
        name=name,
    )
