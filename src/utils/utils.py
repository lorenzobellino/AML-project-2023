import torch
import matplotlib.pyplot as plt

from torchmetrics.classification import MulticlassJaccardIndex

import utils.transform as T

from config.transform import *
from config.baseline import *

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


def compute_moiu(net, val_dataloader):
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


def validation_plot(net, val_dataloader, n_image):
    net = net.to(DEVICE)
    net.train(False)
    rows = 1
    columns = 3
    for b, (imgs, targets) in enumerate(val_dataloader):
        if b == n_image:
            break
        # i = random.randint(BATCH_SIZE)
        imgsfloat = imgs.to(DEVICE, dtype=torch.float32)
        outputs = net(imgsfloat)
        _, preds = torch.max(outputs.data, 1)
        # Added in order to use the decode_segmap function
        preds = preds.cpu()  # or equally preds = preds.to('cpu')

        # pick the first image of each batch
        print(imgs[0].shape, targets[0].shape)
        print("img:", imgs[0].squeeze().shape, " target:", targets[0].squeeze().shape)
        print("pred:", preds.shape)

        figure = plt.figure(figsize=(10, 20))
        figure.add_subplot(rows, columns, 1)
        # plt.imshow(imgs[0].permute((1, 2, 0)).squeeze())
        plt.imshow(imgs[0].permute((1, 2, 0)).squeeze())
        plt.axis("off")
        plt.title("Image")

        figure.add_subplot(rows, columns, 2)
        # plt.imshow(decode_segmap(targets[0].permute((1, 2, 0)).squeeze()))
        plt.imshow(decode_segmap(targets[0]))
        plt.axis("off")
        plt.title("Groundtruth")

        figure.add_subplot(rows, columns, 3)
        # plt.imshow(decode_segmap(preds[0].squeeze()))
        plt.imshow(decode_segmap(preds[0]))
        plt.axis("off")
        plt.title("Prediction")
        plt.show()
    return
