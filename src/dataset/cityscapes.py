import json
import os
import numpy as np

import torch.utils.data as torch_data

from torchvision.io import read_image
from torch import from_numpy


class Cityscapes(torch_data.Dataset):

    """
    image path: data/Cityscapes/images/name_leftImg8bit.png
    taget path: data/Cityscapes/labels/name_gtFine_labelIds.png
    """

    def __init__(self, root, transform=None, cl19=False, filename=None, id_client=None):
        eval_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.root = root

        if filename is None:
            raise ValueError("filename is None")

        if id_client is not None:
            with open(os.path.join(root, filename)) as f:
                dict_data = json.load(f)

            self.paths_images = [l[0] for l in dict_data[str(id_client)]]
            self.paths_tagets = [l[1] for l in dict_data[str(id_client)]]

            self.len = len(self.paths_images)
            self.transform = transform
            self.return_unprocessed_image = False

        else:
            with open(os.path.join(root, filename), "r") as f:
                lines = f.readlines()

            # manipulate each file row in order to obtain the correct path
            self.paths_images = [l.strip().split("@")[0] for l in lines]
            self.paths_tagets = [l.strip().split("@")[1] for l in lines]

            self.len = len(self.paths_images)
            self.transform = transform

        if cl19:
            classes = eval_classes
            mapping = np.zeros((256,), dtype=np.int64) + 255
            for i, cl in enumerate(classes):
                mapping[cl] = i
            self.target_transform = lambda x: from_numpy(mapping[x])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the label of segmentation.
        """

        # # using read_image
        img = read_image(os.path.join(self.root, "images", self.paths_images[index]))
        target = read_image(os.path.join(self.root, "labels", self.paths_tagets[index]))

        # if self.return_unprocessed_image:
        #     transform_PIL = T.ToPILImage()
        #     img = transform_PIL(img)
        #     return img

        if self.transform:
            img, target = self.transform(img, target)

        if self.target_transform:
            target = self.target_transform(target)

        return img, target  # output: Tensor[image_channels, image_height, image_width]

        # # using Image.open + np.array
        # img = Image.open(os.path.join(self.root,"images",self.paths_images[index]))
        # target = Image.open(os.path.join(self.root,"labels",self.paths_tagets[index]))

        # return np.array(img), np.array(target) # output: Tensor[image_height, image_width, image_channels]

    def __len__(self):
        return self.len
