import os
import numpy as np

from PIL import Image
from torch import from_numpy
from torchvision.datasets import VisionDataset


class GTA5(VisionDataset):
    labels2train = {
        "cityscapes": {
            7: 0,
            8: 1,
            11: 2,
            12: 3,
            13: 4,
            17: 5,
            19: 6,
            20: 7,
            21: 8,
            22: 9,
            23: 10,
            24: 11,
            25: 12,
            26: 13,
            27: 14,
            28: 15,
            31: 16,
            32: 17,
            33: 18,
        },
    }

    def __init__(
        self,
        root,
        transform=None,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        cv2=False,
        target_dataset="cityscapes",
    ):
        assert (
            target_dataset in GTA5.labels2train
        ), f"Class mapping missing for {target_dataset}, choose from: {GTA5.labels2train.keys()}"

        self.labels2train = GTA5.labels2train[target_dataset]

        # super().__init__(root, transform=transform, target_transform=None)

        self.root = root
        self.transform = transform
        self.mean = mean
        self.std = std
        self.cv2 = cv2

        self.target_transform = self.__map_labels()

        self.return_unprocessed_image = False
        self.style_tf_fn = None

        with open(os.path.join(self.root, "train.txt"), "r") as f:
            lines = f.readlines()

        # manipulate each file row in order to obtain the correct path
        self.paths_images = [l.strip() for l in lines]
        # self.paths_tagets = [l for l in lines]

        self.len = len(self.paths_images)

    def set_style_tf_fn(self, style_tf_fn):
        self.style_tf_fn = style_tf_fn

    def reset_style_tf_fn(self):
        self.style_tf_fn = None

    def __getitem__(self, index):
        x_path = os.path.join(self.root, "images", self.paths_images[index])
        y_path = os.path.join(self.root, "labels", self.paths_images[index])

        x = Image.open(x_path)
        y = Image.open(y_path)

        ## using read_image
        # x = read_image(x_path)
        # y = read_image(y_path)

        if self.return_unprocessed_image:
            return x

        if self.style_tf_fn is not None:
            x = self.style_tf_fn(x)

        if self.transform is not None:
            x, y = self.transform(x, y)
        y = self.target_transform(y)

        # # TODO: insert directly in the transform Compose ??
        #         transform_Tensor = ToTensor()
        #         x, y = transform_Tensor(x, y)

        return x, y

    def __len__(self):
        return self.len

    def __map_labels(self):
        mapping = np.zeros((256,), dtype=np.int64) + 255
        for k, v in self.labels2train.items():
            mapping[k] = v
        return lambda x: from_numpy(mapping[x])
