import copy
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

from .selftrainingloss import SelfTrainingLoss
from config.config_options import *


class Client:
    # def __init__(self, client_id, dataset, model, logger, writer, args, batch_size, world_size, rank, device=None, **kwargs):
    def __init__(self, client_id, dataset, model, pseudo_lab=False, teacher_model=None):
        self.id = client_id
        self.dataset = dataset
        self.model = model  # copy.deepcopy(model)
        self.device = DEVICE
        self.batch_size = BATCH_SIZE
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
        )

        # ADDED
        self.pseudo_lab = pseudo_lab
        self.teacher_model = copy.deepcopy(teacher_model)
        # Define loss function
        if self.pseudo_lab:
            self.criterion = SelfTrainingLoss()
            self.criterion.set_teacher(self.teacher_model)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255)

    def client_train(self):
        num_train_samples = len(self.dataset)

        parameters_to_optimize = self.model.parameters()
        optimizer = optim.SGD(
            parameters_to_optimize, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
        )

        self.model = self.model.to(DEVICE)
        self.model.train()  # Sets module in training mode

        cudnn.benchmark  # Calling this optimizes runtime

        # Start iterating over the epochs
        for epoch in range(NUM_EPOCHS):
            if epoch % LOG_FREQUENCY_EPOCH == 0:
                print("Starting epoch {}/{}".format(epoch + 1, NUM_EPOCHS))

            # Iterate over the dataset
            for current_step, (images, labels) in enumerate(self.loader):
                images = images.to(DEVICE, dtype=torch.float32)
                labels = labels.to(DEVICE, dtype=torch.long)

                optimizer.zero_grad()

                predictions = self.model(images)
                # ADDED
                if self.pseudo_lab:
                    loss = self.criterion(predictions, images.squeeze())
                else:
                    loss = self.criterion(predictions, labels.squeeze())

                # Log loss
                if current_step % LOG_FREQUENCY == 0:
                    print("Step {}, Loss {}".format(current_step, loss.item()))
                    wandb.log({f"client/loss": loss})

                loss.backward()  # backward pass: computes gradients
                optimizer.step()  # update weights based on accumulated gradients

        return num_train_samples, copy.deepcopy(
            self.model.state_dict()
        )  # generate_update

    def test(
        self,
        metrics,
        ret_samples_ids=None,
        silobn_type=None,
        train_cl_bn_stats=None,
        loader=None,
    ):
        return

    def save_model(self, epochs, path, optimizer, scheduler):
        return
