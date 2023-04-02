import copy
import torch
import wandb
import torch.utils.data as torch_data
import torch.nn as nn
import torch.optim as optim

from torchvision.io import read_image
from torch.utils.data import DataLoader
from torch.backends import cudnn

from config.baseline import *
from config.transform import *
from config.federated import *


class Client:
    # def __init__(self, client_id, dataset, model, logger, writer, args, batch_size, world_size, rank, device=None, **kwargs):
    def __init__(self, client_id, dataset, model):
        self.id = client_id
        self.dataset = dataset
        self.model = model  # copy.deepcopy(model)
        self.device = DEVICE
        self.batch_size = BATCH_SIZE
        # self.args = args
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
        )

        # if args.random_seed is not None:
        #     g = torch.Generator()
        #     g.manual_seed(args.random_seed)
        #     self.loader = data.DataLoader(self.dataset, batch_size=self.batch_size, worker_init_fn=seed_worker, num_workers=4, drop_last=True, pin_memory=True, generator=g)
        # else:
        #     self.loader = data.DataLoader(self.dataset, batch_size=self.batch_size, num_workers=4, drop_last=True, pin_memory=True)

        # self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')

    def client_train(self):
        num_train_samples = len(self.dataset)
        # Define loss function
        criterion = nn.CrossEntropyLoss(ignore_index=255)
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
                loss = criterion(predictions, labels.squeeze())

                # Log loss
                if current_step % LOG_FREQUENCY == 0:
                    print("Step {}, Loss {}".format(current_step, loss.item()))
                    # wandb.log({f"client{self.id}/loss":loss})
                    wandb.log({f"client/loss": loss})

                loss.backward()  # backward pass: computes gradients
                optimizer.step()  # update weights based on accumulated gradients

        # return num_train_samples, copy.deepcopy(self.model.state_dict()) #generate_update
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
