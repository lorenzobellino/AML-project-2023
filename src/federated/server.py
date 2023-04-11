import copy
import torch
import numpy as np
import torch.optim as optim

from collections import OrderedDict

from config.config_options import *


class Server:
    # def __init__(self, model, logger, writer, local_rank, lr, momentum, optimizer=None):
    def __init__(self, model, lr=None, momentum=None, pseudo_lab=False):
        self.model = copy.deepcopy(model)
        self.model_params_dict = copy.deepcopy(self.model.state_dict())
        self.selected_clients = []
        self.updates = []
        self.lr = lr
        self.momentum = momentum
        self.optimizer = optim.SGD(params=self.model.parameters(), lr=1, momentum=0.9)
        self.total_grad = 0

        # ADDED
        self.pseudo_lab = pseudo_lab

    def select_clients(self, my_round, possible_clients, num_clients=4):
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round)
        self.selected_clients = np.random.choice(
            possible_clients, num_clients, replace=False
        )

    def set_clients_teacher(self, train_clients):
        for c in train_clients:
            c.criterion.set_teacher(copy.deepcopy(self.model))

    def _compute_client_delta(self, cmodel):
        delta = OrderedDict.fromkeys(cmodel.keys())
        for k, x, y in zip(
            self.model_params_dict.keys(),
            self.model_params_dict.values(),
            cmodel.values(),
        ):
            delta[k] = (
                y - x if "running" not in k and "num_batches_tracked" not in k else y
            )
        return delta

    def train_round(self):
        self.optimizer.zero_grad()

        clients = self.selected_clients

        for i, c in enumerate(clients):
            print(f"CLIENT {i + 1}/{len(clients)} -> {c.id}:")

            c.model.load_state_dict(
                self.model_params_dict
            )  # load_server_model_on_client
            out = c.client_train()

            num_samples, update = out

            update = self._compute_client_delta(update)

            self.updates.append((num_samples, update))
        return

    def _server_opt(self, pseudo_gradient):
        for n, p in self.model.named_parameters():
            p.grad = -1.0 * pseudo_gradient[n]

        self.optimizer.step()

        bn_layers = OrderedDict(
            {
                k: v
                for k, v in pseudo_gradient.items()
                if "running" in k or "num_batches_tracked" in k
            }
        )
        self.model.load_state_dict(bn_layers, strict=False)

    def _aggregation(self):
        total_weight = 0.0
        base = OrderedDict()

        for client_samples, client_model in self.updates:
            total_weight += client_samples
            for key, value in client_model.items():
                if key in base:
                    base[key] += client_samples * value.type(torch.FloatTensor)
                else:
                    base[key] = client_samples * value.type(torch.FloatTensor)
        averaged_sol_n = copy.deepcopy(self.model_params_dict)
        for key, value in base.items():
            if total_weight != 0:
                averaged_sol_n[key] = value.to("cuda") / total_weight

        return averaged_sol_n

    def _get_model_total_grad(self):
        total_norm = 0
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_grad = total_norm**0.5
        return total_grad

    def update_model(self):
        """FedAvg on the clients' updates for the current round.
        Weighted average of self.updates, where the weight is given by the number
        of samples seen by the corresponding client at training time.
        Saves the new central model in self.client_model and its state dictionary in self.model
        """

        averaged_sol_n = self._aggregation()

        self._server_opt(averaged_sol_n)
        self.total_grad = self._get_model_total_grad()
        self.model_params_dict = copy.deepcopy(self.model.state_dict())

        self.updates = []

    def test_model(
        self, clients_to_test, metrics, ret_samples_bool=False, silobn_type=""
    ):
        return
