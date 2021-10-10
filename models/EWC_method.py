# ------------------------------------------------------------------------------
#    Elastic Weight Consolidation
# ------------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from torch import autograd
from torch.utils.data import DataLoader
import numpy as np


class EWCMultitask:
    """Class to compute Fisher Information Matrix (FIM) and perform EWC training in Multi Task Setting"""

    def __init__(self, model, crit, weight=1e3):
        self.model = model
        self.weight = weight
        self.crit = crit

    def _update_mean_params(self, task_nr):
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace(".", "__")
            self.model.register_buffer(_buff_param_name + "_estimated_mean_" + str(task_nr), param.data.clone())

    def _update_fisher_params(self, current_ds, batch_size, num_batch, device, task_nr):
        dl = DataLoader(current_ds, batch_size, shuffle=True)
        log_liklihoods = []

        self.model.task_nr = task_nr
        self.model.zero_grad()

        for i, (input, target) in enumerate(dl):
            if i > num_batch:
                break
            if self.model.scenario == "task":
                # changing labels for spilt dataset task
                target = torch.remainder(target, self.model.classes_per_task)
            input, target = input.to(device), target.to(device)
            output = F.log_softmax(self.model(input), dim=1)
            log_liklihoods.append(output[:, target])

        log_likelihood = torch.cat(log_liklihoods).mean()
        grad_log_liklihood = autograd.grad(log_likelihood, self.model.parameters(), allow_unused=True)

        _buff_param_names = [param[0].replace(".", "__") for param in self.model.named_parameters()]
        for _buff_param_name, param in zip(_buff_param_names, grad_log_liklihood):
            if param is not None:
                self.model.register_buffer(
                    _buff_param_name + "_estimated_fisher_" + str(task_nr), param.data.clone() ** 2
                )

    def register_ewc_params(self, dataset, batch_size, num_batches, device, task_nr):
        self._update_fisher_params(dataset, batch_size, num_batches, device, task_nr)
        self._update_mean_params(task_nr)

    def compute_fim(self, task_name, dataset, batch_size, device, task_nr, num_batches=None):
        """Method to register ewc params"""
        # # Out of memory on cuda
        # prev_device = device
        # device = torch.device('cpu')
        # self.model.to(device)

        if num_batches is None:
            fisher_len = len(dataset.sub_indices) if hasattr(dataset, "sub_indices") else 60000
            # Out of memory on CUDA with 4GB if we use too much data to compute FIM
            # Max dataset length is 6_000
            fisher_len = min(fisher_len, 6000)
            num_batches = int(np.floor(fisher_len / batch_size)) - 1
            print("Batches used to compute FIM", num_batches)

        self.register_ewc_params(dataset, batch_size, num_batches, device, task_nr)

        # # Changing device previous device
        # self.model.to(prev_device)

    def _compute_consolidation_loss(self):
        loss = 0.0
        for param_name, param in self.model.named_parameters():
            for t in range(self.model.task_nr):
                _buff_param_name = param_name.replace(".", "__")
                # Get value of mean param if exist
                mean_attr_name = "{}_estimated_mean_".format(_buff_param_name) + str(t)

                if hasattr(self.model, mean_attr_name):
                    estimated_mean = getattr(self.model, mean_attr_name)
                    # Get value of fisher param if exist
                    fisher_attr_name = "{}_estimated_fisher_".format(_buff_param_name) + str(t)

                    if hasattr(self.model, fisher_attr_name):
                        estimated_fisher = getattr(self.model, fisher_attr_name)
                        # Update cumulative loss
                        loss += (self.weight / 2) * (estimated_fisher * ((param - estimated_mean) ** 2)).sum()

        return loss

    def forward_backward_update(self, input, target, combined_loss=True):
        output = self.model(input)
        if combined_loss:
            loss = self.crit(output, target) + self._compute_consolidation_loss()
            loss_value = loss.item()
        else:
            loss1 = self.crit(output, target)
            loss2 = self._compute_consolidation_loss()
            loss_value = loss1.item() + loss2.item()

        self.model.optimizer.zero_grad()
        if combined_loss:
            loss.backward(retain_graph=False)
        else:
            loss1.backward(retain_graph=True)
            loss2.backward(retain_graph=False)
        self.model.optimizer.step()

        return loss_value

    def train_epoch(self, train_dataloader, device, task_nr):
        self.model.task_nr = task_nr
        self.model.train()
        total_loss = 0.0

        # Train EWC
        for inputs, targets in train_dataloader:
            if self.model.scenario == "task":
                # changing labels for spilt dataset task
                targets = torch.remainder(targets, self.model.classes_per_task)
            inputs, targets = inputs.to(device), targets.to(device)
            total_loss += self.forward_backward_update(inputs, targets)

        return total_loss

    def save(self, filename):
        torch.save(self.model, filename)

    def load(self, filename):
        self.model = torch.load(filename)
