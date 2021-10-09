# ------------------------------------------------------------------------------
#    Class to perform network pruning
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn


class Pruner(object):
    """Performs pruning on the given model."""

    def __init__(self, model, prune_perc, previous_masks, train_bias, train_bn, device):
        self.model = model
        self.prune_perc = prune_perc
        self.train_bias = train_bias
        self.train_bn = train_bn

        self.current_masks = None
        self.previous_masks = previous_masks

        valid_key = list(previous_masks.keys())[0]
        self.current_dataset_idx = previous_masks[valid_key].max()

        self.device = device


    def _pruning_mask(self, weights, previous_mask, layer_idx):
        """Ranks weights by magnitude. Sets all below kth to 0.
           Returns pruned mask.
        """
        # Select all prunable weights, ie. belonging to current dataset.
        # previous_mask = previous_mask.cuda()
        previous_mask = previous_mask.to(self.device)
        tensor = weights[previous_mask.eq(self.current_dataset_idx)]
        abs_tensor = tensor.abs()
        cutoff_rank = round(self.prune_perc * tensor.numel())
        # cutoff_value = abs_tensor.view(-1).cpu().kthvalue(cutoff_rank)[0][0]
        cutoff_value = abs_tensor.view(-1).cpu().kthvalue(cutoff_rank)[0].item()

        # Remove those weights which are below cutoff and belong to current
        # dataset that we are training for.
        remove_mask = weights.abs().le(cutoff_value) * previous_mask.eq(self.current_dataset_idx)

        # mask = 1 - remove_mask
        previous_mask[remove_mask.eq(1)] = 0
        mask = previous_mask
        print('Layer #%d, pruned %d/%d (%.2f%%) (Total in layer: %d)' %
              (layer_idx, mask.eq(0).sum(), tensor.numel(),
               100 * mask.eq(0).sum() / tensor.numel(), weights.numel()))
        return mask


    def prune(self):
        """Gets pruning mask for each layer, based on previous_masks.
           Sets the self.current_masks to the computed pruning masks.
        """
        print('Pruning for dataset idx: %d' % self.current_dataset_idx)
        # assert not self.current_masks, 'Current mask is not empty? Pruning twice?'
        self.current_masks = {}

        print('Pruning each layer by removing %.2f%% of values' % (100 * self.prune_perc))

        for module_idx, (name, module) in enumerate(self.model.named_modules()):
            # Multihead outputs don't particpate in pruning procedure
            if name.startswith('outputs'):
                continue
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                mask = self._pruning_mask(module.weight.data, self.previous_masks[module_idx], module_idx)
                # self.current_masks[module_idx] = mask.cuda()
                self.current_masks[module_idx] = mask.to(self.device)
                # Set pruned weights to 0.
                weight = module.weight.data
                weight[self.current_masks[module_idx].eq(0)] = 0.0


    def make_grads_zero(self):
        """Sets grads of fixed weights to 0."""
        # assert self.current_masks
        if self.current_masks is not None:

            for module_idx, (name, module) in enumerate(self.model.named_modules()):
                # Multihead outputs don't particpate in pruning procedure
                if name.startswith('outputs'):
                    continue
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    layer_mask = self.current_masks[module_idx]

                    # Set grads of all weights not belonging to current dataset to 0.
                    if module.weight.grad is not None:
                        module.weight.grad.data[layer_mask.ne(self.current_dataset_idx)] = 0
                        if not self.train_bias:
                            # Biases are fixed.
                            if module.bias is not None:
                                module.bias.grad.data.fill_(0)

                elif 'BatchNorm' in str(type(module)):
                    # Set grads of batchnorm params to 0.
                    if not self.train_bn:
                        module.weight.grad.data.fill_(0)
                        module.bias.grad.data.fill_(0)


    def make_pruned_zero(self):
        """Makes pruned weights 0."""
        # assert self.current_masks
        if self.current_masks is not None:
            for module_idx, (name, module) in enumerate(self.model.named_modules()):
                # Multihead outputs don't particpate in pruning procedure
                if name.startswith('outputs'):
                    continue
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    layer_mask = self.current_masks[module_idx]
                    module.weight.data[layer_mask.eq(0)] = 0.0


    def apply_mask(self, dataset_idx):
        """To be done to retrieve weights just for a particular dataset."""
        for module_idx, (name, module) in enumerate(self.model.named_modules()):
            # Multihead outputs don't particpate in pruning procedure
            if name.startswith('outputs'):
                continue
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                # mask = self.previous_masks[module_idx].cuda()
                mask = self.previous_masks[module_idx].to(self.device)
                weight[mask.eq(0)] = 0.0
                weight[mask.gt(dataset_idx)] = 0.0


    def restore_biases(self, biases):
        """Use the given biases to replace existing biases."""
        for module_idx, (name, module) in enumerate(self.model.named_modules()):
            # Multihead outputs don't particpate in pruning procedure
            if name.startswith('outputs'):
                continue
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if module.bias is not None:
                    module.bias.data.copy_(biases[module_idx])


    def get_biases(self):
        """Gets a copy of the current biases."""
        biases = {}
        for module_idx, (name, module) in enumerate(self.model.named_modules()):
            # Multihead outputs don't particpate in pruning procedure
            if name.startswith('outputs'):
                continue
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if module.bias is not None:
                    biases[module_idx] = module.bias.data.clone()
        return biases


    def make_finetuning_mask(self):
        """Turns previously pruned weights into trainable weights for
           current dataset.
        """
        assert self.previous_masks
        self.current_dataset_idx += 1

        for module_idx, (name, module) in enumerate(self.model.named_modules()):
            # Multihead outputs don't particpate in pruning procedure
            if name.startswith('outputs'):
                continue
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                mask = self.previous_masks[module_idx]
                mask[mask.eq(0)] = self.current_dataset_idx

        self.current_masks = self.previous_masks


    def train_epoch(self, dataloader, loss_fn, device, task_nr, verbose=True):
        """ Train 1 epoch of PackNet, gradients of fixed params are set to '0' """

        size = len(dataloader.dataset)
        self.model.task_nr = task_nr

        if self.model.scenario == 'class':
            # List of classes seen so far
            active_classes = self.model.classes_per_task * (task_nr + 1)
        else:
            active_classes = None

        self.model.train()
        loss_value = 0.0

        for batch, (X, y) in enumerate(dataloader):
            if self.model.scenario == 'task':
                # changing labels for spilt dataset task
                y = torch.remainder(y, self.model.classes_per_task)

            # Train batch
            X, y = X.to(device), y.to(device)
            self.model.optimizer.zero_grad()
            pred = self.model.forward(X)
            if self.model.scenario == 'class':
                pred = pred[:, :active_classes]
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward(retain_graph=False)
            # Set gradients of params with mask idx different than current dataset idx to '0'
            # to avoid updating them !!!
            self.make_grads_zero()
            self.model.optimizer.step()
            self.make_pruned_zero()

            if verbose:
                if batch % 100 == 0:
                    loss_value, current = loss.item(), batch * 64
                    print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

        return loss_value


def init_masks(model):
    """ Initialization of masks as Byte Tensors filled with '1' """
    previous_masks = {}
    for module_idx, (name, module) in enumerate(model.named_modules()):
        # Multihead outputs don't particpate in pruning procedure
        if name.startswith('outputs'):
            continue
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            mask = torch.ByteTensor(module.weight.data.size()).fill_(1)
            if 'cuda' in module.weight.data.type():
                mask = mask.cuda()
            previous_masks[module_idx] = mask
    return previous_masks
