import torch
import torch.nn as nn
import torch.nn.functional as F

import abc


class ContinualLerner(nn.Module, metaclass=abc.ABCMeta):
    """Abstract module to add CL capabilities to multitask classifier"""

    def __init__(self):
        super().__init__()
        self.scenario = None
        self.tasks = 0
        self.task_nr = 0
        self.classes_per_task = 0
        self.multihead = None

    @abc.abstractmethod
    def forward(self, x):
        pass


class BaseModel(ContinualLerner, metaclass=abc.ABCMeta):
    """Abstract module to train and test classifier"""

    def __init__(self):
        super().__init__()
        self.optimizer = None

    @abc.abstractmethod
    def forward(self, x):
        pass

    def train_batch(self, X, y, loss_fn, device, active_classes=None):
        # Compute prediction and loss
        X, y = X.to(device), y.to(device)
        self.optimizer.zero_grad()
        pred = self.forward(X)

        if self.scenario == 'class':
            pred = pred[:, :active_classes]

        loss = loss_fn(pred, y)
        # Backpropagation
        loss.backward(retain_graph=False)
        self.optimizer.step()
        return loss


    def train_batch_v2(self, X, y, loss_fn, device, active_classes=None):
        """ Without backprop and weight update"""

        # Compute prediction and loss
        X, y = X.to(device), y.to(device)
        if self.scenario == 'task':
            y = torch.remainder(y, self.classes_per_task)   # changing labels for split dataset task

        pred = self.forward(X)
        if self.scenario == 'class':
            pred = pred[:, :active_classes]

        loss = loss_fn(pred, y)
        return loss


    def train_epoch(self, dataloader, loss_fn, device, task_nr, verbose=True):
        size = len(dataloader.dataset)
        self.task_nr = task_nr

        if self.scenario == 'class':
            # List of classes seen so far
            active_classes = self.classes_per_task * (task_nr + 1)
        else:
            active_classes = None

        self.train()
        total_loss = 0.0
        for batch, (X, y) in enumerate(dataloader):
            if self.scenario == 'task':
                # changing labels for spilt dataset task
                y = torch.remainder(y, self.classes_per_task)

            loss = self.train_batch(X, y, loss_fn, device, active_classes)
            total_loss += loss.item()
            if verbose:
                if batch % 100 == 0:
                    loss_value, current = loss.item(), batch * 64
                    print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

        return total_loss


    def train_epoch_joint(self, MultipleDLs, loss_fn, device, verbose=True):
        self.train()
        for batch_nr, batch in enumerate(MultipleDLs):
            loss_value = 0.0
            losses = []
            for i, dl in enumerate(batch):
                # Unpack values
                X, y = dl[0], dl[1]
                # -------------TRAIN TASK ------------#
                self.task_nr = i
                # Train on batch
                loss = self.train_batch_v2(X, y, loss_fn, device)
                loss_value += loss.item()
                losses.append(loss)

            # Update parameters using combined loss
            self.optimizer.zero_grad()
            total_loss = 0.0
            for loss in losses:
                total_loss += loss
            total_loss.backward(retain_graph=False)
            self.optimizer.step()

            size = len(MultipleDLs.dataloaders[-1].dataset)
            if verbose:
                if batch_nr % 100 == 0:
                    current = batch_nr * 256
                    print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

        return loss_value


    def test_epoch(self, dataloader, loss_fn, device, task_nr, verbose=True):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        self.task_nr = task_nr
        # List of classes seen so far
        active_classes = self.classes_per_task * (task_nr + 1)

        self.eval()
        with torch.no_grad():
            for X, y in dataloader:
                if self.scenario == 'task':
                    # changing labels for spilt dataset task
                    y = torch.remainder(y, self.classes_per_task)

                X, y = X.to(device), y.to(device)
                pred = self.forward(X)

                if self.scenario == 'class':
                    pred = pred[:, :active_classes]

                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        acc = 100 * correct
        if verbose:
            print(f"TASK {task_nr+1:>1d} Test Accuracy: \n Accuracy: {acc:>0.2f}%, Avg loss: {test_loss:>8f} \n")

        return acc, test_loss


class MLP(BaseModel):
    """ Model based on paper BIR """

    def __init__(self, in_size=28, in_channels=1, fc_size=400, classes=10, tasks=2, scenario='task', multihead=True):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Sequential(
            nn.Linear(in_size * in_size * in_channels, fc_size),
            nn.ReLU(),
            nn.Linear(fc_size, fc_size),
            nn.ReLU()
        )
        if multihead:
            # different head for each tasks
            self.outputs = nn.ModuleList([nn.Linear(fc_size, classes) for _ in range(tasks)])
        else:
            self.output = nn.Linear(fc_size, classes * tasks) if scenario == 'class' else nn.Linear(fc_size, classes)

        self.tasks = tasks
        self.classes_per_task = classes
        self.scenario = scenario
        self.multihead = multihead

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden(x)
        if self.multihead:
            # Only head active for current task
            logits = self.outputs[self.task_nr](x)
        else:
            logits = self.output(x)
        return logits


class CNN(BaseModel):
    """ Simple CNN """

    def __init__(self, in_size=28, in_channels=1, c=8, fc_size=128,
                 classes=10, tasks=2, scenario='task', multihead=True):
        super().__init__()
        # 1 input image channel, "c" output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(in_channels, c, 5)
        self.conv2 = nn.Conv2d(c, c*2, 5)

        # an affine operation: y = Wx + b
        img_dim = 4 if in_size == 28 else 5
        self.fc1 = nn.Linear(c*2 * img_dim * img_dim, fc_size)  # 4*4 or 5*5 from image dimension
        self.fc2 = nn.Linear(fc_size, fc_size // 2)

        if multihead:
            # different head for each tasks
            self.outputs = nn.ModuleList([nn.Linear(fc_size // 2, classes) for _ in range(tasks)])
        else:
            self.output = nn.Linear(fc_size // 2, classes * tasks) if scenario == 'class' \
                else nn.Linear(fc_size // 2, classes)

        self.tasks = tasks
        self.scenario = scenario
        self.classes_per_task = classes
        self.multihead = multihead

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        if self.multihead:
            # Only head active for current task
            logits = self.outputs[self.task_nr](x)
        else:
            logits = self.output(x)

        return logits


class DeepCNN(BaseModel):
    """ Simple CNN, implement model from BIR"""

    def __init__(self, in_size=32, in_channels=3, c=16, fc_size=256,
                 classes=10, tasks=2, scenario='task', multihead=True):
        super().__init__()
        self.fc_size = fc_size

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(),

            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(),

            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(),

            nn.Conv2d(c * 4, c * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(),

            nn.Conv2d(c * 8, c * 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 16),
            nn.ReLU()
        )

        self.linear_layers = nn.Sequential(
            nn.Linear((c*16) * 2 * 2, self.fc_size),  # 2*2 from image dimension,
            nn.ReLU(),
            nn.Linear(self.fc_size, self.fc_size),
            nn.ReLU()
        )

        if multihead:
            # different head for each tasks
            self.outputs = nn.ModuleList([nn.Linear(self.fc_size, classes) for _ in range(tasks)])
        else:
            self.output = nn.Linear(self.fc_size, classes * tasks) if scenario == 'class' \
                else nn.Linear(self.fc_size, classes)

        self.tasks = tasks
        self.scenario = scenario
        self.classes_per_task = classes
        self.multihead = multihead


    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = self.linear_layers(x)

        if self.multihead:
            # Only head active for current task
            logits = self.outputs[self.task_nr](x)
        else:
            logits = self.output(x)

        return logits
